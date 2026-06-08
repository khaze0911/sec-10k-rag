"""
src/rag_chain.py

Wires HybridRetriever into a generation chain"
    query → HybridRetriever (BM25 + vector + RRF) → top-K chunks
          → prompt (Mistral chat template) → Mistral-7B (4-bit) → answer
          → return {answer, sources}
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, TypedDict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.retriever import HybridRetriever

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------
"""A single chunk as returned by HybridRetriever.search()

Keys (all present on final search() output):
    chunk_id:    Stable unique id of the chunk (used for citation / dedupe)
    company:     Company display name, e.g. "Coinbase".
    section:     10-K section, e.g. "risk_factors", "mda"
    chunk_index: Position of the chunk within its (company, section)
    text:        The chunk body fed into the prompt context.
    rrf_score:   Fused Reciprocal Rank Fusion score (higher = more relevant)
"""
class Chunk(TypedDict, total=False):
    chunk_id: str
    company: str
    section: str
    chunk_index: int
    text: str
    rrf_score: float

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")

MAX_NEW_TOKENS = 512 # caps generation instead of max_length (prompt + output)

# prompt template
SYSTEM_INSTRUCTION = (
    "You are a financial analyst assistant. Answer the question using ONLY the "
    "context from SEC 10-K filings provided below. If the context does not "
    "contain the answer, say you cannot find it in the provided filings. Cite "
    "the company name when stating a fact. Be concise and factual."
)

PROMPT_BODY = """{system}

Context from SEC 10-K filings:
{context}

Question: {question}

Answer:"""

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------
"""Output contract of the RAG pipeline. FastAPI serializes this to JSON

Attributes:
    answer:  The model's final answer text
    sources: The retrieved chunks used as grounding context, in retrieval order
"""
@dataclass
class RagResult:
    answer: str
    sources: list[Chunk] # API can return chunk_id / company / section per source

# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------
"""
Owns  LLM + retriever for the process lifetime, instantiate one of these at startup and share it
"""
class RagChain:
    """Load the retriever, tokenizer, and quantized LLM once per process

    Args:
        retriever:    An existing HybridRetriever to reuse, if None, a new one is built
        model_name:   HF model id to load. Defaults to the MODEL_NAME env var / Mistral-7B-Instruct-v0.2
        load_in_4bit: Request 4-bit quantization. Only takes effect when CUDA is available; on CPU the model loads fp32 
    Returns:
        None
    Raises:
        OSError / HFValidationError: if model_name can't be downloaded or
            found in MODEL_CACHE_DIR
        RuntimeError: if bitsandbytes/CUDA setup is broken at load time
    """
    def __init__(self, retriever: HybridRetriever | None = None, model_name: str = MODEL_NAME, load_in_4bit: bool = True) ->  None:

        # reuse a passed-in retriever or build
        self.retriever = retriever or HybridRetriever()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        
        # 4-bit quantization via bitsandbytes
        quant_config = None
        if load_in_4bit and torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        
        # accelerate place layers on GPU
        model_kwargs: dict[str, Any] = {"cache_dir": MODEL_CACHE_DIR}
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"
        else:
            # safest fallback for cpu
            model_kwargs["torch_dtype"] = torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # transformer pipline for LangChain
        text_gen = pipeline(
            task="text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, # deterministic
            repetition_penalty=1.1,
            return_full_text=False # return only new tokens
        )

        # LCEL chain: prompt | llm | parser
        self.llm = HuggingFacePipeline(pipeline=text_gen)
        self.prompt = PromptTemplate.from_template(PROMPT_BODY)
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    # helpers
    """Render retrieved chunks into a single prompt-ready context string

    Args:
        chunks: Retrieved chunks in relevance order. Each may contain
                'company', 'section', and 'text'; missing keys fall back to
                safe defaults so a malformed chunk never crashes generation

    Returns:
        A newline-separated block where each chunk is prefixed with a
        numbered "[i] Company — section" header, e.g.:

            [1] Coinbase — risk_factors
            <chunk text>

            [2] Visa — mda
            <chunk text>
    """
    def _format_context(self, chunks: list[Chunk]) -> str:
        blocks: list[str] = []
        for i, c in enumerate(chunks, 1):
            company = c.get("company", "Unkown")
            section = c.get("section", "")
            text = c.get("text", "")
            header = f"[{i}] {company} — {section}".rstrip(" —")
            blocks.append(f"{header}\n{text}")
        return "\n\n".join(blocks)
    
    """Wrap a question in Mistral's chat template control tokens

    Args:
        question: The raw user question (or composed user-turn content)

    Returns:
        The de-tokenized string with Mistral's [INST]…[/INST] wrapping and a
        trailing generation prompt applied by the tokenizer
    """
    def _build_prompt_question(self, question: str) -> str:
        messages = [{"role": "user", "content": question}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    # public API
    
    """Run the full retrieve → format → generate pipeline for one question

    Args:
        question: A natural-language question

    Returns:
        RagResult with the generated answer and the list of source Chunks
        used as grounding context (in retrieval order)

    Raises:
        Propagates retriever errors (e.g. DB connection failures) and
        generation errors; callers at the API layer should map these to
        5xx responses
    """
    def answer(self, question: str) -> RagResult:
        # runs BM25 + vector + RRF returns the top-K fused chunks (default TOP_K=5)
        chunks: list[Chunk] = self.retriever.search(question)

        # build context string from chunks
        context = self._format_context(chunks)

        # generation
        raw: str = self.chain.invoke(
            {
                "system": SYSTEM_INSTRUCTION,
                "context": context,
                "question": question
            }
        )

        return RagResult(answer=raw.strip(), sources=chunks)

# ---------------------------------------------------------------------------
# Module-level singleton accessor (FastAPI-compatable)
# ---------------------------------------------------------------------------
_CHAIN: RagChain | None = None

"""Return the process-wide RagChain singleton, building it on first call

Returns:
    The shared RagChain instance
"""
def get_chain() -> RagChain:
    global _CHAIN
    if _CHAIN is None:
        _CHAIN = RagChain()
    return _CHAIN


# ---------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    
    q = sys.argv[1] if len(sys.argv) > 1 else "What does Visa identify as key risks?"
    print(f"\nQ: {q}\n")
    
    result = get_chain().answer(q)
    print("A:", result.answer)
    print("\n── sources ──")
    for i, s in enumerate(result.sources, 1):
        print(f"  [{i}] {s.get('company','?')} / {s.get('section','?')} "
              f"(chunk {s.get('chunk_index','?')}, id {s.get('chunk_id','?')}, "
              f"rrf {s.get('rrf_score','?')})")
