# app.py
# Requires: transformers>=4.51.0, fastapi, uvicorn, torch
import os
os.environ["HF_HOME"] = "./huggingface_data"

from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

import uvicorn

# APP_TITLE = "Qwen3-Reranker-8B (yes/no) FastAPI"
# MODEL_ID = "Qwen/Qwen3-Reranker-8B"
APP_TITLE = "Qwen3-Reranker-4B (yes/no) FastAPI"
MODEL_ID = "Qwen/Qwen3-Reranker-4B"
# APP_TITLE = "Qwen3-Reranker-0.6B (yes/no) FastAPI"
# MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"

# -------- Model load (once) --------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
use_flash = os.getenv("QWEN_USE_FLASH_ATTENTION", "0") == "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

model_kwargs = {}
if use_flash:
    # pip로 flash-attn 설치 필요 (옵션)
    model_kwargs.update(dict(attn_implementation="flash_attention_2"))

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    **model_kwargs
).to(device).eval()

# yes/no 토큰 아이디 (원본 코드와 동일하게 단일 토큰 가정)
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id  = tokenizer.convert_tokens_to_ids("yes")

# 프롬프트 조각
SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# 미리 토크나이즈
prefix_tokens = tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=False)
suffix_tokens = tokenizer.encode(ASSISTANT_SUFFIX, add_special_tokens=False)

DEFAULT_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"
DEFAULT_MAX_LENGTH = 8192

# -------- FastAPI --------
app = FastAPI(title=APP_TITLE)


# ---------- Schemas ----------
class RerankRequest(BaseModel):
    queries: List[str]
    documents: List[str]
    instruction: Optional[str] = None
    max_length: Optional[int] = None  # 없으면 기본 8192 사용


class RerankResponse(BaseModel):
    scores: List[float]
    used_pairs: List[str]  # 디버깅/재현용(요청 시퀀스 그대로)


# ---------- Core helpers (원본 로직을 함수로 포팅) ----------
def format_instruction(instruction: Optional[str], query: str, doc: str) -> str:
    if instruction is None:
        instruction = DEFAULT_INSTRUCT
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )


def process_inputs(pairs: List[str], max_length: int):
    # prefix/suffix를 고려한 토큰 길이 제한
    available = max_length - len(prefix_tokens) - len(suffix_tokens)
    if available <= 8:
        raise ValueError("max_length가 너무 작습니다. prefix/suffix를 포함해 최소 수십 토큰 이상 필요합니다.")

    inputs = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=available,
    )

    # prefix/suffix 부착
    for i, ids in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ids + suffix_tokens

    # pad 및 텐서 변환
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)

    for key in inputs:
        inputs[key] = inputs[key].to(device)
    return inputs


@torch.no_grad()
def compute_logits(inputs) -> List[float]:
    # 마지막 토큰 로짓에서 yes/no 확률 비교
    logits = model(**inputs, use_cache=False).logits[:, -1, :]
    true_vec = logits[:, token_true_id]
    false_vec = logits[:, token_false_id]
    batch_scores = torch.stack([false_vec, true_vec], dim=1)          # [N, 2]
    batch_scores = torch.nn.functional.log_softmax(batch_scores, 1)   # log-softmax
    scores = batch_scores[:, 1].exp().tolist()                        # "yes" 확률
    return scores


# ---------- Routes ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "device": device,
        "flash_attention": use_flash,
    }


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    if not req.queries or not req.documents:
        raise HTTPException(status_code=400, detail="queries와 documents는 각각 1개 이상이어야 합니다.")
    if len(req.queries) != len(req.documents):
        raise HTTPException(status_code=400, detail="queries와 documents의 길이가 같아야 합니다.")

    pairs = [format_instruction(req.instruction or DEFAULT_INSTRUCT, q, d)
             for q, d in zip(req.queries, req.documents)]

    max_len = int(req.max_length or DEFAULT_MAX_LENGTH)
    try:
        inputs = process_inputs(pairs, max_len)
        with torch.inference_mode():
            scores = compute_logits(inputs)
        response = RerankResponse(scores=scores, used_pairs=pairs)
        return response

    finally:
        # 💡 응답 생성 이후 실행됨 — 메모리 정리
        try:
            del inputs, scores
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



@app.get("/")
def root():
    return {
        "message": "Use POST /rerank with queries & documents.",
        "docs": "/docs",
        "health": "/health",
    }

if __name__ == "__main__":
    uvicorn.run(
        app,        # 모듈이름:객체이름
        host="0.0.0.0",   # 외부 접속 허용
        port=9806,        # 포트
        reload=False       # 코드 변경 시 자동 재시작 (개발용)
    )