from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple, Optional

import numpy as np
import os
import json


@dataclass
class EvalBatch:
    texts: List[str]
    labels: List[str]


def load_wikitext2(n_eval: int = 512, data_root: Optional[str] = None) -> List[str]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise ImportError("pip install datasets to use evaluation loaders") from e
    # Prefer local
    if data_root:
        local_path = os.path.join(data_root, "wikitext-2-raw", "wiki.valid.raw")
        if os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            return lines[:n_eval]
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [x["text"] for x in ds if len(x["text"]) > 0]
    return texts[:n_eval]


def load_lambada_openai(n_eval: int = 500, data_root: Optional[str] = None) -> EvalBatch:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise ImportError("pip install datasets to use evaluation loaders") from e
    # Prefer local
    if data_root:
        local_jsonl = None
        cand1 = os.path.join(data_root, "lambada", "lambada_test.jsonl")
        cand2 = os.path.join(data_root, "lambada", "test.jsonl")
        for c in (cand1, cand2):
            if os.path.exists(c):
                local_jsonl = c
                break
        if local_jsonl:
            texts: List[str] = []
            labels: List[str] = []
            with open(local_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    s = obj.get("text") or obj.get("sentence") or ""
                    parts = s.strip().split()
                    if len(parts) < 2:
                        continue
                    texts.append(" ".join(parts[:-1]))
                    labels.append(parts[-1])
                    if len(texts) >= n_eval:
                        break
            return EvalBatch(texts=texts, labels=labels)

    # Public HF dataset exposes only the "plain_text" config; keep backward compat
    try:
        ds = load_dataset("lambada", "en", split="validation")
    except ValueError:
        ds = load_dataset("lambada", "plain_text", split="validation")
    texts: List[str] = []
    labels: List[str] = []
    for ex in ds:
        s = ex["text"]
        parts = s.strip().split()
        if len(parts) < 2:
            continue
        texts.append(" ".join(parts[:-1]))
        labels.append(parts[-1])
        if len(texts) >= n_eval:
            break
    return EvalBatch(texts=texts, labels=labels)


def load_gsm8k_tiny(n_eval: int = 500, data_root: Optional[str] = None) -> EvalBatch:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise ImportError("pip install datasets to use evaluation loaders") from e
    # Prefer local
    if data_root:
        local = os.path.join(data_root, "gsm8k", "test.jsonl")
        if os.path.exists(local):
            texts: List[str] = []
            labels: List[str] = []
            with open(local, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        ex = json.loads(line)
                    except Exception:
                        continue
                    prompt = ex.get("question", "")
                    answer = ex.get("answer", "")
                    ans = answer.split("####")[-1].strip()
                    if prompt:
                        texts.append(prompt)
                        labels.append(ans)
                        if len(texts) >= n_eval:
                            break
            return EvalBatch(texts=texts, labels=labels)

    ds = load_dataset("gsm8k", "main", split="test")
    texts: List[str] = []
    labels: List[str] = []
    for ex in ds:
        prompt = ex["question"]
        answer = ex["answer"]
        # Simple normalization: take the final numeric answer after '####'
        ans = answer.split("####")[-1].strip()
        texts.append(prompt)
        labels.append(ans)
        if len(texts) >= n_eval:
            break
    return EvalBatch(texts=texts, labels=labels)

