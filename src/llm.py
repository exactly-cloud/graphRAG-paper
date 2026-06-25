"""
统一 LLM 接入层（硅基流动 SiliconFlow，OpenAI 兼容接口）

环境变量（.env）:
  SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, LLM_MODEL

用法:
    from src.llm import chat, chat_json
    chat([{"role":"user","content":"你好"}])
    chat_json("从下文抽取三元组...", schema_hint='[{"head":..,"relation":..,"tail":..}]')
"""
from __future__ import annotations

import json
import os
import re
import time

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3.2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.2,
         max_tokens: int = 2048, retries: int = 3, timeout: int = 120) -> str:
    """调用 chat completions，返回 assistant 文本内容。"""
    if not API_KEY:
        raise RuntimeError("缺少 SILICONFLOW_API_KEY，请在 .env 配置")
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(f"{BASE_URL}/chat/completions", json=payload,
                              headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM 调用失败（{retries} 次）: {last_err}")


def _extract_json(text: str):
    """从模型输出中鲁棒提取 JSON（容忍 ```json 代码块、前后多余文字）。"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # 尝试直接解析
    try:
        return json.loads(text)
    except Exception:
        pass
    # 截取第一个 [ 或 { 到最后一个 ] 或 }
    for open_c, close_c in (("[", "]"), ("{", "}")):
        i, j = text.find(open_c), text.rfind(close_c)
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                continue
    raise ValueError(f"无法解析 JSON: {text[:200]}")


def chat_json(prompt: str, system: str = "你是一个严谨的医学信息抽取助手，只输出 JSON。",
              schema_hint: str | None = None, **kw):
    """要求模型返回 JSON 并解析为 Python 对象。"""
    user = prompt if not schema_hint else f"{prompt}\n\n严格按以下 JSON 格式输出，不要多余文字：\n{schema_hint}"
    out = chat([{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
    return _extract_json(out)


def embed(texts: list[str], model: str | None = None, retries: int = 3,
          timeout: int = 120) -> list[list[float]]:
    """文本向量化（OpenAI 兼容 /embeddings）。一次请求一批，返回向量列表。"""
    if not API_KEY:
        raise RuntimeError("缺少 SILICONFLOW_API_KEY，请在 .env 配置")
    payload = {"model": model or EMBED_MODEL, "input": texts}
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(f"{BASE_URL}/embeddings", json=payload,
                              headers=headers, timeout=timeout)
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda x: x["index"])
            return [d["embedding"] for d in data]
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"embedding 调用失败（{retries} 次）: {last_err}")


if __name__ == "__main__":
    print("模型:", DEFAULT_MODEL)
    print(chat([{"role": "user", "content": "用一句话确认你能正常工作。"}]))
