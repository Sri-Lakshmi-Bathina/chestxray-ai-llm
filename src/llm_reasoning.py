import os
from typing import List, Dict

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

_LOCAL_PIPE = None  # cached HF pipeline

def _format_findings(topk: List[Dict]) -> str:
    if not topk:
        return "No predicted findings available."
    lines = [f"- {t.get('label','?')}: {float(t.get('prob',0.0)):.2f}" for t in topk[:6]]
    return "Predicted findings (label:prob):\n" + "\n".join(lines)

def _reasoning_prompt(topk: List[Dict]):
    system = ("You are a radiology teaching assistant. Explain chest X-ray findings for education only — "
              "not medical advice or diagnosis.")
    user = (_format_findings(topk) +
            "\n\nWrite a concise paragraph (3–6 sentences) explaining what these findings could mean, "
            "typical causes, and what a clinician might look for next. Do not diagnose.")
    return system, user

def _precautions_prompt(topk: List[Dict]):
    system = "You list cautious, general next-step considerations. Not medical advice."
    user = (_format_findings(topk) +
            "\n\nList 3–5 general, non-diagnostic precautions or next steps (e.g., compare with prior CXRs, "
            "clinical correlation, additional views).")
    return system, user

def _get_local_pipe():
    from transformers import pipeline
    global _LOCAL_PIPE
    if _LOCAL_PIPE is None:
        model_id = os.getenv("HF_LOCAL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        max_new = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "300"))
        _LOCAL_PIPE = pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",          # uses MPS on Apple Silicon if available
            max_new_tokens=max_new,
            do_sample=False,
        )
    return _LOCAL_PIPE

def _generate_local(system: str, user: str) -> str:
    pipe = _get_local_pipe()
    prompt = f"[SYSTEM]{system}\n[USER]{user}\n[ASSISTANT]"
    out = pipe(prompt)[0]["generated_text"]
    return out.split("[ASSISTANT]")[-1].strip()

def run_reasoning_and_precautions(topk: List[Dict]):
    rs, ru = _reasoning_prompt(topk)
    ps, pu = _precautions_prompt(topk)
    reasoning = _generate_local(rs, ru)
    precautions = _generate_local(ps, pu)
    return {"reasoning": reasoning, "precautions": precautions}
