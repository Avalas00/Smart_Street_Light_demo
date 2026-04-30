import base64
import json
import os
from typing import Dict, Optional

import requests


class NullVLM:
    def inspect(self, image_jpeg: bytes, context: Optional[Dict] = None) -> Dict:
        return {
            "enabled": False,
            "alert": False,
            "risk_level": "off",
            "message": "VLM disabled",
            "context": context or {},
        }


class OllamaVLM:
    def __init__(self) -> None:
        self.url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
        self.model = os.getenv("VLM_MODEL", "gemma4:e2b")
        self.timeout = float(os.getenv("VLM_TIMEOUT_SEC", "90"))

    def inspect(self, image_jpeg: bytes, context: Optional[Dict] = None) -> Dict:
        context = context or {}
        prompt = (
            "You are a streetlight edge safety VLM. Inspect one live camera frame. "
            "Judge abnormal pedestrian or road-side behavior: falling/sitting/lying, "
            "person staying in a traffic lane, crowding, obstruction, conflict, unsafe night activity, "
            "or any situation that should alert an operator. "
            "YOLO metadata is provided as JSON. Use the image first, metadata second. "
            "Return JSON only with keys: alert(boolean), risk_level('low'|'medium'|'high'), "
            "abnormal_type(string), message(zh-TW Traditional Chinese short sentence, no Simplified Chinese), "
            "recommended_brightness(integer 0-100), confidence(number 0-1). "
            f"YOLO metadata: {json.dumps(context, ensure_ascii=False)}"
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [base64.b64encode(image_jpeg).decode("ascii")],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 180},
        }
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        raw = response.json().get("response", "").strip()
        parsed = self._parse_json(raw)
        parsed.setdefault("enabled", True)
        parsed.setdefault("raw", raw)
        parsed.setdefault("context", context)
        return parsed

    @staticmethod
    def _parse_json(raw: str) -> Dict:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {
            "alert": True,
            "risk_level": "medium",
            "abnormal_type": "unparsed_vlm_response",
            "message": raw[:160] or "VLM response could not be parsed.",
            "confidence": 0.3,
        }


def build_vlm_client():
    if os.getenv("ENABLE_VLM", "0") != "1":
        return NullVLM()
    return OllamaVLM()
