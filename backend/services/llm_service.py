from __future__ import annotations

import json
import re

from services.config import Settings


class LlmAdviceService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, detection: dict, classification: dict, symptoms: str = "") -> dict:
        if not self.settings.openai_api_key:
            return self._fallback_report(classification, "Chua cau hinh OPENAI_API_KEY.")

        try:
            from openai import OpenAI
        except ModuleNotFoundError:
            return self._fallback_report(classification, "Thieu thu vien openai trong moi truong.")

        prompt = self._build_prompt(detection, classification, symptoms)
        client = OpenAI(api_key=self.settings.openai_api_key)

        try:
            response = client.chat.completions.create(
                model=self.settings.openai_model,
                temperature=0.3,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ban la chuyen gia ho tro nhan dien benh la cay. "
                            "Ban chi duoc suy luan tu du lieu YOLO va CNN do he thong cung cap. "
                            "Khong khang dinh chac chan 100%, luon nhac nguoi dung quan sat them."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = self._extract_content(response)
            parsed = self._parse_json(content)
            return {
                "source": "chatgpt",
                "model": self.settings.openai_model,
                "headline": parsed.get("headline", "Da tao nhan xet tu ChatGPT."),
                "summary": parsed.get("summary", "").strip(),
                "care_steps": parsed.get("care_steps", []),
                "next_steps": parsed.get("next_steps", []),
                "warning": parsed.get("warning", "").strip(),
            }
        except Exception as exc:
            return self._fallback_report(
                classification,
                f"ChatGPT tam thoi khong phan hoi, he thong dung goi y mac dinh. Chi tiet: {exc}",
            )

    def _build_prompt(self, detection: dict, classification: dict, symptoms: str) -> str:
        top_predictions = "\n".join(
            f"- {item['display_label']}: {item['confidence'] * 100:.2f}%"
            for item in classification["top_predictions"]
        )
        symptoms_text = symptoms if symptoms else "Khong co mo ta trieu chung bo sung."

        return f"""
Hay tra ve JSON hop le voi dung cac khoa:
headline, summary, care_steps, next_steps, warning

Yeu cau:
- Viet bang tieng Viet, ngan gon, de hieu voi nguoi dung pho thong.
- summary dai 2-3 cau.
- care_steps la mang 3-4 y hanh dong thuc te.
- next_steps la mang 2-3 y quan sat tiep theo.
- warning la 1 cau nhac day chi la goi y tu mo hinh AI.

Du lieu dau vao:
- YOLO tim thay la: {"co" if detection["found"] else "khong"}
- Do tin cay YOLO: {detection["confidence"] * 100:.2f}%
- Ket qua CNN tot nhat: {classification["display_label"]}
- Do tin cay CNN: {classification["confidence"] * 100:.2f}%
- Trieu chung nguoi dung mo ta:
{symptoms_text}
- Top du doan:
{top_predictions}
""".strip()

    def _extract_content(self, response) -> str:
        message = response.choices[0].message.content
        if isinstance(message, str):
            return message
        if isinstance(message, list):
            parts = []
            for item in message:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                else:
                    parts.append(getattr(item, "text", ""))
            return "\n".join(part for part in parts if part)
        return str(message)

    def _parse_json(self, content: str) -> dict:
        cleaned = re.sub(r"^```json|```$", "", content.strip(), flags=re.MULTILINE).strip()
        data = json.loads(cleaned)
        return {
            "headline": str(data.get("headline", "")).strip(),
            "summary": str(data.get("summary", "")).strip(),
            "care_steps": [str(item).strip() for item in data.get("care_steps", []) if str(item).strip()],
            "next_steps": [str(item).strip() for item in data.get("next_steps", []) if str(item).strip()],
            "warning": str(data.get("warning", "")).strip(),
        }

    def _fallback_report(self, classification: dict, reason: str) -> dict:
        label = classification["display_label"]
        confidence = classification["confidence"] * 100
        return {
            "source": "fallback",
            "model": "local-template",
            "headline": f"Ket qua gan nhat: {label}",
            "summary": (
                f"CNN dang nghieng ve lop '{label}' voi do tin cay khoang {confidence:.1f}%. "
                "Ban nen xem day la goi y ban dau de kiem tra la va dieu kien cham soc thuc te."
            ),
            "care_steps": [
                "Tach rieng cay co dau hieu bat thuong de han che lay lan.",
                "Kiem tra lai mat tren, mat duoi la va chup them anh sang ro neu can.",
                "Dieu chinh tuoi nuoc, anh sang va do thong thoang quanh cay.",
                "Loai bo phan la hu nang neu cay da bi ton thuong ro ret.",
            ],
            "next_steps": [
                "Theo doi su thay doi cua dom la trong 3-5 ngay tiep theo.",
                "So sanh them voi anh chuan hoac hoi can bo nong nghiep khi can.",
            ],
            "warning": reason,
        }
