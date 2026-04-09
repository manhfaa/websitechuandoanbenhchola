from __future__ import annotations

import json
import re

from services.config import Settings


class LlmAdviceService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, detection: dict, classification: dict) -> dict:
        if not self.settings.openai_api_key:
            return self._fallback_report(classification, "Chưa cấu hình OPENAI_API_KEY.")

        try:
            from openai import OpenAI
        except ModuleNotFoundError:
            return self._fallback_report(classification, "Thiếu thư viện openai trong môi trường.")

        prompt = self._build_prompt(detection, classification)
        client = OpenAI(api_key=self.settings.openai_api_key)

        try:
            response = client.chat.completions.create(
                model=self.settings.openai_model,
                temperature=0.3,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là chuyên gia hỗ trợ nhận diện bệnh lá cây. "
                            "Bạn chỉ được suy luận từ dữ liệu YOLO và CNN do hệ thống cung cấp. "
                            "Không khẳng định chắc chắn 100%, luôn nhắc người dùng quan sát thêm."
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
                "headline": parsed.get("headline", "Đã tạo nhận xét từ ChatGPT."),
                "summary": parsed.get("summary", "").strip(),
                "care_steps": parsed.get("care_steps", []),
                "next_steps": parsed.get("next_steps", []),
                "warning": parsed.get("warning", "").strip(),
            }
        except Exception as exc:
            return self._fallback_report(
                classification,
                f"ChatGPT tạm thời không phản hồi, hệ thống dùng gợi ý mặc định. Chi tiết: {exc}",
            )

    def _build_prompt(self, detection: dict, classification: dict) -> str:
        top_predictions = "\n".join(
            f"- {item['display_label']}: {item['confidence'] * 100:.2f}%"
            for item in classification["top_predictions"]
        )
        return f"""
Hãy trả về JSON hợp lệ với đúng các khóa:
headline, summary, care_steps, next_steps, warning

Yêu cầu:
- Viết bằng tiếng Việt, ngắn gọn, dễ hiểu với người dùng phổ thông.
- summary dài 2-3 câu.
- care_steps là mảng 3-4 ý hành động thực tế.
- next_steps là mảng 2-3 ý quan sát tiếp theo.
- warning là 1 câu nhắc đây chỉ là gợi ý từ mô hình AI.

Dữ liệu đầu vào:
- YOLO tìm thấy lá: {"có" if detection["found"] else "không"}
- Độ tin cậy YOLO: {detection["confidence"] * 100:.2f}%
- Kết quả CNN tốt nhất: {classification["display_label"]}
- Độ tin cậy CNN: {classification["confidence"] * 100:.2f}%
- Top dự đoán:
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
            "headline": f"Kết quả gần nhất: {label}",
            "summary": (
                f"CNN đang nghiêng về lớp '{label}' với độ tin cậy khoảng {confidence:.1f}%. "
                "Bạn nên xem đây là gợi ý ban đầu để kiểm tra lá và điều kiện chăm sóc thực tế."
            ),
            "care_steps": [
                "Tách riêng cây có dấu hiệu bất thường để hạn chế lây lan.",
                "Kiểm tra lại mặt trên, mặt dưới lá và chụp thêm ảnh sáng rõ nếu cần.",
                "Điều chỉnh tưới nước, ánh sáng và độ thông thoáng quanh cây.",
                "Loại bỏ phần lá hư nặng nếu cây đã bị tổn thương rõ rệt.",
            ],
            "next_steps": [
                "Theo dõi sự thay đổi của đốm lá trong 3-5 ngày tiếp theo.",
                "So sánh thêm với ảnh chuẩn hoặc hỏi cán bộ nông nghiệp khi cần.",
            ],
            "warning": reason,
        }
