from __future__ import annotations

import json
import re
import time

import requests

from services.config import Settings


class LlmAdviceService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, detection: dict, classification: dict, symptoms: str = "") -> dict:
        if not self.settings.openai_api_key:
            return self._fallback_report(classification, "Chưa cấu hình API key cho dịch vụ AI.")

        prompt = self._build_prompt(detection, classification, symptoms)

        try:
            data = self._post_chat_completion(prompt)
            content = self._extract_content(data)
            parsed = self._parse_json(content, classification)
            return {
                "source": self._provider_name(),
                "model": self.settings.openai_model,
                "headline": parsed.get("headline", "Da tao nhan xet tu AI."),
                "summary": parsed.get("summary", "").strip(),
                "care_steps": parsed.get("care_steps", []),
                "next_steps": parsed.get("next_steps", []),
                "warning": parsed.get("warning", "").strip(),
            }
        except Exception as exc:
            return self._fallback_report(
                classification,
                f"Dịch vụ AI tạm thời không phản hồi, hệ thống dùng gợi ý mặc định. Chi tiết: {exc}",
            )

    def chat(self, message: str) -> dict:
        if not self.settings.openai_api_key:
            return {
                "source": "fallback",
                "model": "local-template",
                "reply": "Chưa cấu hình API key cho chuyên gia nông nghiệp.",
            }

        payload = {
            "model": self.settings.openai_model,
            "temperature": 0.35,
            "max_tokens": 900,
            "response_format": {"type": "json_object"},
            "reasoning": {"effort": "minimal", "exclude": True},
            "include_reasoning": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Bạn là chuyên gia nông nghiệp hỗ trợ người trồng cây tại Việt Nam. "
                        "Trả lời ngắn gọn, thực tế, dễ làm. Không chẩn đoán chắc chắn 100%. "
                        "Luôn khuyên người dùng quan sát thêm hoặc hỏi cán bộ nông nghiệp nếu bệnh nặng. "
                        "Chỉ trả JSON hợp lệ với khóa reply."
                    ),
                },
                {"role": "user", "content": message},
            ],
        }

        try:
            data = self._request_completion(payload)
            content = self._extract_content(data)
            try:
                parsed = json.loads(content.strip())
                reply = str(parsed.get("reply", "")).strip()
            except json.JSONDecodeError:
                reply = content.strip()

            return {
                "source": self._provider_name(),
                "model": self.settings.openai_model,
                "reply": reply or "AI đã phản hồi nhưng nội dung trống. Hãy thử hỏi lại ngắn hơn.",
            }
        except Exception as exc:
            return {
                "source": "fallback",
                "model": "local-template",
                "reply": (
                    "Chuyên gia AI đang bận hoặc bị giới hạn lượt gọi. "
                    f"Bạn vẫn có thể mô tả triệu chứng trong ô phân tích ảnh. Chi tiết: {exc}"
                ),
            }

    def _post_chat_completion(self, prompt: str) -> dict:
        payload = {
            "model": self.settings.openai_model,
            "temperature": 0.3,
            "max_tokens": 1400,
            "response_format": {"type": "json_object"},
            "reasoning": {"effort": "minimal", "exclude": True},
            "include_reasoning": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ban la chuyen gia ho tro nhan dien benh la cay. "
                        "Bạn chỉ được suy luận từ dữ liệu YOLO và CNN do hệ thống cung cấp. "
                        "Không khẳng định chắc chắn 100%, luôn nhắc người dùng quan sát thêm."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        return self._request_completion(payload)

    def _request_completion(self, payload: dict) -> dict:
        deadline = time.monotonic() + 24
        chunks: list[bytes] = []

        with requests.post(
            f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://leafcare-frontend.onrender.com",
                "X-Title": self.settings.app_name,
            },
            json=payload,
            stream=True,
            timeout=(5, 5),
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    chunks.append(chunk)
                if time.monotonic() > deadline:
                    raise TimeoutError("OpenRouter qua 24 giay chua tra xong noi dung.")
                if sum(len(item) for item in chunks) > 128_000:
                    raise ValueError("OpenRouter response qua lon.")

        return json.loads(b"".join(chunks).decode("utf-8"))

    def _provider_name(self) -> str:
        if "openrouter.ai" in self.settings.openai_base_url.lower():
            return "openrouter"
        return "chatgpt"

    def _build_prompt(self, detection: dict, classification: dict, symptoms: str) -> str:
        top_predictions = "\n".join(
            f"- {item['display_label']}: {item['confidence'] * 100:.2f}%"
            for item in classification["top_predictions"]
        )
        symptoms_text = symptoms if symptoms else "Không có mô tả triệu chứng bổ sung."

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
- Triệu chứng người dùng mô tả:
{symptoms_text}
- Top du doan:
{top_predictions}
""".strip()

    def _extract_content(self, response: dict) -> str:
        message = response["choices"][0]["message"]["content"]
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

    def _parse_json(self, content: str, classification: dict) -> dict:
        cleaned = re.sub(r"^```json|```$", "", content.strip(), flags=re.MULTILINE).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "headline": f"Nhan xet AI cho: {classification['display_label']}",
                "summary": cleaned[:900] if cleaned else "AI đã phản hồi nhưng nội dung không đúng định dạng JSON.",
                "care_steps": [
                    "Chụp lại ảnh lá rõ hơn dưới ánh sáng tự nhiên.",
                    "Theo dõi thêm màu sắc, đốm lá và tốc độ lan rộng.",
                    "Cách ly cây có dấu hiệu bất thường nếu nghi bệnh lây lan.",
                ],
                "next_steps": [
                    "Thử lại với ảnh lá thật, rõ nét hơn để AI có thêm dữ liệu.",
                    "Kiểm tra điều kiện tưới nước, độ ẩm và thoáng khí.",
                ],
                "warning": "Nội dung AI không đúng JSON hoàn chỉnh, hệ thống đã rút gọn thành tóm tắt.",
            }
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
