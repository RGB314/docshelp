"""Multimodal messages: send an image (a customer's error screenshot) alongside text.

Academy: Intro to LangChain M1 L4 "Multimodal Messages".

Concepts:
  * a message's content can be a list of typed blocks: {"type": "text"} and {"type": "image"}
  * images go in as base64 (or a URL) with a MIME type; the same block format works for every provider
  * not every model can see: `get_llm(vision=True)` picks the provider's image-capable model
    (override with DOCSHELP_VISION_MODEL; with Ollama the lesson tells you which model to pull)

Real-world use: customers paste screenshots instead of describing the error.

Run:  uv run poe lesson 1.4
"""

import base64
import io

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from PIL import Image, ImageDraw

from docshelp.config import ensure_llm_ready, get_llm
from docshelp.console import header


def error_screenshot() -> bytes:
    """Draw a fake Nimbus Notes error dialog, so the lesson needs no binary files in the repo."""
    image = Image.new("RGB", (520, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 519, 34], fill=(200, 40, 40))
    draw.text((12, 10), "Nimbus Notes - Sync failed", fill="white")
    draw.text((12, 55), "Error SYNC-403: workspace storage quota exceeded.", fill="black")
    draw.text((12, 80), "Last successful sync: 2 days ago", fill="black")
    draw.rectangle([400, 115, 500, 145], outline="black")
    draw.text((428, 123), "Retry", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def run(llm: BaseChatModel) -> str:
    header("A screenshot + a question in one message")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "My notes stopped syncing. What does this error mean and what should I do?"},
            {"type": "image", "base64": base64.b64encode(error_screenshot()).decode(), "mime_type": "image/png"},
        ]
    )
    answer = llm.invoke([message]).text
    print(answer)
    return answer


def main() -> None:
    ensure_llm_ready(vision=True)
    llm = get_llm(vision=True)
    print(f"Using vision model {getattr(llm, 'model', None) or getattr(llm, 'model_name', '?')}")
    run(llm)


if __name__ == "__main__":
    main()
