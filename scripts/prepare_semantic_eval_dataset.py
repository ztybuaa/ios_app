import json
import subprocess
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from env_guard import ensure_project_venv


ensure_project_venv()

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "processed" / "eval" / "semantic_image_retrieval"
MANIFEST_PATH = DATASET_ROOT / "manifest.json"
IMAGE_ROOT = DATASET_ROOT / "images"


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                payload = response.read()
            destination.write_bytes(payload)
            return
        except Exception as error:
            last_error = error
            time.sleep(1 + attempt)

    curl = subprocess.run(
        ["curl.exe", "-L", "--retry", "3", "--retry-delay", "2", "-o", str(destination), url],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if curl.returncode != 0:
        raise SystemExit(
            f"Failed to download {url}\n"
            f"urllib error: {last_error}\n"
            f"curl error: {curl.stderr}"
        )


def generate_game_screenshot(destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)

    canvas = Image.new("RGB", (540, 900), (34, 18, 54))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=42)
    label_font = ImageFont.load_default(size=24)
    button_font = ImageFont.load_default(size=30)

    for y in range(canvas.height):
        blue = min(255, 54 + int(40 * y / canvas.height))
        draw.line((0, y, canvas.width, y), fill=(34, 18, blue))

    draw.rectangle((0, 0, 540, 120), fill=(68, 32, 42))
    draw.text((38, 28), "BATTLE QUEST", font=title_font, fill=(180, 238, 245))
    draw.text((350, 82), "LEVEL 12", font=label_font, fill=(255, 220, 140))

    draw.rectangle((38, 150, 502, 490), fill=(108, 76, 52))
    draw.rectangle((58, 180, 215, 430), fill=(178, 120, 70))
    draw.rectangle((325, 180, 482, 430), fill=(102, 72, 118))
    draw.ellipse((84, 218, 188, 322), fill=(255, 205, 80))
    draw.ellipse((352, 218, 456, 322), fill=(110, 110, 250))
    draw.line((225, 300, 315, 245), fill=(105, 230, 255), width=8)
    draw.line((235, 345, 318, 345), fill=(105, 230, 255), width=8)

    draw.rectangle((45, 520, 495, 560), fill=(72, 55, 55))
    draw.rectangle((45, 520, 360, 560), fill=(95, 190, 70))
    draw.text((62, 526), "HP 76/100", font=label_font, fill="white")

    draw.rectangle((45, 595, 245, 690), fill=(180, 100, 65))
    draw.rectangle((295, 595, 495, 690), fill=(80, 80, 160))
    draw.text((83, 625), "ATTACK", font=button_font, fill="white")
    draw.text((355, 625), "SKILL", font=button_font, fill="white")

    draw.rectangle((30, 735, 510, 850), fill=(52, 35, 38))
    draw.text((68, 760), "VICTORY REWARD", font=button_font, fill=(120, 220, 255))
    draw.text((68, 812), "coins 1280   rank S", font=label_font, fill=(255, 230, 210))
    canvas.save(destination, format="PNG")


def validate_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, ValueError) as error:
        raise SystemExit(f"Invalid image fixture: {path}")


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    for image in manifest["images"]:
        destination = IMAGE_ROOT / image["file"]
        if image.get("generated") == "pillow":
            generate_game_screenshot(destination)
        elif "url" in image:
            download(image["url"], destination)
        else:
            raise SystemExit(f"Image fixture has neither url nor generator: {image['id']}")
        validate_image(destination)
        print(f"ready {image['id']}: {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
