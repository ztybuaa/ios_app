import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np

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
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    destination.write_bytes(payload)


def generate_game_screenshot(destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.zeros((900, 540, 3), dtype=np.uint8)
    canvas[:] = (24, 18, 34)

    for y in range(canvas.shape[0]):
        color = int(30 + 40 * y / canvas.shape[0])
        canvas[y, :, 0] = np.clip(canvas[y, :, 0] + color, 0, 255)

    cv2.rectangle(canvas, (0, 0), (540, 120), (42, 32, 68), -1)
    cv2.putText(canvas, "BATTLE QUEST", (38, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.45, (245, 238, 180), 3)
    cv2.putText(canvas, "LEVEL 12", (350, 107), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 220, 255), 2)

    cv2.rectangle(canvas, (38, 150), (502, 490), (52, 76, 108), -1)
    cv2.rectangle(canvas, (58, 180), (215, 430), (70, 120, 178), -1)
    cv2.rectangle(canvas, (325, 180), (482, 430), (118, 72, 102), -1)
    cv2.circle(canvas, (136, 270), 52, (80, 205, 255), -1)
    cv2.circle(canvas, (404, 270), 52, (250, 110, 110), -1)
    cv2.line(canvas, (225, 300), (315, 245), (255, 230, 105), 8)
    cv2.line(canvas, (235, 345), (318, 345), (255, 230, 105), 8)

    cv2.rectangle(canvas, (45, 520), (495, 560), (55, 55, 72), -1)
    cv2.rectangle(canvas, (45, 520), (360, 560), (70, 190, 95), -1)
    cv2.putText(canvas, "HP 76/100", (62, 548), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    cv2.rectangle(canvas, (45, 595), (245, 690), (65, 100, 180), -1)
    cv2.rectangle(canvas, (295, 595), (495, 690), (160, 80, 80), -1)
    cv2.putText(canvas, "ATTACK", (83, 655), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(canvas, "SKILL", (355, 655), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.rectangle(canvas, (30, 735), (510, 850), (38, 35, 52), -1)
    cv2.putText(canvas, "VICTORY REWARD", (68, 790), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 120), 2)
    cv2.putText(canvas, "coins 1280   rank S", (68, 830), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 230, 255), 2)

    success, encoded = cv2.imencode(".png", canvas)
    if not success:
        raise SystemExit(f"Failed to encode generated image: {destination}")
    encoded.tofile(destination)


def validate_image(path: Path) -> None:
    payload = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Invalid image fixture: {path}")


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    for image in manifest["images"]:
        destination = IMAGE_ROOT / image["file"]
        if image.get("generated") == "opencv":
            generate_game_screenshot(destination)
        elif "url" in image:
            download(image["url"], destination)
        else:
            raise SystemExit(f"Image fixture has neither url nor generator: {image['id']}")
        validate_image(destination)
        print(f"ready {image['id']}: {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
