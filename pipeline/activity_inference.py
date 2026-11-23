from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from functools import lru_cache
from torchvision.models.video import r3d_18

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models/merl_activity_6class/merl_resnet3d_6class_best.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


@lru_cache(maxsize=1)
def load_activity_model():

    checkpoint = torch.load(
        WEIGHTS,
        map_location=DEVICE,
        weights_only=True
    )

    classes = checkpoint["classes"]

    model = r3d_18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(classes)
    )

    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    return model, classes


def predict_activity(video_path, start_s, end_s):

    model, classes = load_activity_model()

    cap = cv2.VideoCapture(str(video_path))

    try:
        if not cap.isOpened():
            raise ValueError("Unable to open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total <= 0:
            raise ValueError("Invalid video properties")

        start = max(0, int(start_s * fps))
        end = min(total - 1, int(np.ceil(end_s * fps)) - 1)

        if end <= start:
            raise ValueError("Select a longer video interval")

        indices = np.linspace(
            start, end, 16
        ).astype(int)

        frames = []

        for index in indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))

            ok, frame = cap.read()

            if not ok:
                raise ValueError(f"Cannot read frame {index}")

            frame = cv2.resize(frame, (112, 112))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)

    finally:
        cap.release()

    clip = torch.tensor(
        np.stack(frames)
    ).float() / 255.0

    clip = clip.permute(3, 0, 1, 2)

    mean = torch.tensor(
        [0.43216, 0.394666, 0.37645]
    ).view(3, 1, 1, 1)

    std = torch.tensor(
        [0.22803, 0.22145, 0.216989]
    ).view(3, 1, 1, 1)

    clip = (clip - mean) / std

    with torch.inference_mode():

        output = model(
            clip.unsqueeze(0).to(DEVICE)
        )

        probabilities = torch.softmax(
            output, dim=1
        )[0].cpu().numpy()

    index = int(probabilities.argmax())

    return {
        "activity": classes[index],
        "confidence": float(probabilities[index]),
        "probabilities": {
            name: float(prob)
            for name, prob in zip(classes, probabilities)
        }
    }
