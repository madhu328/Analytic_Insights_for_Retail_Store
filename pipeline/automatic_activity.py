from pathlib import Path
import cv2
import pandas as pd

from pipeline.activity_inference import predict_activity


def analyze_full_video(
    video_path,
    window_seconds=2.0,
    confidence_threshold=0.75,
    progress_callback=None
):

    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    if fps <= 0 or frames <= 1:
        raise ValueError("Invalid video file")

    duration = frames / fps

    results = []

    start = 0.0

    while start < duration:

        end = min(start + window_seconds, duration)

        if end - start < 0.5:
            break

        prediction = predict_activity(
            video_path, start, end
        )

        confidence = prediction["confidence"]

        activity = prediction["activity"]

        if confidence < confidence_threshold:
            activity = "UNCERTAIN"

        results.append({
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "duration_seconds": round(end - start, 3),
            "activity": activity,
            "confidence": confidence
        })

        start = end

        if progress_callback:
            progress_callback(min(start / duration, 1.0))

    return pd.DataFrame(results)
