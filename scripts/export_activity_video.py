from pathlib import Path
import csv
import cv2
import argparse
from bisect import bisect_right

def load_predictions(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    required = {
        "start_seconds",
        "end_seconds",
        "activity",
        "confidence"
    }

    if not rows:
        raise ValueError("Prediction CSV is empty")

    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    predictions = []
    for row in rows:
        start = float(row["start_seconds"])
        end = float(row["end_seconds"])
        confidence = float(row["confidence"])

        if end <= start:
            raise ValueError(f"Invalid interval: {start}, {end}")

        predictions.append({
            "start": start,
            "end": end,
            "activity": row["activity"].strip(),
            "confidence": confidence
        })

    predictions.sort(key=lambda x: x["start"])

    for previous, current in zip(predictions, predictions[1:]):
        if current["start"] < previous["end"] - 1e-6:
            raise ValueError(
                "Overlapping prediction intervals detected"
            )

    return predictions

def annotate_video(video_path, csv_path, output_path):
    predictions = load_predictions(csv_path)
    starts = [p["start"] for p in predictions]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Invalid video metadata")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Cannot initialize MP4 writer")

    frame_no = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            t = frame_no / fps
            idx = bisect_right(starts, t) - 1

            activity = "NOT ANALYZED"
            confidence = None

            if idx >= 0:
                prediction = predictions[idx]

                if prediction["start"] <= t < prediction["end"]:
                    activity = prediction["activity"]
                    confidence = prediction["confidence"]

            if activity == "BACKGROUND":
                color = (200, 160, 60)
            elif activity == "UNCERTAIN":
                color = (0, 180, 255)
            elif activity == "NOT ANALYZED":
                color = (170, 170, 170)
            else:
                color = (70, 210, 100)

            panel_height = min(128, height)
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (0, 0),
                (width, panel_height),
                (20, 20, 20),
                -1
            )
            frame = cv2.addWeighted(
                overlay, 0.72, frame, 0.28, 0
            )

            scale = max(0.45, min(width / 1000, 1.0))
            thickness = max(1, round(scale * 2))

            def draw(text, y, foreground=(255, 255, 255)):
                cv2.putText(
                    frame,
                    text,
                    (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    foreground,
                    thickness,
                    cv2.LINE_AA
                )

            draw("RETAIL SHOPPING ACTIVITY ANALYSIS", 28)
            draw(f"ACTIVITY: {activity}", 61, color)

            if confidence is not None:
                conf_text = f"{confidence * 100:.2f}%"
            else:
                conf_text = "N/A"

            minutes = int(t // 60)
            seconds = t % 60

            draw(
                f"Confidence: {conf_text} | "
                f"Time: {minutes:02d}:{seconds:05.2f}",
                94
            )

            writer.write(frame)
            frame_no += 1

            if frame_no % 900 == 0:
                print(
                    f"Processed {frame_no}/{total} frames",
                    flush=True
                )
    finally:
        capture.release()
        writer.release()

    if frame_no == 0:
        raise RuntimeError("Video contained no readable frames")

    print("\n===== ANNOTATED VIDEO COMPLETE =====")
    print("Input:", video_path)
    print("Predictions:", csv_path)
    print("Frames:", frame_no)
    print("FPS:", round(fps, 2))
    print("Output:", output_path.resolve())
    print("Size MB:", round(
        output_path.stat().st_size / 1048576, 2
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--output",
        default="reports/demo/MERL_annotated_analysis.mp4"
    )
    args = parser.parse_args()

    annotate_video(
        Path(args.video),
        Path(args.predictions),
        Path(args.output)
    )
