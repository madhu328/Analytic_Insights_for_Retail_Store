from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

ACTIONS = [
    "REACH_TO_SHELF",
    "RETRACT_FROM_SHELF",
    "HAND_IN_SHELF",
    "INSPECT_PRODUCT",
    "INSPECT_SHELF"
]

def evaluate(annotation_path, prediction_path, video, output_dir):
    annotations = pd.read_csv(annotation_path)
    predictions = pd.read_csv(prediction_path)

    required_gt = {
        "video", "action", "start_seconds", "end_seconds"
    }
    required_pred = {
        "start_seconds", "end_seconds", "activity"
    }

    if not required_gt.issubset(annotations.columns):
        raise ValueError("Missing ground-truth columns")

    if not required_pred.issubset(predictions.columns):
        raise ValueError("Missing prediction columns")

    gt = annotations.loc[
        annotations["video"] == video
    ].copy()

    if gt.empty:
        raise ValueError(f"No annotations found for {video}")

    gt["action"] = gt["action"].astype(str).str.strip()
    predictions["activity"] = (
        predictions["activity"].astype(str).str.strip()
    )

    unknown = set(gt["action"]) - set(ACTIONS)
    if unknown:
        raise ValueError(f"Unexpected ground-truth labels: {unknown}")

    gt = gt.sort_values("start_seconds")
    predictions = predictions.sort_values("start_seconds")

    duration = float(predictions["end_seconds"].max())

    if duration <= 0:
        raise ValueError("Invalid prediction timeline")

    # Evaluate on a uniform 0.1-second timeline.
    # These are temporal samples, not independent video clips.
    step = 0.1
    times = np.arange(0, duration, step) + step / 2

    gt_labels = []
    pred_labels = []
    ambiguous = 0
    unannotated = 0
    uncovered = 0

    gt_records = gt[
        ["start_seconds", "end_seconds", "action"]
    ].to_records(index=False)

    pred_records = predictions[
        ["start_seconds", "end_seconds", "activity"]
    ].to_records(index=False)

    for t in times:
        matches = [
            str(label)
            for start, end, label in gt_records
            if float(start) <= t < float(end)
        ]

        distinct = set(matches)

        if len(distinct) == 0:
            unannotated += 1
            continue

        if len(distinct) > 1:
            ambiguous += 1
            continue

        predicted = [
            str(label)
            for start, end, label in pred_records
            if float(start) <= t < float(end)
        ]

        if len(predicted) != 1:
            uncovered += 1
            continue

        gt_labels.append(next(iter(distinct)))
        pred_labels.append(predicted[0])

    if not gt_labels:
        raise ValueError("No comparable annotated timestamps found")

    predicted_classes = sorted(
        set(pred_labels) - set(ACTIONS)
    )
    all_labels = ACTIONS + predicted_classes

    report = classification_report(
        gt_labels,
        pred_labels,
        labels=ACTIONS,
        zero_division=0,
        output_dict=True
    )

    matrix = confusion_matrix(
        gt_labels,
        pred_labels,
        labels=all_labels
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=all_labels,
        columns=all_labels
    )

    accuracy = accuracy_score(gt_labels, pred_labels)

    results = {
        "video": video,
        "evaluation_type": "annotated_timeline_sampling",
        "temporal_sampling_seconds": step,
        "ground_truth_source": str(annotation_path),
        "prediction_source": str(prediction_path),
        "annotated_intervals": int(len(gt)),
        "prediction_intervals": int(len(predictions)),
        "video_duration_seconds": duration,
        "evaluated_time_samples": len(gt_labels),
        "evaluated_duration_approx_seconds": (
            round(len(gt_labels) * step, 2)
        ),
        "ambiguous_samples_excluded": ambiguous,
        "unannotated_samples_excluded": unannotated,
        "uncovered_annotated_samples_excluded": uncovered,
        "annotated_action_accuracy": float(accuracy),
        "annotated_action_macro_f1": float(
            report["macro avg"]["f1-score"]
        ),
        "annotation_note": (
            "Only uniquely annotated shopping-action times "
            "are scored. Unannotated gaps are not treated as "
            "verified BACKGROUND. Samples at 0.1-second "
            "spacing are temporally correlated."
        ),
        "classification_report": report
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / "timeline_evaluation.json"
    matrix_file = output_dir / "timeline_confusion_matrix.csv"
    report_file = output_dir / "timeline_classification_report.csv"

    result_file.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    matrix_df.to_csv(matrix_file)

    pd.DataFrame(report).T.to_csv(report_file)

    print("\n===== MERL CONTINUOUS-VIDEO EVALUATION =====")
    print("Video:", video)
    print("Annotated intervals:", len(gt))
    print("Predicted intervals:", len(predictions))
    print("Evaluated temporal samples:", len(gt_labels))
    print("Unannotated samples excluded:", unannotated)
    print("Ambiguous samples excluded:", ambiguous)
    print("Uncovered annotated samples:", uncovered)
    print(f"Annotated-action accuracy: {accuracy * 100:.2f}%")
    print(
        "Annotated-action macro F1:",
        f"{report['macro avg']['f1-score'] * 100:.2f}%"
    )

    print("\nClassification report:")
    print(
        classification_report(
            gt_labels,
            pred_labels,
            labels=ACTIONS,
            zero_division=0
        )
    )

    print("\nConfusion matrix:")
    print(matrix_df.to_string())

    print("\nSaved reports to:", output_dir.resolve())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    evaluate(
        Path(args.annotations),
        Path(args.predictions),
        args.video,
        Path(args.output)
    )
