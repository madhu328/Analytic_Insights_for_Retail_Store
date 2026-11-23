from pathlib import Path
import json

import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "merl_activity_6class"
DATA_DIR = ROOT / "data" / "merl"


def render_shopping_activity():

    st.header("Shopping Activity Intelligence")

    st.caption(
        "Video-based retail behavior analysis using "
        "ResNet3D-18 and the MERL Shopping Dataset."
    )

    st.info(
        "This module uses recorded shopping videos and human-annotated "
        "activity intervals. It is independent of the selected store's "
        "POS and simulated conversion data."
    )

    results_file = MODEL_DIR / "test_results.json"
    actions_file = DATA_DIR / "MERL_shopping_actions.csv"

    if not results_file.exists() or not actions_file.exists():
        st.warning("MERL evaluation files are missing.")
        return

    results = json.loads(
        results_file.read_text(encoding="utf-8")
    )

    actions = pd.read_csv(actions_file)

    c = st.columns(4)

    c[0].metric("Shopping videos", results["total_videos"])
    c[1].metric("Subject IDs", results["total_subject_ids"])
    c[2].metric("Annotated actions", f"{len(actions):,}")
    c[3].metric("Activity classes", len(results["classes"]))

    st.subheader("Independent model evaluation")

    c = st.columns(4)

    c[0].metric(
        "Test accuracy",
        f"{results['test_accuracy'] * 100:.2f}%"
    )

    c[1].metric(
        "Macro precision",
        f"{results['test_macro_precision'] * 100:.2f}%"
    )

    c[2].metric(
        "Macro recall",
        f"{results['test_macro_recall'] * 100:.2f}%"
    )

    c[3].metric(
        "Macro F1-score",
        f"{results['test_macro_f1'] * 100:.2f}%"
    )

    st.caption(
        f"Independent test clips: {results['test_samples']} | "
        f"Selected epoch: {results['selected_epoch']} | "
        "Subject-independent annotated clip classification"
    )

    st.subheader("Ground-truth shopping behavior")

    summary = actions.groupby("action").agg(
        count=("action", "size"),
        average_duration=("duration_seconds", "mean"),
        total_duration=("duration_seconds", "sum")
    ).reset_index()

    summary["activity"] = (
        summary["action"]
        .str.replace("_", " ")
        .str.title()
    )

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            summary.sort_values("count"),
            x="count",
            y="activity",
            orientation="h",
            title="Annotated activity frequency",
            text="count"
        )

        st.plotly_chart(
            fig, use_container_width=True
        )

    with right:
        fig = px.bar(
            summary.sort_values("average_duration"),
            x="average_duration",
            y="activity",
            orientation="h",
            title="Average activity duration (seconds)"
        )

        st.plotly_chart(
            fig, use_container_width=True
        )

    st.subheader("Shopping engagement insights")

    inspection = actions[
        actions.action == "INSPECT_PRODUCT"
    ]

    shelf = actions[
        actions.action == "INSPECT_SHELF"
    ]

    c = st.columns(4)

    c[0].metric("Product inspections", len(inspection))
    c[1].metric("Shelf inspections", len(shelf))

    c[2].metric(
        "Avg product inspection",
        f"{inspection.duration_seconds.mean():.2f} s"
    )

    c[3].metric(
        "Avg shelf inspection",
        f"{shelf.duration_seconds.mean():.2f} s"
    )

    st.subheader("Model training performance")

    left, right = st.columns(2)

    with left:
        path = MODEL_DIR / "training_curves.png"
        if path.exists():
            st.image(str(path))

    with right:
        path = MODEL_DIR / "confusion_matrix.png"
        if path.exists():
            st.image(str(path))

    st.subheader("Class-wise model performance")

    path = MODEL_DIR / "classification_report.csv"

    if path.exists():
        report = pd.read_csv(path)
        st.dataframe(
            report,
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Predictions on unseen shopping clips")

    preview = MODEL_DIR / "test_prediction_preview.png"

    if preview.exists():
        st.image(str(preview))

    path = MODEL_DIR / "test_predictions.csv"

    if path.exists():

        predictions = pd.read_csv(path)

        selected = st.selectbox(
            "Filter predicted activity",
            ["All"] + results["classes"],
            key="merl_activity_filter"
        )

        if selected != "All":
            predictions = predictions[
                predictions.predicted_action == selected
            ]

        st.dataframe(
            predictions[
                [
                    "video",
                    "action",
                    "predicted_action",
                    "confidence",
                    "correct"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "Source: MERL Shopping Dataset. Activity counts above "
        "come from ground-truth annotations, while test metrics "
        "and predicted activities come from the trained model. "
        "These data do not establish actual purchase conversions."
    )


    st.divider()

    st.subheader("Interactive video analysis")

    st.warning(
        "Select an interval containing one shopping action. "
        "The current model is a five-class action classifier "
        "and does not have a trained background class."
    )

    uploaded = st.file_uploader(
        "Upload overhead shopping footage",
        type=["mp4", "avi", "mov"],
        key="merl_video_upload"
    )

    if uploaded is not None:

        import tempfile
        import cv2
        import sys

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        from pipeline.activity_inference import predict_activity

        if uploaded.name.lower().endswith(".mp4"):
            st.video(uploaded.getvalue())

        with tempfile.TemporaryDirectory() as temp_dir:

            video_path = Path(temp_dir) / (
                "uploaded" + Path(uploaded.name).suffix
            )

            video_path.write_bytes(uploaded.getvalue())

            cap = cv2.VideoCapture(str(video_path))

            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            cap.release()

            if fps <= 0 or frames <= 1:
                st.error("Unable to read uploaded video.")
                return

            duration = frames / fps

            st.caption(f"Video duration: {duration:.2f} seconds")

            start = st.number_input(
                "Start time (seconds)",
                min_value=0.0,
                max_value=float(duration),
                value=0.0,
                step=0.1
            )

            end = st.number_input(
                "End time (seconds)",
                min_value=0.0,
                max_value=float(duration),
                value=min(2.0, duration),
                step=0.1
            )

            if st.button(
                "Analyze shopping activity",
                key="merl_predict_button",
                type="primary"
            ):

                if end <= start:
                    st.error("End time must exceed start time.")

                else:

                    try:
                        with st.spinner("Running ResNet3D-18..."):

                            result = predict_activity(
                                video_path, start, end
                            )

                        c1, c2 = st.columns(2)

                        c1.metric(
                            "Predicted activity",
                            result["activity"].replace("_", " ").title()
                        )

                        c2.metric(
                            "Model confidence",
                            f"{result['confidence'] * 100:.2f}%"
                        )

                        if result["confidence"] < 0.60:
                            st.warning("Low-confidence prediction.")

                        probs = pd.DataFrame({
                            "Activity": [
                                x.replace("_", " ").title()
                                for x in result["probabilities"]
                            ],
                            "Probability": [
                                x * 100
                                for x in result["probabilities"].values()
                            ]
                        })

                        fig = px.bar(
                            probs,
                            x="Probability",
                            y="Activity",
                            orientation="h",
                            title="Activity prediction scores"
                        )

                        st.plotly_chart(
                            fig, use_container_width=True
                        )

                    except Exception as exc:
                        st.error(str(exc))


    st.divider()
    st.subheader("Automatic full-video analysis")

    st.caption(
        "Automatically analyze consecutive video intervals "
        "using the trained ResNet3D-18 model."
    )

    st.info(
        "Experimental window-level predictions. "
        "These are not verified action counts or purchase records."
    )

    auto_video = st.file_uploader(
        "Upload a complete shopping video",
        type=["mp4", "avi", "mov"],
        key="automatic_merl_upload"
    )

    threshold = st.slider(
        "Minimum prediction confidence",
        min_value=0.50,
        max_value=0.95,
        value=0.75,
        step=0.05
    )

    window = st.selectbox(
        "Analysis interval (seconds)",
        [1.0, 2.0, 3.0],
        index=1
    )

    if auto_video is not None:

        if st.button(
            "Analyze complete video",
            key="analyze_full_merl",
            type="primary"
        ):

            import tempfile
            import sys

            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from pipeline.automatic_activity import analyze_full_video

            try:

                with tempfile.TemporaryDirectory() as tmp:

                    video_path = Path(tmp) / (
                        "video" + Path(auto_video.name).suffix
                    )

                    video_path.write_bytes(auto_video.getvalue())

                    progress = st.progress(0.0)

                    result = analyze_full_video(
                        video_path,
                        window_seconds=window,
                        confidence_threshold=threshold,
                        progress_callback=lambda p: progress.progress(p)
                    )

                    progress.progress(1.0)

                st.session_state["merl_auto_results"] = result
                st.session_state["merl_auto_filename"] = auto_video.name

            except Exception as exc:
                st.error(str(exc))

    if "merl_auto_results" in st.session_state:

        result = st.session_state["merl_auto_results"]

        st.caption(
            "Analyzed video: " +
            st.session_state["merl_auto_filename"]
        )

        if result.empty:
            st.warning("No valid intervals analyzed.")
            return

        accepted = result[result.activity != "UNCERTAIN"]

        background = accepted[
            accepted.activity == "BACKGROUND"
        ]

        shopping = accepted[
            accepted.activity != "BACKGROUND"
        ]

        uncertain = result[
            result.activity == "UNCERTAIN"
        ]

        c = st.columns(5)

        c[0].metric("Analyzed intervals", len(result))
        c[1].metric("Shopping intervals", len(shopping))
        c[2].metric("Background intervals", len(background))
        c[3].metric("Uncertain intervals", len(uncertain))

        c[4].metric(
            "Mean confidence",
            f"{result.confidence.mean()*100:.1f}%"
        )

        st.subheader("Predicted activity distribution")

        counts = result.activity.value_counts().reset_index()
        counts.columns = ["Activity", "Intervals"]

        fig = px.bar(
            counts,
            x="Intervals",
            y="Activity",
            orientation="h",
            color="Activity"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Predicted activity duration")

        durations = result.groupby("activity")[
            "duration_seconds"
        ].sum().reset_index()

        fig = px.bar(
            durations,
            x="duration_seconds",
            y="activity",
            color="activity",
            orientation="h"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Automatic activity timeline")

        fig = px.scatter(
            result,
            x="start_seconds",
            y="activity",
            color="activity",
            hover_data=["confidence", "end_seconds"]
        )

        fig.update_layout(
            xaxis_title="Video time (seconds)",
            yaxis_title="Predicted activity"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Detailed prediction records")

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download activity analysis CSV",
            result.to_csv(index=False),
            file_name="shopping_activity_analysis.csv",
            mime="text/csv"
        )
