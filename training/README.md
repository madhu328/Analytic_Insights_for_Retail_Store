# Training & accuracy guide: reaching ≥ 95% on your store videos

COCO-pretrained YOLO11m is a strong starting point. Retail CCTV still brings fisheye angles, mirrors, posters of models, mannequins, glare and heavy occlusion at the counter. To push accuracy above 95% you **fine-tune on your own cameras** and **measure on videos the model has never seen**.

```
videos ─▶ 1 extract_frames ─▶ 2 prelabel (YOLO11x) ─▶ review in CVAT ─▶ 3 build_dataset (split by video)
       ─▶ 4 train_detector (YOLO11m fine-tune) ─▶ 5 batch_process (pipeline) ─▶ 6 evaluate_events (vs ground truth)
```

> **No GPU on your laptop?** Use the ready-made Google Colab notebook. It runs steps 2–5 on a free GPU from videos you put in Google Drive:
> [Open training/colab_train.ipynb in Colab](https://colab.research.google.com/github/madhu328/Analytic_Insights_for_Retail_Store/blob/main/training/colab_train.ipynb)

Install once: `pip install -r requirements-pipeline.txt`. A GPU is strongly recommended for step 4; free Colab or Kaggle T4 notebooks work.

## 1. Put the videos in place
```
videos/store_1/cam1.mp4  cam2.mp4  cam3.mp4  cam5.mp4
videos/store_2/...
```
Videos are git-ignored and never committed.

## 2. Extract diverse frames
```bash
python -m training.extract_frames videos/ --out datasets/raw/images --every 1.0
```
This takes one frame per second and skips near-identical frames. Aim for **1,500–3,000 frames** across all cameras, peak and quiet hours included.

## 3. Pre-label, then correct by hand
```bash
python -m training.prelabel datasets/raw/images --out datasets/raw/labels --model yolo11x.pt
```
Import `images/` and `labels/` into **CVAT** or **Label Studio** using the YOLO format. Then:
- **add** missed people: partially occluded shoppers, people at the counter, people near the frame edge
- **delete** false boxes: mirror reflections, posters, mannequins, screens
- leave frames with no people **empty**, so they act as negatives

Export back as YOLO txt into `datasets/raw/labels/`.

## 4. Build a leakage-free dataset
```bash
python -m training.build_dataset --images datasets/raw/images --labels datasets/raw/labels --out datasets/store_person
```
Whole videos go to train, val or test, so the test score reflects unseen footage. A frame-level random split would inflate accuracy.

## 5. Fine-tune YOLO11m
```bash
python -m training.train_detector --data datasets/store_person/data.yaml --epochs 120 --imgsz 960
```
This writes `models/store_yolo11m.pt` and `reports/detector_metrics.json`, which the dashboard shows under **Model accuracy**.

## 6. Run the pipeline with the fine-tuned model
Copy `training/videos_manifest.example.csv` to `training/videos_manifest.csv` and set the real clip start times. Then:
```bash
python -m training.batch_process training/videos_manifest.csv --model models/store_yolo11m.pt --post http://localhost:8000
```

## 7. Measure end-to-end accuracy
Label the real events of 2–3 test clips:
```bash
streamlit run training/label_events_app.py -- --videos videos/
```
Then score the pipeline against them:
```bash
python -m training.evaluate_events --pred data/pipeline_outputs/STORE_BLR_002 --truth training/ground_truth/STORE_BLR_002.csv
```
The report gives per-event precision, recall and F1, counting accuracy (entries, exits, zone visits, queue joins) and dwell accuracy. It is saved to `reports/event_accuracy.json` and shown on the dashboard.

## Checklist when a number is below 95%

| Symptom | Fix |
|---|---|
| Detector **precision** low | Add hard negatives (posters, mirrors, mannequins), or add `exclude_polygons` for that camera in `config/store_layout.json` |
| Detector **recall** low | Label more occluded and small people; raise `--imgsz` to 1280; lower `--conf` to 0.25 |
| **ENTRY/EXIT** off | Re-draw `entry_line` with `run_pipeline --debug-video`; the line should sit across the doorway threshold, and `inside_point` must be inside the store |
| **Zone visits** off | Re-draw the zone polygons on a debug frame; use `anchor: center` for shelf-face polygons and `bottom_center` for floor polygons |
| ID switches (inflated visitor counts) | Raise `track_buffer` in `config/bytetrack_store.yaml`; process at `--fps 8–10` |
| **REENTRY** missed | Install torchreid (OSNet). Tune `--reid-threshold` between 0.75 and 0.85 on a labelled clip |
| Dwell off | Tune `min_enter_s` and `exit_grace_s` (ZoneDwellProcessor) on a labelled clip |

> **About the 95% target:** you can only claim it after measuring on held-out videos with hand-checked labels. These tools produce exactly that measurement. They cannot promise a number before your footage has been labelled and tested.
