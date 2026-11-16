# Training your four uploaded retail datasets (Windows / VS Code)

The uploaded archive names are:
- `data set store.zip`
- `data set file.zip`
- `videos.zip`
- `videos dataset.zip`

**No accuracy has been measured yet.** First inventory the contents. A dataset is suitable for supervised person-detection training only if its images have correct human-reviewed bounding-box labels. Other types of data may still be valuable for the application.

## 1. Clone this training branch

```powershell
git clone -b dataset-training-hardening-20261116 https://github.com/madhu328/Analytic_Insights_for_Retail_Store.git
cd Analytic_Insights_for_Retail_Store
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Create `datasets/incoming` and copy all four ZIPs there (do not commit them).

```powershell
New-Item -ItemType Directory -Force datasets\incoming | Out-Null
python -m training.inspect_archives "datasets/incoming/data set store.zip" "datasets/incoming/data set file.zip" "datasets/incoming/videos.zip" "datasets/incoming/videos dataset.zip" --json reports/archive_inventory.json
```

If the ZIP names differ, adjust the four paths.

## 2. Choose the appropriate pipeline from the inventory

- **Images + valid YOLO TXT person boxes:** use the labeled-image training path below.
- **Videos without reviewed labels:** extract diverse frames, pre-label with pretrained YOLO, correct in CVAT, then use the labeled-image training path. Do not report accuracy against auto-generated labels as ground truth.
- **CSV/POS data:** do not feed into YOLO; inspect the columns and load using the POS workflow or use it for forecasting.
- **Other annotations:** convert them carefully to YOLO person class 0, after confirming their coordinate system and label meaning.

Do not mix images from the same video across training and test. If two archives contain the same clip, deduplicate or assign all copies to the same split.

## 3A. Video path (GPU strongly recommended)

```powershell
python -m pip install -r requirements-pipeline.txt
# Extract the source video ZIPs into the videos/ directory with a normal ZIP extractor.
python -m training.extract_frames videos --out datasets/raw/images --every 1.0
python -m training.prelabel datasets/raw/images --out datasets/raw/labels --model yolo11x.pt --imgsz 1280
```

Review/fix every box in CVAT or another annotation editor. Empty files are valid negatives only when the image was reviewed and contains no people.

## 3B. Annotated-image path

Copy the matching images into `datasets/raw/images` and the corrected YOLO TXT files into `datasets/raw/labels`; every image must have one label file with the same stem. Avoid duplicate names across archives. To preserve the source video grouping for splitting, name each frame `uniqueVideoId__000001.jpg` etc.

```powershell
python -m training.audit_dataset --images datasets/raw/images --labels datasets/raw/labels --report reports/dataset_audit.json
python -m training.build_dataset --images datasets/raw/images --labels datasets/raw/labels --out datasets/store_person
```

Inspect `datasets/store_person/dataset_manifest.json` before training: train/val/test must contain independent video groups. The dataset must include at least 3 independent groups for the default split.

## 4. Train and evaluate

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -m training.train_detector --data datasets/store_person/data.yaml --epochs 120 --imgsz 960 --batch -1
```

Outputs:
- `models/store_yolo11m.pt` (best model weights)
- `reports/detector_metrics.json` (validation and held-out test precision, recall, F1, mAP)

Training is computationally expensive. If your machine has no NVIDIA CUDA GPU, use the included `training/colab_train.ipynb` notebook on a GPU runtime, adapting its input paths.

## 5. Run the application

```powershell
python -m pip install -r requirements.txt
python scripts/seed_demo.py --reset
python -m pytest -q
```

In terminal A:
```powershell
uvicorn app.main:app --reload --port 8000
```

In terminal B:
```powershell
$env:API_URL="http://127.0.0.1:8000"
streamlit run dashboard/streamlit_app.py --server.port 8501
```

Visit `http://127.0.0.1:8501`.

For real footage, edit `training/videos_manifest.example.csv` into `training/videos_manifest.csv` using your real video paths, store IDs, camera IDs and start times. Ensure camera IDs match your configured camera layout. Then:
```powershell
python -m training.batch_process training/videos_manifest.csv --model models/store_yolo11m.pt --post http://127.0.0.1:8000
```

To evaluate operational customer counting, manually label independent test videos with `training/label_events_app.py` and compare with `training/evaluate_events.py`. Detection mAP alone is not counting accuracy.

## Security and project hygiene

Keep customer videos, faces, POS records, annotated frames, `.db` files, and trained weights out of the public Git repository unless you have explicit permission to publish them. Large datasets and model weights are git-ignored. GitHub should contain application code, documentation and anonymized demonstration inputs.
