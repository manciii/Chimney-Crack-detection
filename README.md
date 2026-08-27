# Chimney Damage Detection

A YOLOv8 segmentation model that detects and pixel-level segments structural damage on industrial chimneys — spalling, honeycombing, exposed rebar, and general damage — from photographs.

## Problem

Manual chimney inspection is slow, hazardous, and inconsistent — human inspectors struggle to reliably spot subtle structural damage across large structures. Bounding-box detection also falls short here: it can flag *that* damage exists but not its exact extent. Pixel-level segmentation instead traces the precise outline of each damaged region, enabling accurate damage quantification.

## Approach

- **Model:** YOLOv8n-seg (nano variant), chosen for efficient segmentation on limited hardware.
- **Dataset:** 480 chimney images, annotated with polygon masks in CVAT (Computer Vision Annotation Tool) across 4 damage classes — spalling, honeycombing, exposed rebar, and general damage.
- **Data pipeline:** CVAT XML annotations parsed with Python (`xml.etree`) and converted into normalized YOLO-format polygon label files, split 80% train / 20% validation.
- **Training:** 50 epochs, image size 640, on CPU.

## Results

| Metric | Score |
|---|---|
| mAP50 | 49.42% |
| Precision | 78.35% |
| Recall | 41.52% |
| F1 Score | 52.22% |

The model detects all four damage categories and shows strong generalization for a compact 480-image dataset, though CPU-only training limited iteration speed.

## Repository Contents

- `notebook (1).ipynb`, `notebook (2).ipynb`, `notebook (3) (1).ipynb` — end-to-end pipeline: dataset setup, CVAT-to-YOLO label conversion, train/val split, YOLOv8 training, and evaluation/inference. The three notebooks reflect iterative runs of the same workflow (detection and segmentation variants).
- `Chimney-Damage-Detection.pdf` — project presentation covering the problem statement, methodology, dataset, and results in more detail.

## Pipeline Overview

1. **Annotate** — draw polygon masks around damage regions in CVAT, exported as `annotations.xml`.
2. **Convert** — parse the CVAT XML and write per-image YOLO-format label files (`labels_temp/`).
3. **Split** — shuffle and split images/labels into `dataset/images/{train,val}` and `dataset/labels/{train,val}` (80/20).
4. **Configure** — generate `dataset.yaml` defining the 4 class names and dataset paths.
5. **Train** — fine-tune `yolov8n-seg.pt` for 50 epochs at 640px on CPU.
6. **Evaluate & Predict** — validate with `model.val()` for mAP/precision/recall/F1, and run inference with `model.predict()` on held-out images.

## Future Work

- Expand the dataset beyond 480 images for better generalization.
- Move training to GPU for faster iteration.
- Tune hyperparameters and augmentation strategy.

## Conclusion

A functional chimney damage segmentation system was built using YOLOv8, demonstrating the viability of deep learning for automating structural inspection.

-------
NOTE: Cannot share more data because data privacy has to be maintained
