# Slum-i Portal

A Streamlit web app for informal settlement detection in satellite imagery using EfficientNet-B5 segmentation models trained on Lahore and Islamabad.

![App screenshot](assets/screenshot.jpg)

## How it works

Draw a bounding box on the map, select a city model, and run analysis. The app tiles the selected area, runs segmentation, and overlays detected slum regions on the original imagery.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Model weights (`.h5`) are not included in the repo. Place them in `models/` before running:
- `enb5_seg_lahore.h5`
- `enb5_seg_islamabad.h5`
- `efficientnetb5_notop.h5`
