# SkinCam Advisor

A local Streamlit app that lets you take a webcam selfie, performs a lightweight face/skin signal analysis, and recommends skincare products with Amazon links.

## Features
- Webcam photo capture (`st.camera_input`)
- Face detection using OpenCV Haar cascades
- Basic cosmetic concern scoring (dryness, oiliness, redness, acne-like texture, under-eye darkness)
- Product recommendations mapped to concerns

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the shown local URL in your browser.

## Notes
- This tool is for cosmetic suggestions only, not diagnosis.
- Amazon links are search URLs and may show multiple sellers/products.
