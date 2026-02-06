import cv2
import numpy as np
import streamlit as st
from dataclasses import dataclass
from typing import Dict, List

st.set_page_config(page_title="SkinCam Advisor", page_icon="🧴", layout="wide")


@dataclass
class Product:
    name: str
    concern: str
    why: str
    amazon_url: str


PRODUCT_DB: List[Product] = [
    Product(
        name="CeraVe Hydrating Facial Cleanser",
        concern="dryness",
        why="Gentle cleanser with ceramides and hyaluronic acid.",
        amazon_url="https://www.amazon.com/s?k=CeraVe+Hydrating+Facial+Cleanser",
    ),
    Product(
        name="Neutrogena Hydro Boost Water Gel",
        concern="dryness",
        why="Lightweight hydration for dehydrated skin.",
        amazon_url="https://www.amazon.com/s?k=Neutrogena+Hydro+Boost+Water+Gel",
    ),
    Product(
        name="La Roche-Posay Effaclar Medicated Gel Cleanser",
        concern="acne",
        why="Salicylic acid helps clear pores and reduce breakouts.",
        amazon_url="https://www.amazon.com/s?k=La+Roche-Posay+Effaclar+Medicated+Gel+Cleanser",
    ),
    Product(
        name="The Ordinary Niacinamide 10% + Zinc 1%",
        concern="acne",
        why="Helps control oil and reduce appearance of blemishes.",
        amazon_url="https://www.amazon.com/s?k=The+Ordinary+Niacinamide+10%25+Zinc+1%25",
    ),
    Product(
        name="Paula's Choice 2% BHA Liquid Exfoliant",
        concern="oiliness",
        why="BHA exfoliation targets clogged pores and excess oil.",
        amazon_url="https://www.amazon.com/s?k=Paulas+Choice+2%25+BHA+Liquid+Exfoliant",
    ),
    Product(
        name="EltaMD UV Clear SPF 46",
        concern="redness",
        why="Soothing daily sunscreen suitable for redness-prone skin.",
        amazon_url="https://www.amazon.com/s?k=EltaMD+UV+Clear+SPF+46",
    ),
    Product(
        name="La Roche-Posay Cicaplast Baume B5",
        concern="redness",
        why="Calming barrier repair balm for irritation and redness.",
        amazon_url="https://www.amazon.com/s?k=La+Roche-Posay+Cicaplast+Baume+B5",
    ),
    Product(
        name="CeraVe Eye Repair Cream",
        concern="under_eye",
        why="Hydrating eye cream that can help with tired-looking under-eyes.",
        amazon_url="https://www.amazon.com/s?k=CeraVe+Eye+Repair+Cream",
    ),
]


def detect_face_bgr(image_bgr: np.ndarray):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
    return x, y, w, h


def analyze_skin(face_bgr: np.ndarray) -> Dict[str, float]:
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)

    brightness = float(np.mean(hsv[:, :, 2])) / 255.0
    saturation = float(np.mean(hsv[:, :, 1])) / 255.0

    red = face_bgr[:, :, 2].astype(np.float32)
    green = face_bgr[:, :, 1].astype(np.float32)
    blue = face_bgr[:, :, 0].astype(np.float32)
    redness = float(np.mean((red - (green + blue) / 2.0) / 255.0))

    # Texture proxy: edge density can indicate blemish activity.
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 140)
    edge_density = float(np.mean(edges > 0))

    h, w, _ = face_bgr.shape
    eye_band = gray[int(0.35 * h): int(0.55 * h), int(0.15 * w): int(0.85 * w)]
    cheek_band = gray[int(0.55 * h): int(0.85 * h), int(0.2 * w): int(0.8 * w)]
    under_eye_darkness = float(np.mean(cheek_band) - np.mean(eye_band)) / 255.0 if eye_band.size and cheek_band.size else 0.0

    return {
        "brightness": brightness,
        "saturation": saturation,
        "redness": redness,
        "edge_density": edge_density,
        "under_eye_darkness": max(0.0, under_eye_darkness),
        "uniformity": float(np.std(lab[:, :, 0])) / 128.0,
    }


def infer_concerns(metrics: Dict[str, float]) -> Dict[str, float]:
    concerns = {
        "dryness": max(0.0, 0.62 - metrics["brightness"] + (0.28 - metrics["saturation"]) * 0.6),
        "oiliness": max(0.0, metrics["saturation"] - 0.38 + metrics["brightness"] * 0.15),
        "acne": max(0.0, metrics["edge_density"] * 2.5 + metrics["uniformity"] * 0.4 - 0.12),
        "redness": max(0.0, metrics["redness"] * 2.2),
        "under_eye": max(0.0, metrics["under_eye_darkness"] * 2.0),
    }
    # Normalize to 0..1 for readability
    return {k: float(min(1.0, v)) for k, v in concerns.items()}


def get_recommendations(concerns: Dict[str, float], top_n: int = 3) -> List[Product]:
    ranked = sorted(concerns.items(), key=lambda kv: kv[1], reverse=True)
    selected = [c for c, score in ranked if score > 0.08][:2]
    if not selected:
        selected = [ranked[0][0]]

    results: List[Product] = []
    for concern in selected:
        results.extend([p for p in PRODUCT_DB if p.concern == concern])

    return results[:top_n]


st.title("🧴 SkinCam Advisor")
st.caption("Take a webcam photo, get a lightweight skin analysis, and see Amazon product ideas.")

st.warning(
    "This app gives cosmetic suggestions only. It is not medical advice. "
    "For persistent or severe skin issues, consult a dermatologist."
)

with st.expander("How this works"):
    st.write(
        "The app detects the face in your photo, computes basic image signals (brightness, redness, texture), "
        "then maps them to common skincare concerns."
    )

image_file = st.camera_input("Take a clear selfie in neutral lighting")

if image_file is not None:
    bytes_data = image_file.getvalue()
    np_arr = np.frombuffer(bytes_data, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    face = detect_face_bgr(image_bgr)
    if face is None:
        st.error("No face detected. Please retake photo with your face centered and well lit.")
    else:
        x, y, w, h = face
        overlay = image_bgr.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 200, 0), 2)
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Detected face", use_container_width=True)

        face_crop = image_bgr[y:y + h, x:x + w]
        metrics = analyze_skin(face_crop)
        concerns = infer_concerns(metrics)
        recs = get_recommendations(concerns)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Analysis summary")
            for label, score in sorted(concerns.items(), key=lambda kv: kv[1], reverse=True):
                st.progress(min(1.0, score), text=f"{label.replace('_', ' ').title()}: {score:.2f}")

        with c2:
            st.subheader("Recommended products")
            for p in recs:
                st.markdown(f"**{p.name}**  ")
                st.markdown(f"Why: {p.why}  ")
                st.markdown(f"[View on Amazon]({p.amazon_url})")
                st.divider()
