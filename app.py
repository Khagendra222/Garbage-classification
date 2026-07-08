import os
import json

import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Garbage Classification", page_icon="♻️", layout="wide")

MANAGEMENT_GUIDANCE = {
    "plastic": [
        "Place rinsed plastic bottles and containers in the recycling bin if accepted locally.",
        "Check resin codes (usually on the bottom) — many programs accept PET (1) and HDPE (2).",
        "Remove heavy contamination (food/oil) — contaminated items may be rejected.",
        "Do not mix plastic bags with curbside recyclables; return bags to store drop-off if available."
    ],
    "paper": [
        "Keep paper and cardboard dry and free from food or grease.",
        "Flatten cardboard boxes and cut large pieces to fit your bin.",
        "Shredded paper may need to be contained in a paper bag depending on local rules.",
        "Remove plastic windows from envelopes and discard them separately."
    ],
    "metal": [
        "Empty and rinse cans and metal containers before recycling.",
        "Small metal items (foil, trays) are often recyclable — check local guidance.",
        "Large metal objects or appliances should be taken to designated scrap/recycling centers."
    ],
    "glass": [
        "Rinse glass containers and remove lids (lids may be recycled separately).",
        "Do not recycle broken glass, ceramics, or light bulbs with container glass unless specified.",
        "Use local drop-off points when curbside glass recycling is not available."
    ],
    "cardboard": [
        "Break down and flatten boxes to save space and improve collection efficiency.",
        "Keep cardboard dry; wet or greasy cardboard (pizza boxes) should go to organics or trash if contaminated.",
        "Remove packing materials such as foam or bubble wrap before recycling."
    ],
    "trash": [
        "Separate hazardous waste (batteries, electronics, chemicals) and use local hazardous waste programs.",
        "Compost food and yard waste where possible to reduce landfill disposal.",
        "Consider reducing and reusing items before disposal to minimize waste generation."
    ],
}

RECYCLE_TIPS = {
    "plastic": [
        "Empty and rinse containers to remove food residues.",
        "Remove caps and labels if your local program requires it (some accept caps).",
        "Crush or flatten bottles to save space in bins.",
        "Do not place recyclables in plastic bags unless your program specifically allows it.",
        "Check your local council for accepted plastic types (PET, HDPE, etc.)."
    ],
    "glass": [
        "Rinse bottles and jars; remove lids (lids may be recycled separately).",
        "Separate colors only if your local program asks for it.",
        "Do NOT include ceramics, Pyrex, or light bulbs with container glass.",
        "If curbside pickup is unavailable, take glass to a local recycling drop-off."
    ],
    "metal": [
        "Empty and rinse cans to remove food residue.",
        "Flatten cans when possible to save space.",
        "Remove non-metal attachments (plastic windows, labels) if required.",
        "Small scrap metal and large appliances may need to go to special collection points."
    ],
    "paper": [
        "Keep paper and cardboard dry and free of food/oil contamination.",
        "Flatten cardboard boxes to save space and improve sorting.",
        "Remove non-paper inserts (plastic windows, foam) before recycling.",
        "Shredded paper can be recycled in many programs but check local rules (may require bagging)."
    ],
    "cardboard": [
        "Break down boxes and fold them flat.",
        "Remove packing materials (plastic, foam) before recycling.",
        "Keep cardboard dry; wet/soiled cardboard (pizza boxes with grease) should go to organic waste if available."
    ],
    "trash": [
        "Consider composting food and garden waste when possible instead of sending to landfill.",
        "Dispose hazardous items (batteries, electronics, chemicals) through designated hazardous waste programs.",
        "Reduce and reuse where possible to avoid creating waste in the first place."
    ],
}


def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


@st.cache_resource
def load_model_and_classes():
    try:
        from tensorflow.keras.models import load_model
    except Exception:
        return None, None

    model_path = os.path.join("saved_model", "garbage_classifier.keras")
    classes_path = os.path.join("saved_model", "class_names.json")

    if not os.path.exists(model_path):
        return None, None

    model = load_model(model_path)
    class_names = None
    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            class_names = json.load(f)

    return model, class_names


def preprocess_image(img: Image.Image, target_size=(224, 224)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32")
    try:
        from tensorflow.keras.applications import efficientnet
        arr = efficientnet.preprocess_input(arr)
    except Exception:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, class_names, img: Image.Image):
    if model is None:
        return None, None

    x = preprocess_image(img)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    label = class_names[idx] if class_names else str(idx)
    confidence = float(preds[idx])
    return label, confidence


def load_metrics(path=os.path.join("saved_model", "metrics.json")):
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


def sidebar_nav():
    st.sidebar.title("app")
    pages = ["Prediction", "About Dataset", "About Project", "Model Performance"]
    return st.sidebar.radio("", pages)


def render_prediction_page(model, class_names):
    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        st.subheader("Uploaded Image")
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="uploaded_image")
        if uploaded:
            st.write("Filename:", uploaded.name)
            img = Image.open(uploaded)
            st.image(img, use_container_width=True)

            if st.button("Upload Another Image"):
                st.session_state.uploaded_image = None
                safe_rerun()
        else:
            st.info("Upload an image to preview and classify it.")

    with col2:
        st.subheader("Prediction Result")
        if uploaded and model is not None:
            label, confidence = predict(model, class_names, img)
            conf_pct = f"{confidence * 100:.2f}%"
            st.success(f"♻️ {label.upper()} — Confidence: {conf_pct}")
            st.subheader("Suggested Garbage Type")
            st.write(label.capitalize())

            st.subheader("Proper Management")
            guidance = MANAGEMENT_GUIDANCE.get(label.lower(), ["Follow local waste management guidelines."])
            st.markdown("\n".join([f"- {item}" for item in guidance]))

            st.subheader("Recycle Tips")
            tips = RECYCLE_TIPS.get(label.lower(), [
                "Rinse and empty the item before recycling when appropriate.",
                "Check local recycling guidelines for accepted materials and drop-off locations.",
            ])
            st.markdown("\n".join([f"- {item}" for item in tips]))

        elif uploaded and model is None:
            st.error("Model not available. Make sure saved_model/garbage_classifier.keras exists.")
        else:
            st.write("No prediction yet")


def main():
    choice = sidebar_nav()
    model, class_names = load_model_and_classes()

    st.title("♻️ Garbage Classification")
    st.write("Upload an image to classify the type of garbage and get management tips.")

    if choice == "Prediction":
        render_prediction_page(model, class_names)
    elif choice == "About Dataset":
        st.header("About Dataset")
        st.write("Brief info about the dataset used to train the model.")
    elif choice == "About Project":
        st.header("About Project")
        st.write("Description, goals and usage instructions.")
    elif choice == "Model Performance":
        st.header("Model Performance")
        metrics = load_metrics()
        if metrics is None:
            st.info("No precomputed metrics found. Run `python evaluate.py` in the repo root to compute metrics from the dataset and model.")
        else:
            overall = metrics.get("overall", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{overall.get('accuracy', 0) * 100:.2f}%")
            c2.metric("Precision (macro)", f"{overall.get('precision_macro', 0) * 100:.2f}%")
            c3.metric("Recall (macro)", f"{overall.get('recall_macro', 0) * 100:.2f}%")
            c4.metric("F1 (macro)", f"{overall.get('f1_macro', 0):.2f}")

            st.subheader("Performance by Category")
            per_class = metrics.get("per_class", {})
            items = list(per_class.items())
            for i in range(0, len(items), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(items):
                        name, stats = items[idx]
                        with col:
                            st.markdown(f"### {name.capitalize()}")
                            st.write(f"Precision: {stats.get('precision', 0) * 100:.2f}%")
                            st.write(f"Recall: {stats.get('recall', 0) * 100:.2f}%")
                            st.write(f"F1-score: {stats.get('f1-score', 0):.2f}")


if __name__ == "__main__":
    main()
