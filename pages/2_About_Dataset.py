import streamlit as st
import os
from pathlib import Path

st.set_page_config(page_title="Dataset", page_icon="📊", layout="wide")

# Use Streamlit native components and metrics; avoid custom HTML/CSS for theme compatibility

st.title("📊 Dataset Information")

st.header("Overview")
st.write("""
The Waste Classification Dataset contains thousands of labeled images 
across 6 waste categories used to train our deep learning model.
""")

# Calculate dataset statistics
dataset_dir = Path("dataset")
category_counts = {}
total_images = 0

categories = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

for category in categories:
    cat_path = dataset_dir / category
    if cat_path.exists():
        count = len([f for f in os.listdir(cat_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
        category_counts[category] = count
        total_images += count

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Images", f"{total_images}")

with col2:
    st.metric("Categories", "6")

with col3:
    avg = total_images // 6 if total_images > 0 else 0
    st.metric("Avg per Category", f"{avg}")

with col4:
    st.metric("Image Size", "224×224")

st.header("Categories")

category_info = {
    "Cardboard": "Corrugated boxes, paper boards, packaging materials",
    "Glass": "Glass bottles, jars, containers",
    "Metal": "Aluminum cans, tin cans, metal containers",
    "Paper": "Newspaper, magazines, office paper",
    "Plastic": "Plastic bottles, bags, containers",
    "Trash": "General waste, mixed materials"
}

category_table = [
    {
        "Category": cat,
        "Images": category_counts.get(cat.lower(), 0),
        "Description": desc,
    }
    for cat, desc in category_info.items()
]

st.dataframe(category_table, use_container_width=True, hide_index=True)

st.header("Data Preparation")

st.write("**Data Preparation**")
st.write("- **Resizing:** All images standardized to 224×224 pixels")
st.write("- **Normalization:** Pixel values normalized to [0, 1]")
st.write("- **Split:** 80% training, 10% validation, 10% testing")
st.write("- **Augmentation:** Rotation, flipping, zoom, brightness adjustments")

st.caption("♻️ Waste Classification System")
