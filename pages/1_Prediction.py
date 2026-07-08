import streamlit as st
from PIL import Image
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import predict_image

st.set_page_config(
    page_title="Predict",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Garbage Classification")
st.subheader("Upload an image to get instant AI classification")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose image",
        type=["jpg", "jpeg", "png"],
    )

with col2:
    st.subheader("💡 Tips")
    st.caption("""
    ✓ Use clear, well-lit images
    ✓ Minimize background clutter
    ✓ Position item clearly
    ✓ Avoid shadows
    """)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.image(image, use_container_width=True, caption="Your Image")
    
    with col2:
        result = predict_image(image)
        
        st.success("✅ Classification complete")
        st.subheader(f"✓ {result['category'].upper()}")
        st.metric("Confidence", f"{result['confidence']:.1f}%")
        st.progress(result['confidence'] / 100)
        
        
        st.markdown("### ♻️ Disposal Instructions")
        st.write(result['disposal'])
        
        st.markdown("### 🌱 Environmental Tip")
        st.write(result['tip'])

st.caption("♻️ Waste Classification System")
