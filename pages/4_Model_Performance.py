import streamlit as st

st.set_page_config(page_title="Performance", page_icon="📊", layout="wide")

# Use Streamlit native components to respect light/dark themes and simplify layout

st.title("📊 Model Performance")

st.header("Key Metrics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", "94%")
with col2:
    st.metric("Precision", "92%")
with col3:
    st.metric("Recall", "91%")
with col4:
    st.metric("F1-Score", "0.91")

st.header("Performance by Category")

performance_data = [
    {"Category": "Cardboard", "Accuracy": "96%", "Precision": "94%", "Recall": "95%"},
    {"Category": "Glass", "Accuracy": "95%", "Precision": "93%", "Recall": "92%"},
    {"Category": "Metal", "Accuracy": "92%", "Precision": "91%", "Recall": "90%"},
    {"Category": "Paper", "Accuracy": "94%", "Precision": "92%", "Recall": "91%"},
    {"Category": "Plastic", "Accuracy": "93%", "Precision": "92%", "Recall": "92%"},
    {"Category": "Trash", "Accuracy": "91%", "Precision": "88%", "Recall": "89%"},
]

st.dataframe(performance_data, use_container_width=True, hide_index=True)

st.header("Training Details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Training")
    st.write("""
    - Accuracy: 96.5%
    - Loss: 0.142
    - Epochs: 50
    - Early Stopping: Yes
    """)

with col2:
    st.subheader("Validation")
    st.write("""
    - Accuracy: 94.2%
    - Loss: 0.198
    - Overfitting Risk: Low
    - Generalization: Good
    """)



with col3:
    st.metric("Model Size", "45.8 MB")

st.header("Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("✅ Strengths")
    st.write("""
    - High overall accuracy (94%)
    - Fast inference (<1 second)
    - Robust to lighting variations
    - Good generalization
    - Reliable on diverse data
    """)

with col2:
    st.subheader("⚠️ Areas for Improvement")
    st.write("""
    - Some confusion between similar materials
    - Contaminated items may be misclassified
    - Edge cases not well covered
    - Background sensitivity
    """)


