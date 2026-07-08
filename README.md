# Garbage Classification

A Streamlit web app for classifying waste images into common garbage categories such as cardboard, glass, metal, paper, plastic, and trash.

## Features
- Upload an image and get a predicted waste category
- View the uploaded image preview
- See suggested waste-management guidance and recycling tips
- Explore dataset and model-performance information through the app pages

## Project Structure
- app.py: Main Streamlit application entry point
- pages/: Additional app pages for dataset, project info, and model performance
- dataset/: Image dataset used for training and evaluation
- saved_model/: Trained model and related files
- models/: Model architecture notes
- notebooks/: Jupyter notebooks for exploration and model development

## Requirements
Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Run the App
From the project root, run:

```bash
streamlit run app.py
```

## Notes
- The app expects the trained model file at saved_model/garbage_classifier.keras
- The app also uses saved_model/class_names.json for class labels
- If metrics are not available, the performance page will show a fallback message
