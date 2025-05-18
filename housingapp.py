import streamlit as st
import torch
import numpy as np
from train_model import SimpleRegressionNet, fetch_dataset
import os
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Housing Price Inference", layout="centered")

st.title("🏡 Housing Price Regression Inference")

model = None
X_train, y_train = None, None
X_val, y_val = None, None

def load_model(path):
    m = SimpleRegressionNet()
    checkpoint = torch.load(path)
    m.load_state_dict(checkpoint["model_state_dict"])
    m.eval()
    return m

def show_data(data, label, key):
    st.markdown(f"### {label}")
    index = st.selectbox(f"Select index ({label})", list(range(len(data))), key=key)
    features = data[index]
    st.write(f"Features: {np.round(features, 2)}")
    return index, features

st.sidebar.markdown("### 🔍 Load Model")
model_file = st.sidebar.file_uploader("Upload .pt Model", type=["pt"])
if model_file:
    with open("checkpoints/tmp_model.pt", "wb") as f:
        f.write(model_file.read())
    model = load_model("checkpoints/tmp_model.pt")
    st.success("Model loaded!")

st.sidebar.markdown("### 📥 Load Dataset")
if st.sidebar.button("Load Dataset"):
    X, y = fetch_dataset()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    st.session_state.X_train = X_train
    st.session_state.X_val = X_val
    st.session_state.y_train = y_train
    st.session_state.y_val = y_val
    st.success("Dataset loaded!")

if model and "X_train" in st.session_state:
    tab1, tab2 = st.tabs(["🧪 Inference from Dataset", "✍️ Manual Input"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            train_idx, train_feats = show_data(st.session_state.X_train, "Train Data", key="train")
            if st.button("Predict Train"):
                pred = model(torch.FloatTensor(train_feats).unsqueeze(0)).item()
                true_val = st.session_state.y_train[train_idx]
                st.info(f"Predicted: **{pred:.3f}**, Ground Truth: **{true_val:.3f}**")

        with col2:
            val_idx, val_feats = show_data(st.session_state.X_val, "Validation Data", key="val")
            if st.button("Predict Val"):
                pred = model(torch.FloatTensor(val_feats).unsqueeze(0)).item()
                true_val = st.session_state.y_val[val_idx]
                st.info(f"Predicted: **{pred:.3f}**, Ground Truth: **{true_val:.3f}**")

    with tab2:
        st.markdown("### Manual Feature Input (8 values)")
        manual_input = []
        cols = st.columns(4)
        for i in range(8):
            with cols[i % 4]:
                val = st.number_input(f"Feature {i}", key=f"f{i}", format="%.3f")
                manual_input.append(val)

        if st.button("Predict Manual Input"):
            inp = torch.FloatTensor(manual_input).unsqueeze(0)
            pred = model(inp).item()
            st.success(f"Predicted Price: **{pred:.3f}**")
