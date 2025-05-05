import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import ultralytics
from ultralytics import YOLO
import pandas as pd
import torch


def choose_model():
    st.write("### Chọn mô hình YOLOv8")

    model_option = st.selectbox(
        "Chọn model",
        options=["YOLOv8n", "YOLOv8s", "YOLOv8m"],
        index=0
    )

    # Map model option to path
    model_paths = {
        "YOLOv8n": "yolov8n.pt",
        "YOLOv8s": "yolov8s.pt",
        "YOLOv8m": "yolov8m.pt",
    }

    model_path = model_paths[model_option]

    return model_path, model_option


def train_model():
    st.write("### Huấn luyện mô hình YOLOv8")

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        st.success("CUDA is available! GPU acceleration can be used.")
        device_options = ["cuda", "cpu"]
        default_device = "cuda"
    else:
        st.warning(
            "CUDA is not available. Training will use CPU only, which may be slower.")
        device_options = ["cpu"]
        default_device = "cpu"

    model_path, model_option = choose_model()

    epochs = st.slider("Chọn số lượng epoch", 1, 100, 10)
    imgsz = st.slider("Chọn kích thước ảnh", 320, 1280, 640)
    batch = st.slider("Chọn batch size", 1, 64, 16)
    lr0 = st.slider("Chọn learning rate", 0.000001, 0.1, 0.01)

    optimizer = st.selectbox(
        "Chọn optimizer",
        options=["Adam", "SGD", "AdamW"],
        index=0
    )

    conf_threshold = st.slider(
        "Ngưỡng Confidence", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

    iou_threshold = st.slider(
        "Ngưỡng IOU", min_value=0.1, max_value=1.0, value=0.45, step=0.05)

    if 'device' not in st.session_state or st.session_state.device not in device_options:
        st.session_state.device = default_device

    device = st.selectbox(
        "Chọn device",
        options=device_options,
        index=device_options.index(st.session_state.device),
        key="device"
    )

    # Initialize session state for parameters if they don't exist
    if 'model_option' not in st.session_state:
        st.session_state.model_option = model_option
    if 'epochs' not in st.session_state:
        st.session_state.epochs = epochs
    if 'imgsz' not in st.session_state:
        st.session_state.imgsz = imgsz
    if 'batch' not in st.session_state:
        st.session_state.batch = batch
    if 'lr0' not in st.session_state:
        st.session_state.lr0 = lr0
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = optimizer
    if 'conf_threshold' not in st.session_state:
        st.session_state.conf_threshold = conf_threshold
    if 'iou_threshold' not in st.session_state:
        st.session_state.iou_threshold = iou_threshold
    if 'device' not in st.session_state:
        st.session_state.device = device

    # Display selected parameters
    st.write("### Thông số đã chọn:")

    # Create a dictionary of parameters
    params = {
        "Model": model_option,
        "Epochs": epochs,
        "Image Size": imgsz,
        "Batch Size": batch,
        "Learning Rate": lr0,
        "Confidence Threshold": conf_threshold,
        "IOU Threshold": iou_threshold,
        "Optimizer": optimizer,
        "Device": device
    }

    params_df = pd.DataFrame(params.items(), columns=["Tham số", "Giá trị"])
    st.table(params_df)

    st.write("---")

    # Add a button to start training
    if st.button("Bắt đầu huấn luyện"):
        with st.spinner("Đang huấn luyện mô hình..."):
            try:
                # Add this before training to debug paths
                st.write(f"Current working directory: {os.getcwd()}")
                st.write(
                    f"Dataset path: {os.path.join(os.getcwd(), 'yolov8_dataset/custom_dataset.yaml')}")
                st.write(
                    f"Dataset exists: {os.path.exists(os.path.join(os.getcwd(), 'yolov8_dataset/custom_dataset.yaml'))}")
                st.write(
                    f"Train images path: {os.path.join(os.getcwd(), 'yolov8_dataset/train/images')}")
                st.write(
                    f"Train images exist: {os.path.exists(os.path.join(os.getcwd(), 'yolov8_dataset/train/images'))}")

                # Load the model
                model = YOLO(model_path)

                # Start training
                yaml_path = "./yolov8_dataset/custom_dataset.yaml"
                st.write(f"Using YAML path: {os.path.abspath(yaml_path)}")

                # Check if directories exist
                train_dir = "./yolov8_dataset/train/images"
                val_dir = "./yolov8_dataset/val/images"
                test_dir = "./yolov8_dataset/test/images"

                st.write(f"Train directory exists: {os.path.exists(train_dir)}")
                st.write(f"Val directory exists: {os.path.exists(val_dir)}")
                st.write(f"Test directory exists: {os.path.exists(test_dir)}")

                # Start training with relative path
                results = model.train(
                    data=yaml_path,
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch,
                    lr0=lr0,
                    optimizer=optimizer,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=device,
                    project="runs/train",
                    name=f"{model_option}_e{epochs}_b{batch}"
                )

                st.success("Huấn luyện hoàn tất!")

                # Display training results
                st.write("### Kết quả huấn luyện:")

                # Get the run directory
                run_dir = results.save_dir if hasattr(
                    results, 'save_dir') else f"runs/train/{model_option}_e{epochs}_b{batch}"

                # Display metrics
                if os.path.exists(os.path.join(run_dir, "results.csv")):
                    metrics_df = pd.read_csv(
                        os.path.join(run_dir, "results.csv"))
                    st.write("#### Metrics:")
                    st.dataframe(metrics_df)

                    # Plot metrics
                    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
                    metrics_df.plot(
                        x='epoch', y=['train/box_loss', 'val/box_loss'], ax=ax[0])
                    ax[0].set_title('Box Loss')
                    metrics_df.plot(
                        x='epoch', y=['train/cls_loss', 'val/cls_loss'], ax=ax[1])
                    ax[1].set_title('Class Loss')
                    st.pyplot(fig)

                # Display confusion matrix
                conf_matrix_path = os.path.join(
                    run_dir, "confusion_matrix.png")
                if os.path.exists(conf_matrix_path):
                    st.write("#### Confusion Matrix:")
                    st.image(conf_matrix_path)

                # Display validation images
                val_images_dir = os.path.join(run_dir, "val_batch0_pred.jpg")
                if os.path.exists(val_images_dir):
                    st.write("#### Validation Predictions:")
                    st.image(val_images_dir)

                # Display F1 curve
                f1_curve_path = os.path.join(run_dir, "F1_curve.png")
                if os.path.exists(f1_curve_path):
                    st.write("#### F1 Curve:")
                    st.image(f1_curve_path)

                # Display PR curve
                pr_curve_path = os.path.join(run_dir, "PR_curve.png")
                if os.path.exists(pr_curve_path):
                    st.write("#### Precision-Recall Curve:")
                    st.image(pr_curve_path)

                # Save the trained model path to session state
                st.session_state.trained_model_path = os.path.join(
                    run_dir, "weights/best.pt")
                st.info(
                    f"Trained model saved to: {st.session_state.trained_model_path}")

            except Exception as e:
                st.error(f"Lỗi khi huấn luyện: {str(e)}")
                st.exception(e)  # This will display the full traceback
