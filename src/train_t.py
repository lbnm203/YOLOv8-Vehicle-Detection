import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import tempfile
import ultralytics
from ultralytics import YOLO
import pandas as pd
import torch
import mlflow


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
    st.header("Training Module")
    # Chọn model
    model_path, model_name = choose_model()

    # Upload data config YAML
    st.write("#### Cấu hình dữ liệu (data.yaml)")
    yaml_file = st.file_uploader("Upload file data.yaml", type=['yaml','yml'])
    
    # Thêm input cho đường dẫn gốc của dataset
    st.write("#### Đường dẫn đến thư mục dataset")
    dataset_root = st.text_input("Đường dẫn tuyệt đối đến thư mục chứa dataset", 
                                value=os.path.abspath("./yolov8_dataset"))
    
    if not os.path.exists(dataset_root):
        st.warning(f"Thư mục '{dataset_root}' không tồn tại. Vui lòng kiểm tra lại đường dẫn.")

    # Tham số huấn luyện
    st.write("#### Tham số huấn luyện")
    epochs = st.slider("Số epochs", min_value=1, max_value=100, value=30)
    imgsz = st.slider("Kích thước ảnh", min_value=320, max_value=1280, value=640)
    batch_size = st.slider("Batch size", min_value=1, max_value=64, value=16)
    lr = st.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=0.01, format="%.6f")

    # Display selected parameters
    st.write("### Thông số đã chọn:")

    # Create a dictionary of parameters
    params = {
        "Model": model_name,
        "Epochs": epochs,
        "Image Size": imgsz,
        "Batch Size": batch_size,
        "Learning Rate": lr,
        "Dataset Path": dataset_root
    }

    params_df = pd.DataFrame(params.items(), columns=["Tham số", "Giá trị"])
    st.table(params_df)

    if yaml_file is not None:
        # Đọc nội dung YAML
        yaml_content = yaml.safe_load(yaml_file.read())
        
        # Kiểm tra xem YAML có chứa các khóa bắt buộc không
        required_keys = ['train', 'val']
        missing_keys = [key for key in required_keys if key not in yaml_content]
        
        if missing_keys:
            st.error(f"File YAML thiếu các khóa bắt buộc: {', '.join(missing_keys)}. Vui lòng đảm bảo file YAML có cả 'train' và 'val'.")
            
            # Hiển thị nội dung YAML hiện tại
            st.write("Nội dung YAML hiện tại:")
            st.code(yaml.dump(yaml_content), language="yaml")
            
            # Cung cấp mẫu YAML đúng
            st.write("Mẫu YAML đúng:")
            sample_yaml = {
                'path': dataset_root,
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'names': {
                    0: 'auto',
                    1: 'bicycle',
                    # ... thêm các lớp khác
                }
            }
            st.code(yaml.dump(sample_yaml), language="yaml")
            
            return
        
        # Cập nhật đường dẫn tuyệt đối trong YAML
        yaml_content['path'] = dataset_root
        
        # Kiểm tra xem các thư mục có tồn tại không
        train_path = os.path.join(dataset_root, yaml_content['train'])
        val_path = os.path.join(dataset_root, yaml_content['val'])
        
        if not os.path.exists(train_path):
            st.error(f"Không tìm thấy thư mục train: {train_path}")
            return
            
        if not os.path.exists(val_path):
            st.error(f"Không tìm thấy thư mục validation: {val_path}")
            return
            
        # Hiển thị thông tin về đường dẫn
        st.write("#### Kiểm tra đường dẫn dataset:")
        st.write(f"- Thư mục gốc: {dataset_root} - Tồn tại: {os.path.exists(dataset_root)}")
        st.write(f"- Thư mục train: {train_path} - Tồn tại: {os.path.exists(train_path)}")
        st.write(f"- Thư mục validation: {val_path} - Tồn tại: {os.path.exists(val_path)}")
        
        # Lưu tạm file YAML với đường dẫn tuyệt đối
        temp_yaml = tempfile.NamedTemporaryFile(delete=False, suffix='.yaml')
        with open(temp_yaml.name, 'w') as f:
            yaml.dump(yaml_content, f)
        data_cfg = temp_yaml.name
        
        # Hiển thị nội dung YAML đã cập nhật
        st.write("#### Nội dung YAML đã cập nhật:")
        st.code(yaml.dump(yaml_content), language="yaml")

        if st.button("Bắt đầu huấn luyện 🔥"):
            # Bắt đầu run MLflow
            with mlflow.start_run(run_name=model_name):
                try:
                    # Log parameters
                    mlflow.log_param("model", model_name)
                    mlflow.log_param("epochs", epochs)
                    mlflow.log_param("batch_size", batch_size)
                    mlflow.log_param("learning_rate", lr)
                    mlflow.log_param("data_config", os.path.basename(data_cfg))
                    mlflow.log_param("dataset_path", dataset_root)

                    # Khởi tạo model
                    model = YOLO(model_path)
                    # Thư mục lưu kết quả
                    save_dir = os.path.join('runs', 'train', model_name)
                    os.makedirs(save_dir, exist_ok=True)

                    # Hiển thị thông tin debug
                    st.write(f"Đường dẫn YAML: {data_cfg}")
                    with open(data_cfg, 'r') as f:
                        st.code(f.read(), language="yaml")

                    # Train
                    results = model.train(
                        data=data_cfg,
                        epochs=epochs,
                        imgsz=imgsz,
                        batch=batch_size,
                        lr0=lr,
                        project='runs/train',
                        name=model_name,
                        exist_ok=True
                    )

                    # Hiển thị kết quả metrics
                    metrics = results.metrics
                    df = pd.DataFrame(metrics)
                    st.write("#### Training Metrics")
                    st.dataframe(df)

                    # Plot loss curves
                    if 'train/loss' in df.columns and 'val/loss' in df.columns:
                        fig, ax = plt.subplots()
                        ax.plot(df['epoch'], df['train/loss'], label='Train Loss')
                        ax.plot(df['epoch'], df['val/loss'], label='Val Loss')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.legend()
                        st.pyplot(fig)

                    # Log metrics per epoch
                    for _, row in df.iterrows():
                        epoch = int(row['epoch'])
                        for col in df.columns:
                            if col != 'epoch':
                                mlflow.log_metric(col, float(row[col]), step=epoch)

                    # Log artifacts: plots and model
                    mlflow.log_artifacts(save_dir)
                    st.success("Training và logging MLflow hoàn tất!")
                    
                except Exception as e:
                    st.error(f"Lỗi trong quá trình huấn luyện: {str(e)}")
                    st.exception(e)
                finally:
                    # Xóa file tạm
                    try:
                        os.unlink(data_cfg)
                    except:
                        pass
    else:
        st.info("Vui lòng upload file data.yaml để bắt đầu huấn luyện.")
        
        # # Hiển thị mẫu YAML
        # st.write("Mẫu file data.yaml:")
        # sample_yaml = {
        #     'path': dataset_root if 'dataset_root' in locals() else './yolov8_dataset',
        #     'train': 'train/images',
        #     'val': 'val/images',
        #     'test': 'test/images',
        #     'names': {
        #         0: 'auto',
        #         1: 'bicycle',
        #         2: 'bus',
        #         3: 'car',
        #         4: 'tempo',
        #         5: 'tractor',
        #         6: 'two_wheelers',
        #         7: 'vehicle_truck'
        #     }
        # }
        # st.code(yaml.dump(sample_yaml), language="yaml")
        
        # # Thêm tùy chọn sử dụng file YAML mặc định
        # if st.button("Sử dụng file YAML mặc định"):
        #     try:
        #         default_yaml_path = './yolov8_dataset/custom_dataset.yaml'
        #         if os.path.exists(default_yaml_path):
        #             with open(default_yaml_path, 'r') as f:
        #                 yaml_content = yaml.safe_load(f)
                        
        #             # Cập nhật đường dẫn tuyệt đối
        #             if 'dataset_root' in locals():
        #                 yaml_content['path'] = dataset_root
                        
        #             st.success(f"Đã tải file YAML mặc định từ {default_yaml_path}")
        #             st.write("Nội dung đã cập nhật:")
        #             st.code(yaml.dump(yaml_content), language="yaml")
        #             st.session_state.default_yaml_content = yaml_content
        #         else:
        #             st.error(f"Không tìm thấy file YAML mặc định tại {default_yaml_path}")
        #     except Exception as e:
        #         st.error(f"Lỗi khi tải file YAML mặc định: {str(e)}")


