import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
import tempfile


def load_model(model_path):
    """Load a YOLOv8 model"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def process_image(model, image, conf_threshold=0.25):
    """Process an image with the model and return results"""
    try:
        results = model.predict(image, conf=conf_threshold)
        return results[0]  # Return the first result
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None


def draw_results(image, results, class_names):
    """Draw bounding boxes and labels on the image"""
    img = image.copy()

    if results is not None and results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = box.astype(int)
            label = f"{class_names[cls_id]} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label background
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)

            # Draw label text
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return img


def demo_detection():
    st.header("Demo Vehicle Detection")

    # Model selection
    st.write("### Chọn mô hình")
    model_option = st.selectbox(
        "Chọn model",
        options=["YOLOv8n", "YOLOv8s", "YOLOv8m", "Custom trained model"],
        index=0
    )

    # Map model option to path
    if model_option == "Custom trained model":
        model_path = st.text_input("Đường dẫn đến model đã huấn luyện",
                                   value="runs/train/YOLOv8n/weights/best.pt")
    else:
        model_paths = {
            "YOLOv8n": "models/yolov8n.pt",
            "YOLOv8s": "models/yolov8s.pt",
            "YOLOv8m": "models/yolov8m.pt",
        }
        model_path = model_paths[model_option]

    # Check if model exists
    if not os.path.exists(model_path):
        st.warning(f"Model không tồn tại tại đường dẫn: {model_path}")

    # Load class names
    try:
        yaml_path = "./yolov8_dataset/custom_dataset.yaml"
        if os.path.exists(yaml_path):
            import yaml
            with open(yaml_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
                class_names = list(yaml_content['names'].values())
        else:
            # Default COCO class names
            class_names = ['auto', 'bicycle', 'bus', 'car',
                           'tempo', 'tractor', 'two_wheelers', 'vehicle_truck']
    except Exception as e:
        st.error(f"Error loading class names: {str(e)}")
        class_names = ['auto', 'bicycle', 'bus', 'car',
                       'tempo', 'tractor', 'two_wheelers', 'vehicle_truck']

    # Confidence threshold
    conf_threshold = st.slider(
        "Ngưỡng Confidence", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

    # Input options
    st.write("### Chọn nguồn đầu vào")
    input_option = st.radio(
        "Chọn nguồn", ["Upload ảnh", "Upload video"])

    # Load model
    model = load_model(model_path)

    if model is None:
        st.error("Không thể tải model. Vui lòng kiểm tra đường dẫn.")
        return

    if input_option == "Upload ảnh":
        uploaded_file = st.file_uploader(
            "Chọn ảnh", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            # Display original image
            st.write("### Ảnh gốc")
            st.image(image, caption="Ảnh đầu vào", use_container_width=True)

            # Process image
            results = process_image(model, image_np, conf_threshold)

            if results is not None:
                # Draw results
                output_image = draw_results(image_np, results, class_names)

                # Display output image
                st.write("### Kết quả phát hiện")
                st.image(output_image, caption="Kết quả",
                         use_container_width=True)

                # Display detection details
                if results.boxes is not None and len(results.boxes) > 0:
                    st.write("### Chi tiết phát hiện")

                    # Create a list of detections
                    detections = []
                    boxes = results.boxes.xyxy.cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy()
                    cls_ids = results.boxes.cls.cpu().numpy().astype(int)

                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
                        detections.append({
                            "STT": i+1,
                            "Lớp": class_names[cls_id],
                            "Confidence": f"{conf:.4f}",
                            "Tọa độ (x1,y1,x2,y2)": f"({int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])})"
                        })

                    # Display as table
                    import pandas as pd
                    df = pd.DataFrame(detections)
                    st.table(df)

    elif input_option == "Upload video":
        uploaded_file = st.file_uploader(
            "Chọn video", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            # Save uploaded video to temp file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='.mp4')
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name

            # Open video
            cap = cv2.VideoCapture(video_path)

            # Get video info
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            st.write(
                f"Video FPS: {fps}, Resolution: {frame_width}x{frame_height}")

            # Process video
            st.write("### Xử lý video")

            # Create a placeholder for the processed video
            video_placeholder = st.empty()

            # Process first frame for preview
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process frame
                results = process_image(model, frame_rgb, conf_threshold)

                # Draw results
                output_frame = draw_results(frame_rgb, results, class_names)

                # Display frame
                video_placeholder.image(
                    output_frame, caption="Frame preview", use_container_width=True)

            # Option to process full video
            if st.button("Xử lý toàn bộ video"):
                # Create output video file
                output_path = "output_video.mp4"
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
                    *'mp4v'), fps, (frame_width, frame_height))

                # Reset video capture
                cap.release()
                cap = cv2.VideoCapture(video_path)

                # Process each frame
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress_bar = st.progress(0)

                for i in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Process frame
                    results = process_image(model, frame_rgb, conf_threshold)

                    # Draw results
                    output_frame = draw_results(
                        frame_rgb, results, class_names)

                    # Convert back to BGR for video writing
                    output_frame_bgr = cv2.cvtColor(
                        output_frame, cv2.COLOR_RGB2BGR)

                    # Write frame to output video
                    out.write(output_frame_bgr)

                    # Update progress
                    progress_bar.progress((i + 1) / frame_count)

                # Release resources
                cap.release()
                out.release()

                # Provide download link
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="Tải video đã xử lý",
                        data=f,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )

            # Clean up
            cap.release()
            os.unlink(video_path)
