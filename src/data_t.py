import streamlit as st
import os
import yaml
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np
import matplotlib.patches as patches


def data_description():
    st.write("### Tập Dữ Liệu Indian Vehicle Dataset")

    st.write("Tập dữ liệu này được thu thâp bởi DataCluster Labs. Bộ dữ liệu này là một tập hợp gồm hơn 50.000 hình ảnh xe gốc được chụp và thu thập từ hơn 1000 khu vực thành thị và nông thôn, trong đó mỗi hình ảnh đều được các chuyên gia về thị giác máy tính tại Datacluster Labs xem xét và xác minh thủ công")

    st.write("#### Một số thông tin về tập dữ liệu:")

    st.write("- **Kích thước tập dữ liệu:** 50.000+ hình ảnh")
    st.write("- **Được chụp bởi:** Hơn 1000+ người đóng góp cộng đồng")
    st.write("- **Độ phân giải:** 100% hình ảnh là HD trở lên (1920x1080 trở lên)")
    st.write("- **Địa điểm:** Chụp ảnh tại hơn 1000 thành phố trên khắp Ấn Độ")
    st.write("- **Tính đa dạng:** Nhiều điều kiện ánh sáng khác nhau như ngày, đêm, khoảng cách khác nhau, điểm nhìn, ...")
    st.write(
        "- **Thiết bị sử dụng:** Chụp bằng điện thoại di động trong năm 2020-2022")
    st.write("- **Công dụng:** Phát hiện xe cộ, Phát hiện ô tô, Phát hiện xe xây dựng, Hệ thống tự lái, ...")
    st.write("- **Những lớp/phương tiện có trong tập dữ liệu:** auto, bicycle, bus, car, tempo, tractor, two_wheelers, vehicle_truck")
    st.write("---")

    # Đọc tệp YAML
    with open('./yolov8_dataset/custom_dataset.yaml', 'r') as f:
        dataset_config = yaml.safe_load(f)

    # Lấy thông tin từ YAML
    yaml_dir = os.path.dirname('./yolov8_dataset/custom_dataset.yaml')
    train_dir = os.path.join(yaml_dir, dataset_config['train'])
    val_dir = os.path.join(yaml_dir, dataset_config['val'])
    test_dir = os.path.join(yaml_dir, dataset_config['test'])
    # ['auto', 'bicycle', ...]
    class_names = list(dataset_config['names'].values())

    # Đếm số lượng hình ảnh trong mỗi tập
    train_count = len([f for f in os.listdir(train_dir)
                      if f.endswith(('.jpg', '.png'))])
    val_count = len([f for f in os.listdir(val_dir)
                    if f.endswith(('.jpg', '.png'))])
    test_count = len([f for f in os.listdir(test_dir)
                     if f.endswith(('.jpg', '.png'))])

    # a. Vẽ biểu đồ phân bố số lượng hình ảnh
    st.write("#### Phân bố số lượng hình ảnh trong các tập dữ liệu")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['Train', 'Val', 'Test'], [train_count, val_count,
           test_count], color=['blue', 'orange', 'green'])
    ax.set_title('Phân bố số lượng hình ảnh trong các tập dữ liệu')
    ax.set_xlabel('Tập dữ liệu')
    ax.set_ylabel('Số lượng hình ảnh')
    st.pyplot(fig)

    # Gọi hàm phân tích dataset với các biến đã định nghĩa
    analyze_dataset(train_dir, class_names)


# b. Phân tích phân bố lớp (nếu có nhãn)
def get_class_distribution(label_dir, class_names):
    class_counts = [0] * len(class_names)
    if not os.path.exists(label_dir):
        return class_counts

    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    return class_counts


@st.cache_data
def analyze_dataset(train_dir, class_names):
    st.write("#### Phân bố số lượng đối tượng theo lớp (Tập Train)")
    train_label_dir = train_dir.replace('images', 'labels')
    train_class_counts = get_class_distribution(train_label_dir, class_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(class_names, train_class_counts, color='purple')
    ax.set_title('Phân bố số lượng đối tượng theo lớp (Tập Train)')
    ax.set_xlabel('Lớp')
    ax.set_ylabel('Số lượng đối tượng')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # c. Hiển thị hình ảnh mẫu với bounding boxes (nếu có)
    display_sample_images_with_boxes(
        train_dir, train_label_dir, 'Train', class_names)

    # d. Phân tích phân bố giá trị pixel (histogram)
    plot_pixel_distribution(train_dir)


@st.cache_data
def display_sample_images_with_boxes(image_dir, label_dir, title, class_names, num_samples=4):
    st.write("#### Một số mẫu với bounding boxes")
    image_files = [f for f in os.listdir(
        image_dir) if f.endswith(('.jpg', '.png'))]
    if not image_files:
        st.warning(f"Không tìm thấy hình ảnh trong thư mục {image_dir}")
        return

    sample_files = random.sample(
        image_files, min(num_samples, len(image_files)))

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i, img_file in enumerate(sample_files):
        if i >= num_samples:
            break

        # Đọc hình ảnh
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path)
        img_array = np.array(img)

        # Đọc nhãn (nếu có)
        label_path = os.path.join(label_dir, img_file.replace(
            '.jpg', '.txt').replace('.png', '.txt'))
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, center_x, center_y, width, height = map(
                        float, line.split())
                    class_id = int(class_id)
                    # Chuyển đổi tọa độ YOLO sang tọa độ pixel
                    h, w = img_array.shape[:2]
                    x = (center_x - width / 2) * w
                    y = (center_y - height / 2) * h
                    box_w = width * w
                    box_h = height * h
                    boxes.append([x, y, box_w, box_h])
                    labels.append(class_names[class_id])

        # Vẽ hình ảnh và bounding boxes
        axes[i].imshow(img_array)
        for box, label in zip(boxes, labels):
            rect = patches.Rectangle(
                (box[0], box[1]), box[2], box[3], linewidth=2, edgecolor='r', facecolor='none')
            axes[i].add_patch(rect)
            axes[i].text(box[0], box[1], label, color='red',
                         fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        axes[i].set_title(f'{title} - Mẫu {i+1}')
        axes[i].axis('off')

    plt.tight_layout()
    st.pyplot(fig)


@st.cache_data
def plot_pixel_distribution(image_dir):
    image_files = [f for f in os.listdir(
        image_dir) if f.endswith(('.jpg', '.png'))]
    if not image_files:
        st.warning(f"Không tìm thấy hình ảnh trong thư mục {image_dir}")
        return

    sample_img = Image.open(os.path.join(image_dir, image_files[0]))
    img_array = np.array(sample_img)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, color in enumerate(['red', 'green', 'blue']):
        ax.hist(img_array[:, :, i].ravel(), bins=256,
                color=color, alpha=0.5, label=color.capitalize())
    ax.set_title('Phân bố giá trị pixel (RGB) - Tập Train')
    ax.set_xlabel('Giá trị pixel')
    ax.set_ylabel('Tần suất')
    ax.legend()
