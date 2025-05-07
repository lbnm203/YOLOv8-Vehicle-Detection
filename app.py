import streamlit as st

from src.data_t import *
from src.theory_t import *
from src.train_t import *
from src.demo_t import *
from src.mlflow_t import *

st.set_page_config(
    page_title="YOLOv8 Vehicle Detection",
    layout="centered"
)


def check_dataset_availability():
    """Check if dataset is available and provide download options if not"""
    yaml_path = './yolov8_dataset/custom_dataset.yaml'
    with st.container(border=True):
        if not os.path.exists(yaml_path):
            st.warning("⚠️ Dataset not detected")
            st.info("""
            ### Dataset Options:
            
            1. Download automatically from Hugging Face (recommended)
            2. Download manually from [Kaggle](https://www.kaggle.com/datasets/dataclusterlabs/indian-vehicle-dataset)
            """)

            # Add Hugging Face download option in sidebar
            if st.button("Download from Hugging Face"):
                from src.data_t import download_dataset_from_huggingface
                download_dataset_from_huggingface()
        else:
            st.success("✅ Dataset detected")
            # Add Hugging Face attribution
            st.markdown(
                "Dataset hosted on [Hugging Face](https://huggingface.co/datasets/lbnm203/yolov8_dataset)")


def main():
    st.title("YOLOv8 Vehicle Detection")

    data_tab, theory_tab, train_tab, demo_tab, mlflow_tab = st.tabs(
        ["Data", "Theory", "Training", "Demo", "MLflow"]
    )

    with data_tab:
        # Check dataset availability
        check_dataset_availability()
        data_description()

    with theory_tab:
        yolov8_theory()

    with train_tab:
        train_model()

    with demo_tab:
        demo_detection()

    with mlflow_tab:
        display_mlflow_runs()


if __name__ == "__main__":
    main()
