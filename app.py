import streamlit as st
import cv2

from src.data_t import *
from src.theory_t import *
from src.train_t import *

st.set_page_config(
    page_title="YOLOv8 Vehicle Detection",
    layout="centered"
)


def main():
    st.title("YOLOv8 Vehicle Detection")

    data_tab, theory_tab, train_tab, demo_tab, mlflow_tab = st.tabs(
        ["Data", "Theory", "Training", "Demo", "MLflow"]
    )

    with data_tab:
        data_description()

    with theory_tab:
        yolov8_theory()

    with train_tab:
        train_model()

    with demo_tab:
        pass

    with mlflow_tab:
        pass


if __name__ == "__main__":
    main()
