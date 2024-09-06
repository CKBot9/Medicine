import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import shutil
import cv2
from ultralytics import YOLO
from qdrant_client import QdrantClient
import streamlit as st
from tempfile import NamedTemporaryFile

# Initialize the models and Qdrant client (using your existing setup)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
segment_model_path = "best_segment.pt"
model_detection = YOLO(segment_model_path)
model_segmentation = YOLO(segment_model_path)
collection_name = "Medicine_Group"
qdrant_client = QdrantClient(
    url="https://e7516d85-75a7-402f-977f-451fbfc03297.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="_rs9qDa_Hal_p0yKWhee-8w40Ok2psZ4QqsTA5m40YzAIPyBtWOOiA",
)

def detect_and_crop_objects(image):
    results = model_detection(image)
    cropped_images = []
    bboxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
            cropped_images.append(cropped_image)
            bboxes.append((x1, y1, x2, y2))
    return cropped_images, bboxes

def segment_image(image: np.ndarray, bbox: list) -> np.ndarray:
    x_min, y_min, x_max, y_max = map(int, bbox)
    cropped_image = image[y_min:y_max, x_min:x_max]
    results = model_segmentation.predict(source=cropped_image)
    black_image = np.zeros_like(cropped_image)
    for result in results:
        if result.masks is not None:
            masks = result.masks.xy
            for mask in masks:
                mask = np.array(mask, dtype=np.int32)
                cv2.fillPoly(black_image, [mask], (255, 255, 255))
    binary_mask = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(binary_mask, 1, 255, cv2.THRESH_BINARY)
    segmented_image = cv2.bitwise_and(cropped_image, cropped_image, mask=binary_mask)
    result_image = cv2.resize(segmented_image, (640, 640))
    return result_image

def get_image_vector(image_path):
    image = cv2.imread(image_path)
    cropped_images, bboxes = detect_and_crop_objects(image)
    if cropped_images:
        processed_image = segment_image(image, bboxes[0])
    else:
        processed_image = segment_image(image, [0, 0, image.shape[1], image.shape[0]])
    pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)).convert("RGB")
    inputs = processor(images=pil_image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.cpu().numpy().flatten(), processed_image

def find_top_k_similar_classes_with_qdrant(image_path, unseenmedicine_folder, k=5, top_n=1000):
    start_time = time.time()
    image_vector, processed_image = get_image_vector(image_path)
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=image_vector.tolist(),
        limit=top_n,
        with_payload=True,
        with_vectors=False
    )
    top_results = [(result.payload.get("class_name", "unknown"), result.score) for result in search_result]
    unique_classes = {}
    for class_name, score in top_results:
        if class_name not in unique_classes:
            unique_classes[class_name] = score
        if len(unique_classes) == k:
            break
    filtered_top_k_classes = sorted(unique_classes.items(), key=lambda x: x[1], reverse=True)
    processing_time = time.time() - start_time

    if len(filtered_top_k_classes) < k:
        st.warning("Not enough unique classes found.")
        return processed_image, [], processing_time

    top_1_score = filtered_top_k_classes[0][1]
    top_2_score = filtered_top_k_classes[1][1] if len(filtered_top_k_classes) > 1 else 0
    class_names = []
    if top_1_score > 0.9 and (top_1_score - top_2_score) > 0.02:
        class_names.append(f"Top prediction: {filtered_top_k_classes[0][0]}")
        st.write(f"Confidence: {top_1_score:.4f}")
    elif top_1_score > 0.9 and (top_1_score - top_2_score) <= 0.02:
        class_names.append(f"The model can't confirm the class. It might be: ( {filtered_top_k_classes[0][0]} ) Please provide a clearer picture.")
        st.write(f"Confidence: {top_1_score:.4f}")
    elif top_1_score <= 0.9 and top_1_score >= 0.85:
        class_names.append(f"Please take a clearer picture. This picture has too much noise.")
        st.write(f"Confidence: {top_1_score:.4f}")
    elif top_1_score < 0.85:
        st.error(f"The model has never seen this medicine before. (The picture was saved in folder 'unseen_med')")
        st.write(f"Confidence: {top_1_score:.4f}")
        if not os.path.exists(unseenmedicine_folder):
            os.makedirs(unseenmedicine_folder)
        shutil.copy(image_path, unseenmedicine_folder)
        class_names.append(f"Image saved to {unseenmedicine_folder}")

    return processed_image, class_names, processing_time

# Streamlit GUI
st.title("Medicine Image Processing")

# Upload Image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
unseenmedicine_folder = "unseen_med"

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        image_path = temp_file.name

    # Display the uploaded image
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_path, caption="Uploaded Image", use_column_width=True)

    # Start Processing
    if st.button("Start Processing"):
        processed_image, class_names, processing_time = find_top_k_similar_classes_with_qdrant(image_path, unseenmedicine_folder, k=5)

        # Display the processed image
        with col2:
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)

        # Display class names and similarity scores
        for name in class_names:
            st.write(name)

        # Display processing time
        st.write(f"Processing Time: {processing_time:.2f} seconds")
