
# Fix old keras-vggface imports for Keras 2.15
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("keras_vggface")

if spec and spec.submodule_search_locations:
    package_dir = Path(next(iter(spec.submodule_search_locations)))

    replacements = {
        "models.py": [
            ("from keras.utils import layer_utils",
             "from keras.src.utils import layer_utils"),
            ("from keras.utils.data_utils import get_file",
             "from keras.utils import get_file"),
            ("from keras.engine.topology import get_source_inputs",
             "from keras.utils import get_source_inputs"),
        ],
        "utils.py": [
            ("from keras.utils.data_utils import get_file",
             "from keras.utils import get_file"),
        ],
    }

    for filename, fixes in replacements.items():
        file_path = package_dir / filename

        if file_path.exists():
            source = file_path.read_text(encoding="utf-8")

            for old, new in fixes:
                source = source.replace(old, new)

            file_path.write_text(source, encoding="utf-8")


# Imports
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN

import streamlit as st
from PIL import Image
import pickle
import os
import cv2
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Bollywood Celebrity Recognition",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Which Bollywood Celebrity Are You?")
st.write("Upload your photo to find your closest Bollywood celebrity match!")

os.makedirs("uploads", exist_ok=True)


# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    detector = MTCNN()
    model = VGGFace(
        model="resnet50",
        include_top=False,
        input_shape=(224, 224, 3),
        pooling="avg"
    )
    return detector, model


@st.cache_data
def load_data():
    with open("embedding.pkl", "rb") as f:
        feature_list = pickle.load(f)

    with open("filenames.pkl", "rb") as f:
        filenames = pickle.load(f)

    return np.asarray(feature_list), filenames


try:
    detector, model = load_model()
    feature_list, filenames = load_data()

    if len(feature_list) != len(filenames):
        st.error(
            "The number of embeddings and filenames does not match. "
            "Please regenerate them using the same dataset order."
        )
        st.stop()

except Exception as e:
    st.error(f"Could not load the model or dataset: {e}")
    st.stop()


# -------------------- FEATURE EXTRACTION --------------------
def extract_features(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("The uploaded image could not be opened.")

    # OpenCV loads BGR; MTCNN expects RGB images.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)

    if not results:
        raise ValueError(
            "No face detected. Please upload a clear photo "
            "with a visible face."
        )

    # Select the largest detected face.
    face_info = max(
        results,
        key=lambda face: face["box"][2] * face["box"][3]
    )

    x, y, width, height = face_info["box"]

    # Keep the face coordinates inside the image.
    h, w = img_rgb.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + width)
    y2 = min(h, y + height)

    face = img_rgb[y1:y2, x1:x2]

    if face.size == 0:
        raise ValueError("Could not crop the detected face.")

    face_image = Image.fromarray(face).resize((224, 224))
    face_array = np.asarray(face_image).astype("float32")
    face_array = np.expand_dims(face_array, axis=0)

    face_array = preprocess_input(face_array)
    features = model.predict(face_array, verbose=0).flatten()

    return features


# -------------------- CELEBRITY MATCHING --------------------
def recommend(features):
    similarities = cosine_similarity(
        features.reshape(1, -1),
        feature_list
    )[0]

    best_index = int(np.argmax(similarities))
    best_score = float(similarities[best_index])

    return best_index, best_score


def get_celebrity_name(image_path):
    # Expected dataset structure: data/Actor_Name/image.jpg
    normalized_path = os.path.normpath(image_path)
    parent_folder = os.path.basename(os.path.dirname(normalized_path))

    # Convert names such as Salman_Khan to Salman Khan.
    return parent_folder.replace("_", " ")


# -------------------- STREAMLIT UI --------------------
uploaded_image = st.file_uploader(
    "Choose a photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    col1, col2 = st.columns(2)

    display_image = Image.open(uploaded_image).convert("RGB")

    with col1:
        st.subheader("Your uploaded photo")
        st.image(display_image, use_container_width=True)

    with st.spinner("Detecting face and finding your celebrity match..."):
        try:
            upload_path = os.path.join("uploads", uploaded_image.name)

            # Save the uploaded image.
            display_image.save(upload_path)

            # Extract features and find the nearest match.
            features = extract_features(upload_path)
            index_pos, score = recommend(features)

            matched_path = filenames[index_pos]

            if not os.path.exists(matched_path):
                raise FileNotFoundError(
                    f"Dataset image not found: {matched_path}"
                )

            predicted_actor = get_celebrity_name(matched_path)

            with col2:
                st.subheader(f"✨ Your closest match: {predicted_actor}")
                st.image(
                    matched_path,
                    caption=predicted_actor,
                    use_container_width=True
                )
                st.metric(
                    "Cosine similarity",
                    f"{score:.2%}"
                )

            st.caption(
                "This is the closest match in your dataset, "
                "not a guaranteed identification. Similarity is not "
                "the same as prediction accuracy."
            )

        except Exception as e:
            st.error(str(e))
