
import os
import pickle
import uuid

import cv2
import numpy as np
import streamlit as st

from PIL import Image
from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn.metrics.pairwise import cosine_similarity


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Bollywood Celebrity Recognition",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Which Bollywood Celebrity Are You?")
st.write(
    "Upload your photo to find your closest Bollywood celebrity match!"
)


# -------------------- FILE PATHS --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embedding.pkl")
FILENAMES_PATH = os.path.join(BASE_DIR, "filenames.pkl")


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


# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    with open(EMBEDDINGS_PATH, "rb") as f:
        feature_list = pickle.load(f)

    with open(FILENAMES_PATH, "rb") as f:
        filenames = pickle.load(f)

    feature_list = np.asarray(feature_list, dtype=np.float32)

    if feature_list.ndim != 2:
        raise ValueError(
            "embedding.pkl must contain a 2D array of feature vectors."
        )

    if len(feature_list) != len(filenames):
        raise ValueError(
            "The number of embeddings and filenames does not match."
        )

    if len(filenames) == 0:
        raise ValueError("No celebrity images were found in the dataset.")

    if not np.isfinite(feature_list).all():
        raise ValueError("The embeddings contain invalid numeric values.")

    return feature_list, filenames


# -------------------- INITIALIZE --------------------
try:
    if not os.path.exists(EMBEDDINGS_PATH):
        st.error("embedding.pkl was not found in the repository.")
        st.stop()

    if not os.path.exists(FILENAMES_PATH):
        st.error("filenames.pkl was not found in the repository.")
        st.stop()

    with st.spinner("Loading face recognition model..."):
        detector, model = load_model()
        feature_list, filenames = load_data()

except Exception as e:
    st.error("Could not initialize the recognition system.")
    st.exception(e)
    st.stop()


# -------------------- FACE FEATURE EXTRACTION --------------------
def extract_features(pil_image):
    """
    Detect the largest face and generate a VGGFace ResNet50
    feature vector using the same preprocessing as the dataset.
    """

    img_rgb = np.asarray(pil_image.convert("RGB"))

    results = detector.detect_faces(img_rgb)

    if not results:
        raise ValueError(
            "No face detected. Please upload a clear photo "
            "with a visible face."
        )

    # Select the largest detected face.
    face_info = max(
        results,
        key=lambda item: (
            max(0, item["box"][2]) * max(0, item["box"][3])
        )
    )

    x, y, width, height = face_info["box"]

    # Clip face coordinates to image boundaries.
    image_height, image_width = img_rgb.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image_width, x + width)
    y2 = min(image_height, y + height)

    face = img_rgb[y1:y2, x1:x2]

    if face.size == 0:
        raise ValueError(
            "Could not crop the detected face. Try another photo."
        )

    # Resize and prepare the image.
    face_image = Image.fromarray(face).resize((224, 224))
    face_array = np.asarray(face_image, dtype=np.float32)
    face_array = np.expand_dims(face_array, axis=0)

    face_array = preprocess_input(face_array)

    features = model.predict(face_array, verbose=0).flatten()

    if not np.isfinite(features).all():
        raise ValueError("Feature extraction returned invalid values.")

    return features


# -------------------- CELEBRITY MATCHING --------------------
def recommend(features):
    """
    Find the closest embedding using cosine similarity.
    """

    if features.shape[0] != feature_list.shape[1]:
        raise ValueError(
            "Feature dimensions do not match the saved embeddings. "
            "Regenerate embeddings with the same VGGFace model."
        )

    similarities = cosine_similarity(
        features.reshape(1, -1),
        feature_list
    )[0]

    best_index = int(np.argmax(similarities))
    best_score = float(similarities[best_index])

    # Also return the top 3 matches, or fewer if the dataset is small.
    top_indices = np.argsort(similarities)[::-1][:min(3, len(similarities))]

    return best_index, best_score, top_indices, similarities


# -------------------- CELEBRITY NAME --------------------
def get_celebrity_name(image_path):
    """
    Expected structure:
    data/Actor_Name/image.jpg
    """

    normalized_path = os.path.normpath(str(image_path))
    parent_folder = os.path.basename(
        os.path.dirname(normalized_path)
    )

    return parent_folder.replace("_", " ")


# -------------------- USER INTERFACE --------------------
uploaded_image = st.file_uploader(
    "Choose your photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:

    try:
        display_image = Image.open(uploaded_image).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Your Photo")
            st.image(
                display_image,
                use_container_width=True
            )

        with st.spinner(
            "Detecting face and finding your closest celebrity match..."
        ):
            features = extract_features(display_image)

            best_index, best_score, top_indices, similarities = (
                recommend(features)
            )

            matched_path = str(filenames[best_index])

            # Support absolute paths and paths relative to the project.
            if not os.path.isabs(matched_path):
                matched_path = os.path.join(BASE_DIR, matched_path)

            if not os.path.isfile(matched_path):
                raise FileNotFoundError(
                    "The matched dataset image was not found: "
                    + matched_path
                )

            predicted_actor = get_celebrity_name(matched_path)

            with col2:
                st.subheader("✨ Closest Celebrity Match")
                st.image(
                    matched_path,
                    caption=predicted_actor,
                    use_container_width=True
                )

                st.markdown(f"### {predicted_actor}")

                st.metric(
                    "Cosine similarity",
                    f"{best_score:.2%}"
                )

        st.info(
            "This is the closest match in your dataset, not a guaranteed "
            "identification. Cosine similarity is not the same as "
            "prediction accuracy."
        )

        with st.expander("See other close matches"):
            for rank, idx in enumerate(top_indices, start=1):
                path = str(filenames[int(idx)])

                if not os.path.isabs(path):
                    path = os.path.join(BASE_DIR, path)

                if not os.path.isfile(path):
                    continue

                actor = get_celebrity_name(path)

                st.write(
                    f"**{rank}. {actor}** — "
                    f"Similarity: {similarities[int(idx)]:.2%}"
                )

                st.image(
                    path,
                    caption=actor,
                    width=180
                )

    except Exception as e:
        st.error("Unable to recognize a celebrity from this photo.")
        st.warning(str(e))
