import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load pre-trained model and data
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# App Title and Description
st.markdown(
    """
    <style>
        .main-title {
            font-size: 45px;
            font-weight: 700;
            color: #2E4053;
            text-align: center;
            margin-bottom: 20px;
        }
        .description {
            font-size: 18px;
            color: #555555;
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-section {
            text-align: center;
            margin-bottom: 20px;
        }
        .recommend-title {
            font-size: 24px;
            color: #1C2833;
            font-weight: 600;
            text-align: center;
            margin-top: 40px;
        }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="main-title">Image Recommender System</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Upload an image, and we\'ll find similar images from our database.</div>', unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File Upload and Recommendation Display
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'], key="fileUploader")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image", use_column_width=True, output_format="PNG")
        
        # Feature extraction
        with st.spinner('Analyzing your image...'):
            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        
        # Get recommendations
        indices = recommend(features, feature_list)
        
        # Display recommendations
        st.markdown('<div class="recommend-title">Similar Images</div>', unsafe_allow_html=True)
        
        # Display recommended images with fixed width to avoid blurriness
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            if idx < len(indices[0]):
                recommended_image = Image.open(filenames[indices[0][idx]])
                col.image(recommended_image, width=150, caption=f"Match {idx + 1}", output_format="PNG")
    else:
        st.error("Some error occurred during file upload. Please try again.")

# CSS for additional styling
st.markdown(
    """
    <style>
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .file-uploader {
            text-align: center;
            margin: 0 auto;
            font-weight: 500;
        }
        .recommend-title {
            font-size: 22px;
            color: #2E86C1;
            text-align: center;
            margin-top: 30px;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True
)
