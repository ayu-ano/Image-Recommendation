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
import sys

# --------------------------
# Initial Setup and Checks
# --------------------------
def verify_directories():
    """Ensure required directories exist"""
    required_dirs = ['uploads', 'images']
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            st.warning(f"Created missing directory: {directory}")

def verify_data_files():
    """Check if required data files exist"""
    required_files = ['embeddings.pkl', 'filenames.pkl']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        st.error(f"Missing critical files: {', '.join(missing_files)}")
        return False
    return True

# --------------------------
# Model and Data Loading
# --------------------------
@st.cache_resource
def load_model():
    try:
        model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model.trainable = False
        model = tensorflow.keras.Sequential([
            model,
            GlobalMaxPooling2D()
        ])
        st.session_state['model_loaded'] = True
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.session_state['model_loaded'] = False
        return None

def load_embeddings():
    try:
        with open('embeddings.pkl', 'rb') as f:
            feature_list = np.array(pickle.load(f))
        with open('filenames.pkl', 'rb') as f:
            filenames = pickle.load(f)
        
        if len(feature_list) != len(filenames):
            st.error("Mismatch between embeddings and filenames!")
            return None, None
            
        return feature_list, filenames
    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        return None, None

# --------------------------
# Core Functions
# --------------------------
def save_uploaded_file(uploaded_file):
    try:
        upload_path = os.path.join('uploads', uploaded_file.name)
        with open(upload_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"File save failed: {str(e)}")
        return False

def feature_extraction(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return None

def recommend(features, feature_list):
    try:
        if features is None or len(feature_list) == 0:
            return None
            
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        return indices[0]  # Return just the first array
    except Exception as e:
        st.error(f"Recommendation failed: {str(e)}")
        return None

# --------------------------
# UI Components
# --------------------------
def setup_ui():
    st.markdown("""
    <style>
        .main-title {
            font-size: 45px;
            font-weight: 700;
            color: #D4AF37;
            text-align: center;
            margin-bottom: 20px;
        }
        .description {
            font-size: 18px;
            color: #555555;
            text-align: center;
            margin-bottom: 30px;
        }
        .recommend-title {
            font-size: 24px;
            color: #D4AF37;
            font-weight: 600;
            text-align: center;
            margin-top: 40px;
        }
        .stAlert {
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">Image Recommender System</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Upload an image to find visually similar images</div>', unsafe_allow_html=True)

# --------------------------
# Main Application
# --------------------------
def main():
    # Initial setup
    verify_directories()
    setup_ui()
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("Debug Mode")
    
    if not verify_data_files():
        st.stop()
    
    # Load model and data
    model = load_model()
    feature_list, filenames = load_embeddings()
    
    if model is None or feature_list is None:
        st.stop()
    
    if debug_mode:
        st.sidebar.subheader("Debug Information")
        st.sidebar.write(f"Embeddings shape: {feature_list.shape}")
        st.sidebar.write(f"Number of images: {len(filenames)}")
        st.sidebar.write(f"First 5 filenames: {filenames[:5]}")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            try:
                # Display uploaded image
                display_image = Image.open(uploaded_file)
                st.image(display_image, caption="Uploaded Image", width=200)
                
                # Feature extraction
                with st.spinner('Analyzing your image...'):
                    features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
                
                if features is not None:
                    # Get recommendations
                    indices = recommend(features, feature_list)
                    
                    if indices is not None:
                        st.markdown('<div class="recommend-title">Similar Images</div>', unsafe_allow_html=True)
                        
                        # Display recommendations in a grid
                        cols = st.columns(5)
                        for idx, col in enumerate(cols):
                            if idx < len(indices):
                                try:
                                    img_path = filenames[indices[idx]]
                                    if os.path.exists(img_path):
                                        recommended_image = Image.open(img_path)
                                        col.image(recommended_image, width=150, caption=f"Match {idx + 1}")
                                    else:
                                        col.error(f"File not found: {img_path}")
                                except Exception as e:
                                    col.error(f"Error loading image: {str(e)}")
                    else:
                        st.warning("No recommendations could be generated.")
                else:
                    st.warning("Feature extraction failed - cannot generate recommendations.")
                
                # Clean up upload
                try:
                    os.remove(os.path.join("uploads", uploaded_file.name))
                except:
                    pass
                    
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                if debug_mode:
                    st.exception(e)

if __name__ == "__main__":
    main()