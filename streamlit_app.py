import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
import shutil
import zipfile
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Gastric Cancer Classification",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 5px solid #2E8B57;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 5px;
        overflow: hidden;
        height: 20px;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

def _build_model():
    """Rebuild the exact model architecture from GastricCancer.ipynb."""
    base_model = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(8, activation="softmax"),
    ])
    model.build(input_shape=(None, 224, 224, 3))
    return model


@st.cache_resource
def load_model(model_path):
    """Load model by rebuilding architecture and loading weights from .keras file."""
    try:
        model_path = os.path.abspath(model_path)
        cache_dir = os.path.join(os.environ.get("TEMP", "."), "_gc_model_cache")
        os.makedirs(cache_dir, exist_ok=True)

        model = _build_model()

        if model_path.endswith(".keras"):
            extract_dir = os.path.join(cache_dir, "extracted")
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            os.makedirs(extract_dir)

            with zipfile.ZipFile(model_path, "r") as z:
                z.extractall(extract_dir)

            weights_path = os.path.join(extract_dir, "model.weights.h5")
            if not os.path.exists(weights_path):
                for fname in os.listdir(extract_dir):
                    if fname.endswith(".h5"):
                        weights_path = os.path.join(extract_dir, fname)
                        break

            model.load_weights(weights_path)
            shutil.rmtree(extract_dir, ignore_errors=True)

        elif model_path.endswith(".h5"):
            local_copy = os.path.join(cache_dir, os.path.basename(model_path))
            if not os.path.exists(local_copy) or \
               os.path.getmtime(model_path) > os.path.getmtime(local_copy):
                shutil.copy2(model_path, local_copy)
            model.load_weights(local_copy)

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_model_list():
    """Get list of all models in the models folder"""
    models_folder = "models"
    if not os.path.exists(models_folder):
        return []
    
    # Look for common model file extensions
    model_patterns = ["*.keras", "*.h5", "*.pb", "*.tflite"]
    model_files = []
    
    for pattern in model_patterns:
        model_files.extend(glob.glob(os.path.join(models_folder, pattern)))
    
    return [os.path.basename(f) for f in model_files]

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model inference.
    
    NOTE: The model has a built-in Rescaling(1/255) layer,
    so we do NOT normalize here — just resize and expand dims.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    img_array = np.array(image, dtype='float32')
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(model, processed_image):
    """Make prediction on processed image"""
    try:
        predictions = model.predict(processed_image)
        return predictions
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def display_prediction_results(predictions, class_names=None):
    """Display prediction results with confidence bars"""
    if predictions is None:
        return
    
    # Get the prediction probabilities
    if len(predictions.shape) > 1:
        probs = predictions[0]
    else:
        probs = predictions
    
    # If binary classification
    if len(probs) == 1:
        prob_positive = float(probs[0])
        prob_negative = 1 - prob_positive
        
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.subheader("🔍 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Cancer Probability", f"{prob_positive:.2%}")
            st.progress(prob_positive)
        
        with col2:
            st.metric("Normal Probability", f"{prob_negative:.2%}")
            st.progress(prob_negative)
        
        # Overall prediction
        prediction_class = "Cancer Detected" if prob_positive > 0.5 else "Normal Tissue"
        confidence = max(prob_positive, prob_negative)
        
        if prob_positive > 0.5:
            st.error(f"⚠️ {prediction_class} (Confidence: {confidence:.2%})")
        else:
            st.success(f"✅ {prediction_class} (Confidence: {confidence:.2%})")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # If multi-class classification
    else:
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(probs))]
        
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.subheader("🔍 Prediction Results")
        
        # Sort by confidence
        sorted_indices = np.argsort(probs)[::-1]
        
        for i, idx in enumerate(sorted_indices[:3]):  # Top 3 predictions
            class_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
            confidence = float(probs[idx])
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{class_name}")
                st.progress(confidence)
            with col2:
                st.write(f"{confidence:.2%}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Main application
def main():
    st.markdown('<h1 class="main-header">🔬 Gastric Cancer Classification</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for model selection and settings
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        # Get available models
        available_models = get_model_list()
        
        if not available_models:
            st.error("No models found in the 'models' folder!")
            st.info("Please ensure you have model files (.keras, .h5, .pb, .tflite) in the 'models' directory.")
            st.stop()
        
        # Model selection dropdown
        selected_model = st.selectbox(
            "Select Model:",
            available_models,
            help="Choose a model for inference"
        )
        
        st.caption("📐 Input size: 224×224 (ResNet50)")
        
        # Class names from GastricCancer.ipynb
        class_names = ['ADI', 'DEB', 'LYM', 'MUC', 'MUS', 'NOR', 'STR', 'TUM']
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a histopathology image:",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload a histopathology image for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.info(f"📋 Image Info: {image.size[0]}x{image.size[1]} pixels, Mode: {image.mode}")
    
    with col2:
        st.header("🤖 Analysis Results")
        
        if uploaded_file is not None and selected_model:
            # Load model
            model_path = os.path.join("models", selected_model)
            
            with st.spinner("Loading model..."):
                model = load_model(model_path)
            
            if model is not None:
                # Show model info
                st.success(f"✅ Model loaded: {selected_model}")
                
                # Preprocess image
                with st.spinner("Preprocessing image..."):
                    processed_image = preprocess_image(image, target_size=(224, 224))
                
                # Make prediction
                with st.spinner("Analyzing image..."):
                    predictions = predict_image(model, processed_image)
                
                # Display results
                if predictions is not None:
                    display_prediction_results(predictions, class_names)
            else:
                st.error("❌ Failed to load the selected model.")
        
        elif uploaded_file is None:
            st.info("👆 Please upload an image to begin analysis")
        
        else:
            st.info("⚙️ Please select a model from the sidebar")
    
    # Additional information
    st.markdown("---")
    
    with st.expander("ℹ️ About This Application"):
        st.write("""
        This application uses deep learning models to analyze histopathology images for gastric cancer detection.
        
        **How to use:**
        1. Select a model from the sidebar dropdown
        2. Upload a histopathology image
        3. View the analysis results
        
        **Supported formats:** PNG, JPG, JPEG, TIFF, BMP
        
        **Note:** This tool is for research purposes only and should not be used for medical diagnosis without proper validation.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>Built with Streamlit • TensorFlow • For Research Use Only</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()