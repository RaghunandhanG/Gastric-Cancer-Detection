import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
import json
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
    """Rebuild the exact model architecture from GastricCancer.ipynb.

    The model is a Sequential stack:
        Rescaling(1/255) → ResNet50(include_top=False) → GAP → Dropout(0.5) → Dense(8)

    We use weights='imagenet' so the ResNet50 layer names and shapes match
    what was saved during training.  The saved weights will overwrite these
    initial ImageNet weights immediately after.
    """
    base_model = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    # During inference we don't care about trainable flags, but set them
    # the same way the notebook did so layer shapes/names stay consistent.
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
    # Build so weight tensors are created
    model.build(input_shape=(None, 224, 224, 3))
    return model


@st.cache_resource
def load_model(model_path):
    """Load a .keras checkpoint by rebuilding the architecture and loading weights.

    This avoids Keras 2 ↔ 3 serialisation issues entirely — we never
    deserialise the *graph* from the .keras file, only the weight values.
    """
    try:
        model_path = os.path.abspath(model_path)

        # Use a short temp path to avoid Windows long-path / OneDrive issues
        cache_dir = os.path.join(os.environ.get("TEMP", "."), "_gc_model_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # --- rebuild architecture locally ---
        model = _build_model()

        # --- extract & load weights from the .keras ZIP ---
        if model_path.endswith(".keras"):
            extract_dir = os.path.join(cache_dir, "extracted")
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            os.makedirs(extract_dir)

            with zipfile.ZipFile(model_path, "r") as z:
                z.extractall(extract_dir)

            weights_path = os.path.join(extract_dir, "model.weights.h5")
            if not os.path.exists(weights_path):
                # fall back: look for any .h5 inside
                for fname in os.listdir(extract_dir):
                    if fname.endswith(".h5"):
                        weights_path = os.path.join(extract_dir, fname)
                        break

            model.load_weights(weights_path)

            # clean up
            shutil.rmtree(extract_dir, ignore_errors=True)

        elif model_path.endswith(".h5"):
            # Plain HDF5 weight file — load directly
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
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("💡 Make sure the model was trained with the ResNet50 architecture from GastricCancer.ipynb.")
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
    
    NOTE: The model already has a Rescaling(1./255) layer built-in,
    so we do NOT normalize here — just resize and expand dims.
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to exact size used in notebook (224x224)
    image = image.resize(target_size)
    
    # Convert to array — keep as uint8/float, model handles rescaling internally
    img_array = np.array(image, dtype='float32')
    
    # Add batch dimension (matches notebook: tf.expand_dims(img, 0))
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(model, processed_image):
    """Make prediction using the same approach as GastricCancer.ipynb"""
    try:
        # Predict using same method as notebook: model.predict()
        predictions = model.predict(processed_image, verbose=0)
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
        
        st.subheader("🔍 Prediction Results")
        
        # Sort by confidence
        sorted_indices = np.argsort(probs)[::-1]
        
        # Top prediction highlight
        top_idx = sorted_indices[0]
        top_class = class_names[top_idx] if top_idx < len(class_names) else f"Class {top_idx}"
        top_conf = float(probs[top_idx]) * 100
        st.markdown(f"### 🎯 Predicted: **{top_class}** — {top_conf:.2f}%")
        st.divider()
        
        # Top 3 with percentage bars
        st.markdown("**Top 3 Predictions:**")
        for rank, idx in enumerate(sorted_indices[:3], start=1):
            class_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
            confidence = float(probs[idx])
            pct = confidence * 100
            
            st.progress(confidence, text=f"**#{rank} {class_name}** — {pct:.2f}%")
        


# Main application
def main():
    st.markdown('<h1 class="main-header">🔬 Gastric Cancer Classification</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for model selection and settings
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        # Upload new model
        st.subheader("📤 Upload Model")
        uploaded_model = st.file_uploader(
            "Upload a .keras or .h5 model file",
            type=["keras", "h5"],
            help="Upload a model to the models folder for inference",
            key="model_uploader"
        )
        
        if uploaded_model is not None:
            save_path = os.path.join("models", uploaded_model.name)
            if not os.path.exists(save_path):
                os.makedirs("models", exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                st.success(f"✅ Saved: {uploaded_model.name}")
                st.cache_resource.clear()  # Clear cached model so new one can be loaded
                st.rerun()
            else:
                st.info(f"Model `{uploaded_model.name}` already exists.")
        
        st.divider()
        
        # Get available models
        available_models = get_model_list()
        
        if not available_models:
            st.error("No models found in the 'models' folder!")
            st.info("Upload a model above or place .keras/.h5 files in the 'models' directory.")
            st.stop()
        
        # Model selection dropdown
        selected_model = st.selectbox(
            "Select Model:",
            available_models,
            help="Choose a model for inference"
        )
        
        st.caption("📐 Input size: 224×224 (ResNet50)")
        
        # Class names (if known)
        st.subheader("🏷️ Class Labels")
        use_custom_labels = st.checkbox("Use Custom Class Labels")
        
        if use_custom_labels:
            labels_input = st.text_area(
                "Enter class labels (one per line):",
                placeholder="ADI\nDEB\nLYM\nMUC\nMUS\nNOR\nSTR\nTUM",
                value="ADI\nDEB\nLYM\nMUC\nMUS\nNOR\nSTR\nTUM"
            )
            class_names = [label.strip() for label in labels_input.split('\n') if label.strip()]
        else:
            # Exact class names from GastricCancer.ipynb
            class_names = ['ADI','DEB','LYM','MUC','MUS','NOR','STR','TUM']
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a histopathology image:",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload a histopathology image for analysis",
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
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
        This application uses a **ResNet50-based deep learning model** to analyze histopathology images for gastric cancer classification.
        
        **Model Architecture:**
        - Base: ResNet50 (ImageNet pretrained)
        - Global Average Pooling
        - Dropout (0.5)
        - Dense layer (8 classes, softmax)
        
        **8 Tissue Classes:**
        - **ADI**: Adipose tissue
        - **DEB**: Debris
        - **LYM**: Lymphocytes
        - **MUC**: Mucus
        - **MUS**: Muscle
        - **NOR**: Normal tissue
        - **STR**: Stroma
        - **TUM**: Tumor
        
        **How to use:**
        1. Select your trained model from the sidebar
        2. Upload a histopathology image (224x224 recommended)
        3. View detailed classification results
        
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