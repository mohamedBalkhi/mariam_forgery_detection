# app.py
"""
Streamlit web application for digital image forgery detection.

This application allows users to upload an image (JPEG, PNG, TIFF, BMP)
and uses a pre-trained dual-input (RGB and ELA) Convolutional Neural Network
(CNN) model to predict whether the image is authentic or tampered.
The model leverages dual ResNet50V2 backbones for feature extraction.
Error Level Analysis (ELA) is performed
on the uploaded image to provide an additional input stream to the model.

The application handles model downloading from Google Drive if not found
locally (with optional checksum verification), provides UI elements for image upload and
result display, and includes information about the technology used.
TensorFlow/Keras is required for model operations.
"""

import streamlit as st
import hashlib
import logging
from pathlib import Path
import warnings
from PIL import Image, ImageChops, ImageEnhance, UnidentifiedImageError
import numpy as np
from datetime import datetime
import io

# Suppress TensorFlow warnings for cleaner output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)

try:
    import tensorflow as tf
    import keras
    TF_AVAILABLE = True
    # Suppress specific TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
except ImportError:
    TF_AVAILABLE = False
    # Mock keras for type hinting if TF is not available
    class keras: # type: ignore
        class layers:
            Layer = object
        class Model:
            pass
        class saving:
            def register_keras_serializable(self):
                def decorator(cls):
                    return cls
                return decorator
        class models:
            @staticmethod
            def load_model(*args, **kwargs):
                return None


# --- Application Configuration ---

# Image processing parameters
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)  # For model input: (height, width)
IMAGE_SIZE_PIL = (IMG_WIDTH, IMG_HEIGHT) # For PIL.Image.resize: (width, height)
ELA_QUALITY = 90  # JPEG quality for ELA generation
ELA_SCALE_FACTOR = 15  # Brightness enhancement for ELA image
MAX_FILE_SIZE_MB = 25 # Max upload file size in MB
MAX_IMAGE_DIMENSION = 8000 # Max width or height for an uploaded image

# Model details
MODEL_DIR = Path('models')
MODEL_NAME = 'best_dual_model_robust_head.keras'
MODEL_PATH = MODEL_DIR / MODEL_NAME  # Local path to the Keras model
GDRIVE_FILE_ID = '1aiLqyEm1kdUbxmjBJfE755H88r5Lu8Hq'  # Google Drive ID for model download

# Checksum verification - set to None to bypass
BYPASS_CHECKSUM = True  # Set to True to bypass checksum verification
MODEL_CHECKSUM = "e0b9a9d98c8a6f62d7f2b8c49f9e1a1b0a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d" # Example Checksum

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log', # Logs to app.log file
    filemode='a'
)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="كاشف تزوير الصور",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and Right-to-Left (RTL) layout support.
st.markdown("""
<style>
    /* Main Content Area Wrapper - For RTL text and centered layout */
    .rtl-main-area {
        direction: rtl;
        text-align: right;
        max-width: 950px;
        margin: 0 auto;
        padding: 0 1rem;
    }

    /* Headers */
    .app-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .app-subheader {
        font-size: 1.3rem;
        color: #555;
        text-align: center;
        margin-bottom: 2.5rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #3498db;
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 0.5rem;
    }

    /* File Uploader Area */
    .upload-box {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        text-align: center;
    }

    /* Verify Button - Updated for better compatibility */
    button[kind="primary"] { 
        min-width: 220px !important;
        font-size: 1.2rem !important;
        font-weight: bold !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        background-color: #3498db !important;
        color: white !important;
        border: none !important;
    }
    button[kind="primary"]:hover {
        background-color: #2980b9 !important;
    }

    /* Result Display */
    .result-box {
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin: 1rem auto;
        text-align: center;
        border-width: 2px;
        border-style: solid;
        max-width: 600px;
    }
    .result-box h3 {
        font-size: 1.6rem;
        font-weight: bold;
        margin-bottom: 0.5rem !important;
    }
    .authentic {
        background-color: #e6ffed;
        border-color: #5cb85c;
        color: #3c763d;
    }
    .authentic h3 {
        color: #3c763d !important;
    }
    .forged {
        background-color: #ffe6e6;
        border-color: #d9534f;
        color: #a94442;
    }
    .forged h3 {
        color: #a94442 !important;
    }

    /* Sidebar Styling - Updated for better compatibility */
    section[data-testid="stSidebar"] {
        direction: rtl; 
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        text-align: right !important;
        color: #0056b3; 
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li {
        text-align: right !important;
        font-size: 0.95rem;
        color: #333;
        margin-bottom: 0.5rem;
        line-height: 1.6;
    }
    section[data-testid="stSidebar"] ul,
    section[data-testid="stSidebar"] ol {
        padding-right: 20px;
        padding-left: 0;
        margin-right: 10px; 
    }
    .sidebar-content {
        margin-bottom: 1.5rem;
    }
    section[data-testid="stSidebar"] img { 
        display: block;
        margin-left: auto;
        margin-right: auto;
        margin-bottom: 0.5rem; 
    }

    /* Metric styling */
    div[data-testid="metric-container"] {
        background-color: transparent !important;
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center; 
        margin: 1rem auto;
        width: 280px; 
        direction: rtl; 
    }
    .stMetric label { 
        font-size: 1rem;
        color: #555;
        text-align: right !important; 
        display: block; 
        margin-bottom: 0.25rem;
    }
    .stMetric p { 
        font-size: 1.8rem !important;
        font-weight: bold;
        color: #333 !important;
        margin-bottom: 0 !important;
        text-align: center !important; 
    }

    /* Expander */
    .stExpander {
        border: 1px solid #eee;
        border-radius: 8px;
        margin: 1.5rem auto;
        max-width: 600px;
    }
    .stExpander header { 
        font-size: 1.1rem;
        text-align: right !important; 
    }
    .stExpander div[data-testid="stMarkdownContainer"] p,
    .stExpander div[data-testid="stMarkdownContainer"] ul,
    .stExpander div[data-testid="stMarkdownContainer"] li {
        direction: rtl;
        text-align: right !important;
    }
    .stExpander div[data-testid="stMarkdownContainer"] ul {
        padding-right: 20px; 
        padding-left: 0;
    }

    .centered-text {
        text-align: center;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Check for gdown dependency
try:
    import gdown
except ImportError:
    logger.warning("gdown package not found. Please install it: pip install gdown")
    st.error("❌ حزمة 'gdown' غير مثبتة. هذه الحزمة ضرورية لتنزيل النموذج.")
    st.info("الرجاء تثبيت الحزمة يدوياً عن طريق تشغيل الأمر التالي في الطرفية الخاصة بك ثم إعادة تشغيل التطبيق:")
    st.code("pip install gdown")
    st.stop()


@keras.saving.register_keras_serializable()
class SRMLayer(keras.layers.Layer):
    """
    A Keras layer implementing Spatial Rich Model (SRM) filters.

    SRM filters are high-pass filters designed to extract subtle noise-like
    features from images, which can be indicative of manipulation. This layer
    applies a predefined set of SRM kernels as a 2D convolution.
    The kernels used are standard SRM filters (k1, k2, k3) for basic
    manipulation detection.

    The layer is registered as a Keras serializable object to allow
    model saving and loading.
    """
    def __init__(self, **kwargs):
        """
        Initializes the SRMLayer.

        Sets up the SRM filter kernels. These kernels are designed to
        capture high-frequency noise patterns.
        k1: A simple identity-like filter.
        k2: A simple negative identity-like filter.
        k3: A Laplacian-like filter often used in SRM.
        The kernels are stacked and reshaped to be compatible with 3-channel
        (RGB) images.
        """
        super().__init__(**kwargs)
        k1 = [[0,0,0],[0,1,0],[0,0,0]]
        k2 = [[0,0,0],[0,-1,0],[0,0,0]]
        k3 = [[-1,2,-1],[2,-4,2],[-1,2,-1]]
        
        ker = tf.constant([k1, k2, k3], tf.float32)      # Shape: (3, 3, 3)
        ker = tf.reshape(ker, (3,3,1,3))                 # Shape: (3, 3, 1, 3)
        ker = tf.repeat(ker, 3, axis=2)                  # Shape: (3, 3, 3, 3) (Input channels, Output feature maps)
        self.kernel = ker

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Applies the SRM convolution to the input tensor.

        Args:
            x: Input tensor, typically an image batch (B, H, W, C_in).

        Returns:
            Tensor with SRM features, with absolute values taken.
            Shape is (B, H, W, C_out), where C_out is determined by kernels.
        """
        x = tf.nn.conv2d(x, self.kernel, strides=1, padding='SAME')
        return tf.math.abs(x)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """
        Computes the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Shape of the output tensor.
        """
        return tf.TensorShape([
            input_shape[0],
            input_shape[1],
            input_shape[2],
            self.kernel.shape[-1] # Number of output filters
        ])


# --- Helper Functions ---
def calculate_sha256(filepath: Path) -> str:
    """Calculates the SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_model_if_needed(model_path: Path, gdrive_file_id: str, expected_checksum: str = None) -> bool:
    """
    Downloads the model file from Google Drive if it doesn't exist locally
    and optionally verifies its checksum.

    Args:
        model_path: The local path (Path object) where the model should be saved.
        gdrive_file_id: The Google Drive file ID for the model.
        expected_checksum: The expected SHA256 checksum for the model file (optional).

    Returns:
        True if the model file exists locally and checksum matches (if verification enabled), 
        False otherwise.
    """
    models_dir = model_path.parent
    if not models_dir.exists():
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating models directory {models_dir}: {e}", exc_info=True)
            st.error("❌ خطأ في إنشاء مجلد النماذج. يرجى التحقق من الأذونات.")
            return False

    if not model_path.exists():
        url = f'https://drive.google.com/uc?id={gdrive_file_id}'
        status_text = st.empty()
        status_text.info("⏬ جاري تنزيل النموذج من Google Drive... يرجى الانتظار.")
        
        try:
            with st.spinner("Downloading model... This may take a few moments."):
                 gdown.download(url, str(model_path), quiet=False) # quiet=False shows gdown's progress

            # Checksum verification (optional)
            if not BYPASS_CHECKSUM and expected_checksum:
                downloaded_checksum = calculate_sha256(model_path)
                if downloaded_checksum != expected_checksum:
                    logger.error(f"Model checksum mismatch for {model_path}. Expected {expected_checksum}, got {downloaded_checksum}")
                    st.error("❌ خطأ: التحقق من سلامة النموذج فشل. قد يكون الملف تالفًا أو تم التلاعب به.")
                    if model_path.exists():
                        model_path.unlink() # Delete corrupted/tampered file
                    status_text.empty()
                    return False
            
            status_text.success("✅ تم تنزيل النموذج بنجاح!")
            status_text.empty() # Clear message after a short while or next action
        except Exception as e:
            logger.error(f"Error downloading model from GDrive ID {gdrive_file_id} to {model_path}: {e}", exc_info=True)
            st.error("❌ حدث خطأ أثناء تنزيل النموذج. يرجى التحقق من اتصالك بالإنترنت.")
            if model_path.exists(): # Clean up partially downloaded file
                try:
                    model_path.unlink()
                except OSError as unlink_e:
                    logger.error(f"Error deleting partially downloaded model {model_path}: {unlink_e}", exc_info=True)
            status_text.empty()
            return False
    else: # Model already exists, optionally verify its checksum
        if not BYPASS_CHECKSUM and expected_checksum:
            try:
                existing_checksum = calculate_sha256(model_path)
                if existing_checksum != expected_checksum:
                    logger.warning(f"Existing model {model_path} has checksum mismatch. Expected {expected_checksum}, got {existing_checksum}. Re-downloading.")
                    if model_path.exists():
                        model_path.unlink()
                    return download_model_if_needed(model_path, gdrive_file_id, expected_checksum) # Recurse to download
            except Exception as e:
                logger.error(f"Error calculating checksum for existing model {model_path}: {e}", exc_info=True)
                st.warning("⚠️ خطأ في التحقق من النموذج الموجود. سيتم تجاهل التحقق من Checksum.")

    return model_path.exists()

@st.cache_resource
def load_my_model(model_path: Path, gdrive_file_id: str, expected_checksum: str = None) -> keras.Model | None:
    """
    Loads the Keras model for image forgery detection.

    Args:
        model_path: The local path (Path object) to the Keras model file.
        gdrive_file_id: The Google Drive file ID for downloading the model.
        expected_checksum: The expected SHA256 checksum for the model file (optional).

    Returns:
        The loaded Keras model if successful, otherwise None.
    """
    if not TF_AVAILABLE:
        st.error("❌ TensorFlow غير مثبت. يرجى تثبيت TensorFlow أولاً.")
        st.code("""
# لأجهزة Mac مع شريحة Apple Silicon (M1/M2/M3):
pip install tensorflow-macos tensorflow-metal

# لأجهزة أخرى (Windows/Linux/Mac Intel):
pip install tensorflow
        """)
        return None
    
    if not download_model_if_needed(model_path, gdrive_file_id, expected_checksum):
        return None 
    
    model_loading_message = st.empty()
    try:
        model_loading_message.info("⏳ جاري تحميل النموذج...")
        model = keras.models.load_model(
            str(model_path), # Keras expects string path
            compile=False # Skip compiling optimizer/loss for inference
        )
        model_loading_message.success("✅ تم تحميل النموذج بنجاح!")
        model_loading_message.empty()
        return model
    except Exception as e:
        logger.error(f"Error loading Keras model from {model_path}: {e}", exc_info=True)
        model_loading_message.error("❌ حدث خطأ أثناء تحميل النموذج. تحقق من ملف النموذج.")
        return None

# ------------------  PRE-PROCESSING HELPERS  ------------------

def generate_ela_image(pil_img: Image.Image, quality: int = ELA_QUALITY, scale_factor: float = ELA_SCALE_FACTOR) -> Image.Image:
    """
    Generates an Error Level Analysis (ELA) image from a PIL image.

    Args:
        pil_img: The original PIL Image object (should be in RGB mode).
        quality: The JPEG quality (0-100) for re-compression.
        scale_factor: Factor to enhance brightness of the difference image.

    Returns:
        A PIL Image object representing the ELA image.
    """
    # Ensure image is RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    # Save to buffer as JPEG
    temp_buffer = io.BytesIO()
    pil_img.save(temp_buffer, 'JPEG', quality=quality)
    temp_buffer.seek(0)
    try:
        resaved = Image.open(temp_buffer)
    except UnidentifiedImageError as e:
        logger.error(f"Pillow UnidentifiedImageError during ELA resave: {e}", exc_info=True)
        st.error("حدث خطأ أثناء معالجة الصورة لـ ELA. قد تكون الصورة تالفة.")
        return Image.new("RGB", pil_img.size, (0,0,0))
    ela = ImageChops.difference(pil_img, resaved)
    ela = ImageEnhance.Brightness(ela).enhance(scale_factor)
    temp_buffer.close()
    return ela

def preprocess_image_rgb(pil_img: Image.Image, target_size_pil: tuple = IMAGE_SIZE_PIL) -> np.ndarray:
    """
    Preprocesses an RGB PIL Image for model input.

    Args:
        pil_img: The input PIL Image object.
        target_size_pil: A tuple (width, height) for resizing using PIL.

    Returns:
        A NumPy array representing the preprocessed image. Shape: (1, height, width, 3).
    """
    # Ensure image is RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(target_size_pil)
    arr = keras.preprocessing.image.img_to_array(pil_img)
    arr = tf.keras.applications.resnet_v2.preprocess_input(arr)
    return np.expand_dims(arr, 0)

def preprocess_image_ela(ela_pil: Image.Image, target_size_pil: tuple = IMAGE_SIZE_PIL) -> np.ndarray:
    """
    Preprocesses an ELA (Error Level Analysis) PIL Image for model input.

    Args:
        ela_pil: The input ELA PIL Image object.
        target_size_pil: A tuple (width, height) for resizing using PIL.

    Returns:
        A NumPy array representing the preprocessed ELA image. Shape: (1, height, width, 3).
    """
    if ela_pil.mode != "RGB":
        ela_pil = ela_pil.convert("RGB")
    ela_pil = ela_pil.resize(target_size_pil)
    arr = keras.preprocessing.image.img_to_array(ela_pil)  # values 0-255
    return np.expand_dims(arr, 0)

# Removed deprecated tf.function with experimental_relax_shapes
# Using standard tf.function without deprecated parameters
if TF_AVAILABLE:
    @tf.function
    def _tf_function_predict(model: keras.Model, rgb_input: tf.Tensor, ela_input: tf.Tensor) -> tf.Tensor:
        """Helper function to run model prediction within tf.function."""
        return model([rgb_input, ela_input], training=False)

def verify_image(pil_rgb: Image.Image, model: keras.Model) -> tuple[float, Image.Image | None]:
    """
    Verifies an image for tampering using the loaded dual-input model.

    Args:
        pil_rgb: The original PIL Image object (RGB) to verify.
        model: The loaded Keras model for prediction.

    Returns:
        A tuple containing:
            - proba (float): The raw probability score from the model.
            - ela_pil (Image.Image | None): The generated ELA PIL image, or None if ELA fails.
    """
    ela_pil = generate_ela_image(pil_rgb, quality=ELA_QUALITY, scale_factor=ELA_SCALE_FACTOR)
    if ela_pil is None or isinstance(ela_pil, tuple): # Defensive check if generate_ela_image changes error handling
        logger.error("ELA image generation failed.")
        return 0.0, None # Or handle error appropriately

    rgb_arr = preprocess_image_rgb(pil_rgb, target_size_pil=IMAGE_SIZE_PIL)
    ela_arr = preprocess_image_ela(ela_pil, target_size_pil=IMAGE_SIZE_PIL)
    
    if TF_AVAILABLE:
        # Use model.predict() which is the recommended approach for inference
        proba = model.predict([rgb_arr, ela_arr], verbose=0)[0,0]
    else: # Fallback if TF somehow became unavailable after initial check (should not happen)
        logger.error("TensorFlow became unavailable during verify_image.")
        st.error("خطأ حرج في TensorFlow.")
        return 0.0, ela_pil # Return default/error state

    return float(proba), ela_pil

@st.cache_resource
def initialize_app() -> keras.Model | None:
    """
    Initializes the application by loading the forgery detection model.

    Returns:
        The loaded Keras model if successful, otherwise None.
    """
    if TF_AVAILABLE:
        # Show bypass warning if checksum is bypassed
        if BYPASS_CHECKSUM:
            st.info("ℹ️ تم تجاهل التحقق من Checksum النموذج مؤقتاً للتطوير.", icon="ℹ️")

        # Pass None for checksum if bypassed
        checksum_to_use = None if BYPASS_CHECKSUM else MODEL_CHECKSUM
        model = load_my_model(MODEL_PATH, GDRIVE_FILE_ID, checksum_to_use)
        if model is None:
            st.error("❌ فشل تحميل النموذج. لا يمكن تشغيل التطبيق.")
            return None
        return model
    else:
        st.error("❌ TensorFlow غير مثبت. لا يمكن تشغيل التطبيق بدون TensorFlow.")
        st.warning("⚠️ الرجاء تثبيت TensorFlow بالأمر المناسب لنظامك:")
        st.code("""
# لأجهزة Mac مع شريحة Apple Silicon (M1/M2/M3):
pip install tensorflow-macos tensorflow-metal

# لأجهزة أخرى (Windows/Linux/Mac Intel):
pip install tensorflow
        """)
        return None

# --- واجهة المستخدم لتطبيق Streamlit ---
def main():
    """
    Main function to run the Streamlit application for image forgery detection.
    """
    # --- Sidebar Content ---
    with st.sidebar:
        st.image("https://img.icons8.com/external-flaticons-lineal-color-flat-icons/64/000000/external-detective-literature-flaticons-lineal-color-flat-icons.png", width=70) 

        st.markdown('<h2 style="text-align:center; margin-bottom:1rem;">عن التطبيق</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sidebar-content">
        يهدف هذا التطبيق التجريبي إلى كشف التلاعب في الصور الرقمية باستخدام تقنيات التعلم العميق. يتم تحليل كل صورة عبر مسارين: الألوان الأصلية (RGB) وتحليل مستويات الخطأ (ELA)، للكشف عن أدق التعديلات.
        <br><br>
        النموذج الحالي تم تدريب رأسه المخصص على مجموعة بيانات CASIA v2.0، مع الاستفادة من شبكتي ResNet50V2 (مدربة مسبقًا على ImageNet) كقاعدة مجمدة لاستخلاص الميزات.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<h3 style="text-align:center; margin-top:2rem; margin-bottom:1rem;">كيفية الاستخدام</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sidebar-content">
        <ol>
            <li>اختر صورة من جهازك (JPG, PNG, TIFF, BMP).</li>
            <li>اضغط على زر "تحقق من الصورة الآن!".</li>
            <li>انتظر ظهور النتيجة وتحليل ELA.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<h3 style="text-align:center; margin-top:2rem; margin-bottom:1rem;">التكنولوجيا المستخدمة</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sidebar-content">
        <ul>
            <li><b>النموذج:</b> شبكة CNN بإدخال مزدوج (ResNet50V2 x2).</li>
            <li><b>القواعد الأساسية:</b> ResNet50V2 (ImageNet، مجمدة).</li>
            <li><b>رأس النموذج:</b> طبقات تصنيف مُدربة خصيصًا.</li>
            <li><b>المدخلات:</b> RGB و ELA.</li>
            <li><b>بيانات التدريب (للرأس):</b> CASIA v2.0.</li>
            <li><b>الأدوات:</b> TensorFlow, Keras, Streamlit, Pillow, gdown.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="sidebar-content" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-bottom:0.3rem;">إعداد الطالبة</h4>', unsafe_allow_html=True) 
        st.markdown('**مريم عبد العال | Mariam Abd Alaal**', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        current_time = datetime.now()
        st.markdown(f'<p style="text-align: center; font-size: 0.85em; color:#666;">📅 {current_time.strftime("%Y-%m-%d %H:%M")}</p>', unsafe_allow_html=True)

        # Decision threshold slider - value will be used by main logic
        decision_threshold_from_slider = st.slider(
            "عتبة القرار (تعتبر الصورة مزورة إذا كانت النتيجة ≥ العتبة)",
            min_value=0.05, max_value=0.95, value=0.45, step=0.01,
            key="threshold_slider"
        )
    # --- End of Sidebar Content ---

    model = initialize_app()

    st.markdown('<div class="rtl-main-area">', unsafe_allow_html=True) 

    st.markdown('<h1 class="app-header">🔍 كاشف تزوير الصور الرقمية</h1>', unsafe_allow_html=True)
    st.markdown('<p class="app-subheader">اكتشف ما إذا كانت صورك أصلية أم تم التلاعب بها بدقة عالية</p>', unsafe_allow_html=True)

    if model is not None:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header" style="margin-top:0; border-bottom:none; margin-bottom:1.5rem;">ارفع صورة للتحليل</h2>', unsafe_allow_html=True)
        
        _col_fu_left, col_fu_mid, _col_fu_right = st.columns([1,2,1])
        with col_fu_mid:
            uploaded_file = st.file_uploader(
                " ", 
                type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], 
                label_visibility="collapsed", 
                key="main_file_uploader"
            )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            # File size check
            if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"❌ حجم الملف يتجاوز الحد المسموح به ({MAX_FILE_SIZE_MB} ميجابايت).")
                logger.warning(f"User uploaded file larger than limit: {uploaded_file.name}, size: {uploaded_file.size}")
                img_rgb_pil = None
            else:
                try:
                    img_rgb_pil = Image.open(uploaded_file)
                    # Image dimension check
                    if img_rgb_pil.width > MAX_IMAGE_DIMENSION or img_rgb_pil.height > MAX_IMAGE_DIMENSION:
                        st.error(f"❌ أبعاد الصورة كبيرة جدًا (الحد الأقصى {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} بكسل).")
                        logger.warning(f"User uploaded image with excessive dimensions: {uploaded_file.name}, {img_rgb_pil.size}")
                        img_rgb_pil = None
                    elif img_rgb_pil.mode != "RGB": # Ensure it's RGB after opening
                         img_rgb_pil = img_rgb_pil.convert("RGB")

                except UnidentifiedImageError:
                    st.error("❌ لا يمكن التعرف على ملف الصورة. يرجى رفع ملف صورة صالح.")
                    logger.warning(f"User uploaded an unidentifiable image file: {uploaded_file.name}", exc_info=True)
                    img_rgb_pil = None
                except Exception as e:
                    st.error("❌ خطأ في تحميل الصورة. يرجى المحاولة بصورة أخرى.")
                    logger.error(f"Error loading uploaded image {uploaded_file.name}: {e}", exc_info=True)
                    img_rgb_pil = None

            if img_rgb_pil:
                col_rgb, col_ela = st.columns(2, gap="large")

                with col_rgb:
                    st.markdown("#### الصورة الأصلية")
                    st.image(img_rgb_pil, use_container_width=True, caption="الصورة المرفوعة")

                with col_ela:
                    st.markdown("#### صورة تحليل مستوى الخطأ (ELA)")
                    ela_slot = st.empty()
                    ela_slot.info("اضغط زر التحقق لإظهار صورة ELA.")

                check_btn_center = st.columns([1,1,1])[1]
                with check_btn_center:
                    if st.button("🔬 تحقق من الصورة الآن!", key="verify_button", use_container_width=True):
                        with st.spinner("⏳ جاري التحليل..."):
                            proba, ela_pil_result = verify_image(img_rgb_pil, model)

                        if ela_pil_result:
                            with col_ela: # Update ELA image display
                                ela_slot.image(ela_pil_result, use_container_width=True, caption="صورة ELA الناتجة")
                        else:
                            with col_ela:
                                ela_slot.error("لم يتمكن من إنشاء صورة ELA.")

                        st.markdown("---")
                        st.markdown('<h2 class="section-header">📜 نتيجة التحليل</h2>', unsafe_allow_html=True)
                        
                        # Use threshold from sidebar slider
                        if proba >= decision_threshold_from_slider:
                            conf = proba * 100
                            st.markdown('<div class="result-box forged"><h3>⚠️ الصورة تبدو مزورة</h3></div>', unsafe_allow_html=True)
                            st.metric("درجة الثقة (مزورة)", f"{conf:.2f}%")
                        else:
                            conf = (1 - proba) * 100
                            st.markdown('<div class="result-box authentic"><h3>✅ الصورة تبدو أصلية</h3></div>', unsafe_allow_html=True)
                            st.metric("درجة الثقة (أصلية)", f"{conf:.2f}%")

                        with st.expander("تفاصيل إضافية"):
                            st.write(f"النتيجة الأولية من النموذج: **{proba:.6f}**")
                            st.write(f"عتبة القرار المستخدمة: **{decision_threshold_from_slider:.2f}**")
                            st.caption("إذا كانت النتيجة الأولية ≥ عتبة القرار، تعتبر الصورة مزورة.")
            
            elif uploaded_file and not img_rgb_pil: # Error occurred during file processing
                 pass # Error messages handled above
        
        else: # No file uploaded
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background-color: #e9ecef; border-radius: 10px; margin: 2rem auto; max-width: 700px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
                <img src="https://img.icons8.com/fluency/96/000000/image-file.png" alt="Upload icon" width="70" style="margin-bottom: 1rem;">
                <p style="font-size: 1.3rem; margin-bottom: 0.5rem;">لم يتم رفع أي صورة بعد.</p>
                <p style="font-size: 1rem; color: #495057;">يرجى استخدام منطقة التحميل أعلاه لاختيار صورة والبدء في عملية التحقق.</p>
            </div>
            """, unsafe_allow_html=True)
    else: 
        st.error("❌ خطأ فادح: لم يتم تحميل نموذج الذكاء الاصطناعي. لا يمكن متابعة عملية التحقق.")
        st.info("الرجاء التأكد من اتصالك بالإنترنت لتنزيل النموذج إذا كانت هذه هي المرة الأولى، أو تحقق من مسار النموذج المحلي وإعدادات TensorFlow.")
        logger.critical("Application cannot run because the model failed to initialize.")

    st.markdown('</div>', unsafe_allow_html=True) # End of rtl-main-area

if __name__ == "__main__":
    # Note on TensorFlow optimization:
    # Ensure TensorFlow is installed with optimizations for your CPU (e.g., oneDNN for Intel)
    # or with GPU support if available. This can significantly impact model inference speed.
    # For advanced users: Model quantization (e.g., to INT8) can further speed up inference
    # at the cost of a small potential accuracy drop.
    main()