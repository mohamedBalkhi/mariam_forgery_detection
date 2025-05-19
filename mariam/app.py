# app.py (احفظي هذا الكود كملف Python)

import streamlit as st
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
from PIL import Image
import numpy as np
import os
from datetime import datetime
import io
from PIL import ImageChops, ImageEnhance

# Import gdown for downloading from Google Drive
try:
    import gdown
except ImportError:
    st.warning("Installing gdown package for Google Drive download...")
    import subprocess
    subprocess.check_call(["pip", "install", "gdown"])
    import gdown

# --- إعدادات وتكوينات ---
# نفس إعدادات الصور التي استخدمتيها في التدريب
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# مسار النموذج المحفوظ ومعرف ملف Google Drive
MODEL_PATH = 'models/best_dual_model_robust_head.keras'
GDRIVE_FILE_ID = '1aiLqyEm1kdUbxmjBJfE755H88r5Lu8Hq'

# --- دوال مساعدة ---

# دالة لتنزيل النموذج من Google Drive إذا لم يكن موجودًا محليًا
def download_model_if_needed(model_path, gdrive_file_id):
    """تنزيل النموذج من Google Drive إذا لم يكن موجودًا."""
    if not os.path.exists(model_path):
        url = f'https://drive.google.com/uc?id={gdrive_file_id}'
        st.info("جاري تنزيل النموذج من Google Drive...")
        try:
            gdown.download(url, model_path, quiet=False)
            st.success("تم تنزيل النموذج بنجاح!")
        except Exception as e:
            st.error(f"حدث خطأ أثناء تنزيل النموذج: {e}")
    return os.path.exists(model_path)

# دالة لتحميل النموذج (مع تجنب إعادة التحميل في كل مرة يتغير فيها الإدخال)
@st.cache_resource # استخدم cache_resource للنماذج والموارد الكبيرة
def load_my_model(model_path, gdrive_file_id):
    """تحميل نموذج Keras المحفوظ."""
    if not TF_AVAILABLE:
        st.error("TensorFlow غير مثبت. يرجى تثبيت TensorFlow أولاً.")
        st.code("""
# لأجهزة Mac مع شريحة Apple Silicon (M1/M2/M3):
pip install tensorflow-macos
pip install tensorflow-metal  # لدعم GPU

# لأجهزة Mac مع معالجات Intel أو أجهزة أخرى:
pip install tensorflow
        """)
        return None
    
    # محاولة تنزيل النموذج إذا لم يكن موجودًا
    model_exists = download_model_if_needed(model_path, gdrive_file_id)
    
    if not model_exists:
        st.error(f"لم يتم العثور على ملف النموذج وفشل التنزيل")
        return None
    
    try:
        model = keras.models.load_model(model_path)
        print("النموذج تم تحميله بنجاح")
        return model
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل النموذج: {e}")
        return None

def preprocess_image(image_pil, target_size):
    """معالجة الصورة لتناسب مدخلات النموذج."""
    if not TF_AVAILABLE:
        st.error("لا يمكن معالجة الصورة لأن TensorFlow غير مثبت.")
        return None
        
    # التأكد من أن الصورة RGB (للتوافق مع 3 قنوات)
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")
    
    img_resized = image_pil.resize(target_size)
    img_array = keras.preprocessing.image.img_to_array(img_resized) # (height, width, channels)
    img_array_expanded = np.expand_dims(img_array, axis=0)          # (1, height, width, channels) - لإضافة بُعد الدفعة
    
    # تطبيق نفس المعالجة المسبقة المستخدمة في التدريب
    # هذه الدالة تقوم بتطبيع البكسلات إلى النطاق الذي يتوقعه ResNet50V2
    preprocessed_img = tf.keras.applications.resnet_v2.preprocess_input(img_array_expanded)
    return preprocessed_img

def generate_ela_image(image_pil, quality=90, scale_factor=15):
    temp_buffer = io.BytesIO()
    image_pil.save(temp_buffer, 'JPEG', quality=quality)
    temp_buffer.seek(0)
    resaved_image = Image.open(temp_buffer)
    ela_image = ImageChops.difference(image_pil, resaved_image)
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale_factor)
    temp_buffer.close()
    return ela_image

# --- تحميل النموذج ---
# سيتم تحميل النموذج مرة واحدة فقط بفضل cache_resource
if TF_AVAILABLE:
    model = load_my_model(MODEL_PATH, GDRIVE_FILE_ID)
else:
    model = None
    st.error("TensorFlow غير مثبت. لا يمكن تشغيل التطبيق بدون TensorFlow.")
    st.info("الرجاء تثبيت TensorFlow بالأمر المناسب لنظامك:")
    st.code("""
# لأجهزة Mac مع شريحة Apple Silicon (M1/M2/M3):
pip install tensorflow-macos
pip install tensorflow-metal  # لدعم GPU

# لأجهزة Mac مع معالجات Intel أو أجهزة أخرى:
pip install tensorflow
    """)

# --- واجهة المستخدم لتطبيق Streamlit ---
st.title("تطبيق كشف تزوير الصور الرقمية")
st.write("قم برفع صورة للتحقق مما إذا كانت أصلية أم مزورة.")
st.markdown("---")

# عنصر تحميل الملفات
uploaded_file = st.file_uploader("اختر ملف صورة...", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

if uploaded_file is not None and model is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="الصورة التي تم تحميلها", use_container_width=True)
        st.markdown("---")
        
        if st.button("تحقق من الصورة"):
            with st.spinner("جاري معالجة الصورة والتنبؤ..."):
                # Generate ELA image
                ela_image_generated = generate_ela_image(image, quality=90, scale_factor=15)
                ela_image_resized = ela_image_generated.resize(IMAGE_SIZE)
                ela_image_rgb_pil = ela_image_resized.convert("RGB")  # <--- Ensure 3 channels

                # Preprocess RGB
                rgb_array = keras.preprocessing.image.img_to_array(image.resize(IMAGE_SIZE))
                rgb_array = np.expand_dims(rgb_array, axis=0)
                rgb_preprocessed = tf.keras.applications.resnet_v2.preprocess_input(rgb_array)

                # Preprocess ELA - بدون قسمة على 255، لأن النموذج يحتوي على طبقة Rescaling
                ela_array = keras.preprocessing.image.img_to_array(ela_image_rgb_pil)
                ela_array = np.expand_dims(ela_array, axis=0)
                ela_preprocessed = ela_array  # NO division by 255.0 - النموذج نفسه يحتوي على طبقة للتطبيع

                # Predict
                prediction_proba = model.predict([rgb_preprocessed, ela_preprocessed])[0][0]
                st.write(f"Raw prediction probability from model: {prediction_proba:.8f}") # طباعة القيمة
                
                # تحديد التسمية بناءً على الاحتمالية
                # افترضنا أن 'authentic' -> 0 و 'forged' -> 1
                # إذا كانت الاحتمالية قريبة من 1، فهي مزورة. إذا كانت قريبة من 0، فهي أصلية.
                threshold = 0.5 # عتبة التصنيف
                
                st.markdown("---")
                st.subheader("نتيجة التحقق:")
                
                if prediction_proba > threshold:
                    label = "مزورة (Forged)"
                    confidence_score = prediction_proba * 100
                    st.error(f"**النتيجة:** الصورة تبدو **{label}**")
                    st.info(f"**درجة الثقة (بأنها مزورة):** {confidence_score:.2f}%")
                else:
                    label = "أصلية (Authentic)"
                    confidence_score = (1 - prediction_proba) * 100
                    st.success(f"**النتيجة:** الصورة تبدو **{label}**")
                    st.info(f"**درجة الثقة (بأنها أصلية):** {confidence_score:.2f}%")
                
                st.write(f"(الاحتمالية الأولية من النموذج: {prediction_proba:.4f})")

    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة الصورة أو عرضها: {e}")
        st.error("يرجى التأكد من أن الملف هو ملف صورة صالح.")

elif uploaded_file is None and model is not None:
    st.info("يرجى رفع صورة للبدء.")

elif model is None:
    st.error("لم يتم تحميل النموذج بنجاح. لا يمكن إجراء التنبؤات.")
    st.warning("يرجى مراجعة مسار ملف النموذج والتأكد من توفره وصلاحيته.")

st.markdown("---")
st.sidebar.header("عن التطبيق")
st.sidebar.info(
    "هذا التطبيق يستخدم نموذج ResNet50V2 مدرب مسبقًا لكشف تزوير الصور الرقمية. "
    "تم تدريب النموذج على مجموعة بيانات CASIA."
)
st.sidebar.markdown("---")
current_time = datetime.now()
st.sidebar.markdown("تاريخ اليوم: " + str(current_time.strftime("%Y-%m-%d")))