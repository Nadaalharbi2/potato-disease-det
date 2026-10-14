import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# إعداد واجهة الصفحة
st.set_page_config(
    page_title="منظومة الفحص الذكي لدرنات البطاطس",
    layout="centered"
)

st.title("🌱 منظومة الفحص الصحي الذكي لدرنات البطاطس")
st.write(" مشروع بحثي وتطبيقي مشترك بين **جامعة حائل** و**شركة نادك (NADEC)** منظومة ذكية قائمة على الذكاء الاصطناعي للكشف عن اللفحة الفطرية وتقدير شدة الإصابة آلياً لدعم القرار الزراعي.")

# خانة رفع الصور (تم تفعيل رفع أكثر من صورة)
uploaded_files = st.file_uploader(
    "ارفع صور أوراق البطاطس للفحص (يمكنك اختيار عدة صور معاً):", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    # تحميل النموذج مرة واحدة خارج التكرار لتسريع الأداء
    with st.spinner("جاري تحميل النموذج..."):
        model = YOLO("best.pt")
    
    st.write(f"📊 إجمالي الصور المرفوعة: **{len(uploaded_files)}**")
    st.divider()

    # معالجة كل صورة على حدة
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        st.markdown(f"### 🍃 فحص الصورة رقم ({idx}): `{uploaded_file.name}`")
        
        # قراءة الصورة
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # تقسيم العرض إلى عمودين (الصورة الأصلية vs صورة الكشف)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="الصورة الأصلية", use_container_width=True)
            
        # الكشف وتحديد الصناديق عبر YOLO
        results = model.predict(source=img_array, conf=0.25)
        res = results[0]
        
        res_bgr = res.plot()
        res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.image(res_rgb, caption="تحديد مناطق الإصابة (YOLOv8)", use_container_width=True)
            
        # استخراج الفئات
        labels = [model.names[int(b.cls[0])] for b in res.boxes]
        is_infected = any("infect" in l.lower() for l in labels)
        is_healthy = any("health" in l.lower() for l in labels)
        
        # تقييم النتيجة وتحديد الإجراء الميداني
        if is_infected:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            
            mask_leaf = cv2.inRange(hsv, np.array([10, 30, 20]), np.array([90, 255, 255]))
            mask_blight = cv2.inRange(hsv, np.array([10, 40, 20]), np.array([30, 255, 180]))
            
            leaf_area = cv2.countNonZero(mask_leaf)
            blight_area = cv2.countNonZero(mask_blight)
            
            severity_pct = (blight_area / leaf_area * 100) if leaf_area > 0 else 0.0
            
            if severity_pct < 15:
                tier = "خفيف - Tier 1 (< 15%)"
                action = "إزالة الورقة يدوياً (Pinch-off) دون استخدام أي مبيدات كيميائية."
            elif severity_pct <= 35:
                tier = "متوسط - Tier 2 (15% - 35%)"
                action = "رش فطري موجه يقتصر على الصينية أو الطاولة المتضررة فقط."
            else:
                tier = "شديد - Tier 3 (> 35%)"
                action = "حجر وعزل فوري للصينية لمنع انتشار الأبواغ عبر مراوح وسادات التبريد."
            
            st.error("⚠️ النتيجة: مصابة باللفحة الفطرية (Infection)")
            st.metric(label="نسبة شدة المرض (Disease Severity)", value=f"{severity_pct:.2f}%")
            st.info(f"**مستوى التدخل:** {tier}")
            st.warning(f"**الإجراء الموصى به:** {action}")
            
        elif is_healthy:
            st.success("✅ النتيجة: ورقة سليمة تماماً (Healthy)")
            st.write("الورقة خالية من أعراض المرض؛ لا يلزم أي تدخل.")
        else:
            st.warning("لم يتم التعرف على ورقة بوضوح؛ تأكدي من التقاط صورة أقرب وبإضاءة مناسبة.")
            
        st.divider()
