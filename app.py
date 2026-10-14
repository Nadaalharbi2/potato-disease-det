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
st.write("مشروع بحثي وتطبيقي مشترك بين **جامعة حائل** و**شركة نادك (NADEC)** للكشف عن اللفحة الفطرية وحساب شدة الإصابة آلياً.")

# خانة رفع الصورة
uploaded_file = st.file_uploader("ارفع صورة ورقة البطاطس للفحص:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # قراءة الصورة وعرضها
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    st.image(image, caption="الصورة المدخلة", use_container_width=True)
    
    with st.spinner("جاري تحليل الصورة عبر نموذج الذكاء الاصطناعي..."):
        # 1. تحميل النموذج الم درب
        model = YOLO("best.pt")
        
        # 2. الكشف وتحديد الصناديق
        results = model.predict(source=img_array, conf=0.25)
        res = results[0]
        
        # رسم الصناديق المكتشفة على الصورة
        res_bgr = res.plot()
        res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
        
        # استخراج الفئات المتوقعة
        labels = [model.names[int(b.cls[0])] for b in res.boxes]
        is_infected = any("infect" in l.lower() for l in labels)
        is_healthy = any("health" in l.lower() for l in labels)
        
        st.subheader("نتائج التحليل:")
        st.image(res_rgb, caption="تحديد مناطق الإصابة (YOLOv8)", use_container_width=True)
        
        # 3. حساب شدة الإصابة وتحديد قرار التدخل
        if is_infected:
            # التحويل لفضاء HSV لعزل البقع الداكنة
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            
            # قناع مساحة الورقة وقناع مساحة البقع الفطرية
            mask_leaf = cv2.inRange(hsv, np.array([10, 30, 20]), np.array([90, 255, 255]))
            mask_blight = cv2.inRange(hsv, np.array([10, 40, 20]), np.array([30, 255, 180]))
            
            leaf_area = cv2.countNonZero(mask_leaf)
            blight_area = cv2.countNonZero(mask_blight)
            
            severity_pct = (blight_area / leaf_area * 100) if leaf_area > 0 else 0.0
            
            # بروتوكول التدخل الميداني لنادك
            if severity_pct < 15:
                tier = "خفيف - Tier 1 (< 15%)"
                action = "إزالة الورقة يدوياً (Pinch-off) دون استخدام أي مبيدات كيميائية."
            elif severity_pct <= 35:
                tier = "متوسط - Tier 2 (15% - 35%)"
                action = "رش فطري موجه يقتصر على الصينية أو الطاولة المتضررة فقط."
            else:
                tier = "شديد - Tier 3 (> 35%)"
                action = "حجر وعزل فوري للصينية لمنع انتشار الأبواغ عبر مراوح وسادات التبريد."
            
            st.error("⚠️ الحالة: مصابة باللفحة الفطرية (Infection)")
            st.metric(label="نسبة شدة المرض (Disease Severity)", value=f"{severity_pct:.2f}%")
            st.info(f"**مستوى التدخل المقترح:** {tier}")
            st.warning(f"**الإجراء الميداني الموصى به:** {action}")
            
        elif is_healthy:
            st.success("✅ الحالة: ورقة سليمة تماماً (Healthy)")
            st.write("الورقة خالية من بقع اللفحة؛ لا يلزم أي تدخل كيميائي أو ميكانيكي.")
        else:
            st.warning("لم يتم التعرف على ورقة بوضوح؛ يرجى تجربة التقاط صورة أقرب للورقة وبإضاءة جيدة.")
