# Home.py
import streamlit as st

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Sentiment Analysis Project",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 Sentiment Analysis Web App")
st.markdown("### วิเคราะห์ความรู้สึกจากข้อความรีวิว")

# แสดงข้อมูลโปรเจค
col1, col2 = st.columns(2)  # แบ่งหน้าเป็น 2 คอลัมน์

with col1:
    st.info("**Dataset ที่ใช้**\n\n- IMDB Movie Reviews\n- Amazon Food Reviews")

with col2:
    st.info("**โมเดลที่พัฒนา**\n\n- ML Ensemble (LR + RF + GB)\n- Neural Network (DistilBERT)")

st.markdown("---")
st.markdown("👈 เลือกหน้าจากแถบด้านซ้ายเพื่อดูรายละเอียด")

# pages/1_ML_Model_Info.py
import streamlit as st
import joblib
import matplotlib.pyplot as plt

st.title("📊 โมเดลที่ 1: ML Ensemble")

st.header("1. การเตรียมข้อมูล")
st.markdown("""
- ลบ HTML tags และตัวอักษรพิเศษ
- แปลงเป็นตัวพิมพ์เล็ก
- แปลงข้อความเป็นตัวเลขด้วย **TF-IDF**
""")

st.header("2. อัลกอริทึม")
st.markdown("""
ใช้ **VotingClassifier** ประกอบจาก 3 โมเดล:
| โมเดล | หน้าที่ |
|-------|---------|
| Logistic Regression | หาขอบเขตการตัดสินใจเชิงเส้น |
| Random Forest | สร้าง decision trees หลายต้น |
| Gradient Boosting | เรียนรู้จากข้อผิดพลาดสะสม |
""")

st.header("3. ผลการประเมิน")
# โหลดผลจากไฟล์ที่บันทึกไว้
try:
    import json
    with open('models/ensemble_metrics.json') as f:
        metrics = json.load(f)
    st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
except:
    st.warning("ยังไม่ได้ train โมเดล")

st.header("4. แหล่งอ้างอิง")
st.markdown("""
- IMDB Dataset: Kaggle
- Amazon Dataset: Kaggle  
- Scikit-learn Documentation: https://scikit-learn.org
""")

# pages/3_Test_ML_Model.py
import streamlit as st
import joblib

st.title("🧪 ทดสอบ ML Ensemble Model")

# โหลดโมเดล (โหลดแค่ครั้งเดียวด้วย cache)
@st.cache_resource
def load_model():
    model = joblib.load('models/ensemble_model.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    return model, tfidf

model, tfidf = load_model()

# รับข้อความจากผู้ใช้
user_input = st.text_area(
    "📝 พิมพ์รีวิวภาษาอังกฤษที่นี่:",
    placeholder="e.g. This movie was absolutely amazing!"
)

if st.button("🔍 วิเคราะห์ความรู้สึก"):
    if user_input.strip() == "":
        st.warning("กรุณาพิมพ์ข้อความก่อน")
    else:
        # แปลงข้อความและทำนาย
        text_tfidf = tfidf.transform([user_input])
        prediction = model.predict(text_tfidf)[0]
        probability = model.predict_proba(text_tfidf)[0]

        # แสดงผล
        if prediction == 1:
            st.success(f"😊 **Positive** (ความมั่นใจ: {probability[1]:.1%})")
        else:
            st.error(f"😞 **Negative** (ความมั่นใจ: {probability[0]:.1%})")

# pages/4_Test_NN_Model.py
import streamlit as st
from transformers import pipeline

st.title("🧠 ทดสอบ Neural Network (DistilBERT)")

@st.cache_resource  # โหลดโมเดลแค่ครั้งเดียว
def load_bert():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )

with st.spinner("กำลังโหลด BERT model (อาจใช้เวลา 1-2 นาที)..."):
    bert = load_bert()

user_input = st.text_area("📝 พิมพ์รีวิวภาษาอังกฤษที่นี่:")

if st.button("🔍 วิเคราะห์ด้วย BERT"):
    if user_input.strip() == "":
        st.warning("กรุณาพิมพ์ข้อความก่อน")
    else:
        with st.spinner("กำลังวิเคราะห์..."):
            result = bert(user_input)[0]

        label = result['label']
        score = result['score']

        if label == 'POSITIVE':
            st.success(f"😊 **Positive** (ความมั่นใจ: {score:.1%})")
        else:
            st.error(f"😞 **Negative** (ความมั่นใจ: {score:.1%})")
