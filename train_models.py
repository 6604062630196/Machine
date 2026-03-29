# train_models.py
import pandas as pd
import joblib  # ใช้บันทึก/โหลดโมเดล
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from utils.preprocessing import load_imdb, load_amazon

# ── โหลดข้อมูล ──────────────────────────────────────────
df_imdb = load_imdb('datasets/imdb.csv')
df_amazon = load_amazon('datasets/amazon.csv')

# รวม 2 dataset เข้าด้วยกัน
df = pd.concat([df_imdb, df_amazon], ignore_index=True)
df = df.dropna()  # ลบแถวที่มีค่าว่าง

# ── แบ่งข้อมูล Train/Test ──────────────────────────────
# test_size=0.2 หมายถึงใช้ 20% เป็นข้อมูลทดสอบ
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'],
    test_size=0.2, random_state=42
)

# ── แปลงข้อความเป็นตัวเลข (TF-IDF) ───────────────────
# TF-IDF = นับความสำคัญของแต่ละคำในข้อความ
tfidf = TfidfVectorizer(max_features=5000)  # เก็บแค่ 5000 คำที่สำคัญที่สุด
X_train_tfidf = tfidf.fit_transform(X_train)  # เรียนรู้จาก train set
X_test_tfidf = tfidf.transform(X_test)         # แปลง test set ด้วย pattern เดิม

# ── โมเดลที่ 1: Ensemble (VotingClassifier) ───────────
# ประกอบจาก 3 โมเดล: Logistic Regression + Random Forest + Gradient Boosting
model_lr = LogisticRegression(max_iter=1000)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

ensemble = VotingClassifier(
    estimators=[('lr', model_lr), ('rf', model_rf), ('gb', model_gb)],
    voting='soft'  # ใช้ค่าความน่าจะเป็นในการโหวต (แม่นกว่า hard voting)
)

print("กำลัง train Ensemble model...")
ensemble.fit(X_train_tfidf, y_train)
y_pred = ensemble.predict(X_test_tfidf)
print(f"Ensemble Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# บันทึกโมเดลและ TF-IDF vectorizer
joblib.dump(ensemble, 'models/ensemble_model.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
print("✅ บันทึก Ensemble model แล้ว")

# ต่อจาก train_models.py
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

print("กำลังโหลด BERT Pretrained Model...")

# ใช้โมเดล distilbert ที่เล็กกว่า BERT ปกติ แต่แม่นพอกัน
# sentiment-analysis pipeline จะโหลดโมเดลที่ถูก fine-tune มาแล้ว
bert_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,    # ตัดข้อความที่ยาวเกินไปอัตโนมัติ
    max_length=512      # BERT รับข้อความได้สูงสุด 512 tokens
)

# ทดสอบกับ test set (ใช้แค่ 500 ตัวอย่างเพื่อประหยัดเวลา)
print("กำลังทดสอบ BERT...")
sample_texts = X_test[:500].tolist()
sample_labels = y_test[:500].tolist()

# ทำนายทีละ batch เพื่อไม่ให้ RAM เต็ม
bert_preds = []
for i in range(0, len(sample_texts), 32):  # ทีละ 32 ข้อความ
    batch = sample_texts[i:i+32]
    results = bert_pipeline(batch)
    # แปลง POSITIVE=1, NEGATIVE=0
    preds = [1 if r['label'] == 'POSITIVE' else 0 for r in results]
    bert_preds.extend(preds)

bert_acc = accuracy_score(sample_labels, bert_preds)
print(f"BERT Accuracy: {bert_acc:.4f}")

# บันทึก accuracy ไว้แสดงในเว็บ
with open('models/bert_metrics.json', 'w') as f:
    json.dump({'accuracy': bert_acc}, f)

print("✅ BERT พร้อมใช้งานแล้ว (ไม่ต้อง save เพราะ load จาก HuggingFace ได้เลย)")
