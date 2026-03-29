# utils/preprocessing.py
import pandas as pd
import re  # ใช้สำหรับ regex (ค้นหา/แทนที่ตัวอักษร)

def clean_text(text):
    """
    ทำความสะอาดข้อความ
    - ลบ HTML tags เช่น <br>, <p>
    - ลบตัวอักษรพิเศษ เช่น @#$%
    - แปลงเป็นตัวพิมพ์เล็กทั้งหมด
    """
    text = re.sub(r'<.*?>', '', text)        # ลบ HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text) # เก็บแค่ตัวอักษรและช่องว่าง
    text = text.lower().strip()              # เปลี่ยนเป็นตัวพิมพ์เล็ก
    return text

def load_imdb(filepath):
    """โหลดและเตรียม IMDB dataset"""
    df = pd.read_csv(filepath)
    # แปลง label: positive=1, negative=0
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    df['clean_text'] = df['review'].apply(clean_text)
    return df[['clean_text', 'label']]

def load_amazon(filepath):
    """โหลดและเตรียม Amazon dataset"""
    df = pd.read_csv(filepath).dropna(subset=['Text', 'Score'])
    df = df.sample(10000, random_state=42)  # สุ่มแค่ 10,000 แถว
    # คะแนน 4-5 = positive(1), 1-2 = negative(0), ตัด 3 ออก
    df = df[df['Score'] != 3]
    df['label'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)
    df['clean_text'] = df['Text'].apply(clean_text)
    return df[['clean_text', 'label']]
