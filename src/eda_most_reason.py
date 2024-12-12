import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ตั้งค่าฟอนต์สำหรับภาษาไทย
rcParams['font.family'] = 'Tahoma'  # หรือ 'Angsana New', 'TH Sarabun New'

# โหลดข้อมูล
file_path = 'accident_combine_records_cleaned_final.xlsx'
df = pd.read_excel(file_path)

# ทำความสะอาดข้อมูล
df.columns = df.columns.str.strip()  # ลบช่องว่างออกจากชื่อคอลัมน์
df['มูลเหตุสันนิษฐาน'] = df['มูลเหตุสันนิษฐาน'].fillna('ไม่ทราบ')  # เติมค่าที่ขาดหาย

# นับจำนวนการเกิดอุบัติเหตุแยกตาม "มูลเหตุสันนิษฐาน"
cause_counts = df['มูลเหตุสันนิษฐาน'].value_counts()

# เลือก 8 อันดับที่เกิดบ่อยที่สุด
top_8_causes = cause_counts.head(10)
# สร้างกราฟแท่ง
plt.figure(figsize=(10, 6))
top_8_causes.plot(kind='bar', color='skyblue', edgecolor='black')

# ตั้งค่าชื่อและแกนของกราฟ
plt.title('10 Most Accident Reason', fontsize=16)
plt.xlabel('Reason', fontsize=14)
plt.ylabel('Amout', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# แสดงกราฟ
plt.tight_layout()
plt.show()
