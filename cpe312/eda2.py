import pandas as pd
import matplotlib.pyplot as plt

# อ่านข้อมูลจากไฟล์c
df = pd.read_excel('/cpe312/accident2019.xlsx')  # ตรวจสอบว่าไฟล์อยู่ในที่เดียวกับโค้ด

# ตรวจสอบให้แน่ใจว่าคอลัมน์ 'วันที่เกิดเหตุ' เป็นประเภท datetime
df['วันที่เกิดเหตุ'] = pd.to_datetime(df['วันที่เกิดเหตุ'], errors='coerce')
df = df.dropna(subset=['วันที่เกิดเหตุ'])  # ลบข้อมูลที่ไม่มีวันที่

# สร้างคอลัมน์ 'ปี' และ 'เดือน' เพื่อใช้สำหรับการจัดกลุ่ม
df['ปี'] = df['วันที่เกิดเหตุ'].dt.year
df['เดือน'] = df['วันที่เกิดเหตุ'].dt.month

# นับจำนวนอุบัติเหตุในแต่ละ "มูลเหตุสันนิษฐาน" แยกตามปีและเดือน
top_causes = df.groupby(['ปี', 'เดือน', 'มูลเหตุสันนิษฐาน']).size().reset_index(name='จำนวนอุบัติเหตุ')

# เลือกมูลเหตุสันนิษฐานที่เกิดขึ้นมากที่สุด 5 อันดับในแต่ละเดือน-ปี
top_5_causes = top_causes.groupby(['ปี', 'เดือน']).apply(lambda x: x.nlargest(5, 'จำนวนอุบัติเหตุ')).reset_index(drop=True)

# แปลงข้อมูลเดือนและปีเป็น Datetime เพื่อใช้เป็นแกน X
top_5_causes['เดือน_ปี'] = pd.to_datetime(top_5_causes['ปี'].astype(str) + '-' + top_5_causes['เดือน'].astype(str))

# สร้างกราฟแสดงมูลเหตุสันนิษฐานที่พบบ่อย 5 อันดับในแต่ละเดือน-ปี
plt.figure(figsize=(14, 7))
for cause in top_5_causes['มูลเหตุสันนิษฐาน'].unique():
    cause_data = top_5_causes[top_5_causes['มูลเหตุสันนิษฐาน'] == cause]
    print(cause_data)
    plt.plot(cause_data['เดือน_ปี'], cause_data['จำนวนอุบัติเหตุ'], marker='o', linestyle='-', label=cause)

# ตั้งค่าชื่อกราฟและป้ายแกน
plt.title('5 มูลเหตุสันนิษฐานที่พบบ่อยที่สุดในแต่ละเดือน-ปี')
plt.xlabel('เดือน-ปี')
plt.ylabel('จำนวนอุบัติเหตุ')
plt.legend(title='มูลเหตุสันนิษฐาน')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()
