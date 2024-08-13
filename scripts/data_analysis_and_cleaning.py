import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Temizlenmiş veriyi yükleme
cleaned_data_path = '../data/processed/cleaned_data.csv'
cleaned_data_df = pd.read_csv(cleaned_data_path)

# Eksik veri olup olmadığını kontrol etme
missing_values = cleaned_data_df.isnull().sum()
print("Missing Values:")
print(missing_values)

# Sadece sayısal sütunları seçme
numeric_cols = cleaned_data_df.select_dtypes(include=[np.number])

# Aykırı değerleri belirlemek için IQR yöntemini kullanma
Q1 = numeric_cols.quantile(0.25)
Q3 = numeric_cols.quantile(0.75)
IQR = Q3 - Q1

outliers = ((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).sum()
print("Outliers:")
print(outliers)

# Aykırı değerleri temizleme
cleaned_data_df = cleaned_data_df[~((numeric_cols < (Q1 - 1.5 * IQR)) |(numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]

# Müşteri başına Segment Score varyansını hesaplama
variance_df = cleaned_data_df.groupby('Müşteri Id')['Segment Score'].var().reset_index()
variance_df.columns = ['Müşteri Id', 'Segment Score Variance']

# Düşük varyanslı müşteri segmentlerini belirleme
low_variance_threshold = 1e-5  # Bu eşik değeri gerektiğinde ayarlanabilir
low_variance_ids = variance_df[variance_df['Segment Score Variance'] < low_variance_threshold]['Müşteri Id']
print("Low variance Segment Scores:")
print(low_variance_ids)

# Düşük varyanslı segment skorlarını temizleme
cleaned_data_df = cleaned_data_df[~cleaned_data_df['Müşteri Id'].isin(low_variance_ids)]

# Veriyi eğitim ve test seti olarak bölme
train_df, test_df = train_test_split(cleaned_data_df, test_size=0.2, random_state=42)

# Eğitim ve test verilerini kaydetme
train_df.to_csv('../data/processed/train_data.csv', index=False)
test_df.to_csv('../data/processed/test_data.csv', index=False)
print("Train and test data saved to ../data/processed/")