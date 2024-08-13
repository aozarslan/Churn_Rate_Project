import pandas as pd

# Temizlenmiş veriyi yükleme
cleaned_data_path = '../data/processed/cleaned_data.csv'
cleaned_data_df = pd.read_csv(cleaned_data_path)

# Temel istatistiksel analizler
print("Descriptive Statistics:\n", cleaned_data_df.describe())
print("\nData Information:\n")
print(cleaned_data_df.info())

# Verinin ilk birkaç satırını görüntüleme
print("\nFirst few rows of the data:\n", cleaned_data_df.head())