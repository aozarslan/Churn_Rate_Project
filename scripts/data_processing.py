import pandas as pd# Ham veri dosyasının yoludata_path = '../data/raw/5-10_churn.xlsx'# Veriyi okumadata_df = pd.read_excel(data_path)# Sütun adlarını yazdırarak kontrol edelimprint("Sütun adları:", data_df.columns)# Doğru sütun adını kullanarak tarihi datetime formatına çevirelimdata_df['Quarter Start Date'] = pd.to_datetime(data_df['Quarter Start Date'])# Veriyi kaydetmeprocessed_data_path = '../data/processed/processed_data.csv'data_df.to_csv(processed_data_path, index=False)print(f"Processed data saved to {processed_data_path}")