import pandas as pd
import joblib
import os

# Temizlenmiş veri dosyasının yolu
data_path = '../data/processed/cleaned_data.csv'

# Veriyi okuma
data_df = pd.read_csv(data_path)

# Sütun isimlerini temizleme
data_df.columns = data_df.columns.str.strip()

# Zaman serisi tarih indeksini oluşturma
data_df['Quarter Start Date'] = pd.to_datetime(data_df['Quarter Start Date'])
data_df.set_index('Quarter Start Date', inplace=True)

# Modellerin bulunduğu dizin
model_dir = '../models/'

# Tahmin sonuçlarını saklamak için bir DataFrame oluşturma
predictions = []

# Her müşteri için tahmin yapma
for customer_id in data_df['Müşteri Id'].unique():
    model_filename = os.path.join(model_dir, f'model_{customer_id}.pkl')
    fallback_model_filename = os.path.join(model_dir, f'model_{customer_id}_fallback.pkl')
    
    if os.path.exists(model_filename):
        model = joblib.load(model_filename)
    elif os.path.exists(fallback_model_filename):
        model = joblib.load(fallback_model_filename)
    else:
        print(f"No model found for customer {customer_id}")
        continue
    
    # Bir sonraki üç aylık periyot için tahmin yapma
    if isinstance(model, float):  # Model basit ortalama ise
        next_quarter_prediction = model
    else:
        try:
            next_quarter_prediction = model.forecast(1).iloc[0]
        except KeyError as e:
            print(f"Error forecasting for customer {customer_id}: {e}")
            next_quarter_prediction = model  # Ortalamayı kullan
        
    predictions.append({
        'Müşteri Id': customer_id,
        'Quarter Start Date': pd.Timestamp.now().normalize(),  # Tahmin tarihi
        'cwg': data_df.loc[data_df['Müşteri Id'] == customer_id, 'cwg'].values[-1],  # Son çeyrek CWG
        'Segment Score': data_df.loc[data_df['Müşteri Id'] == customer_id, 'Segment Score'].values[-1],  # Son çeyrek Segment Puanı
        'Predict_Segment_Score': next_quarter_prediction
    })
    
    print(f"Prediction for customer {customer_id}: {next_quarter_prediction}")

# Tahmin sonuçlarını bir DataFrame'e dönüştürme
predictions_df = pd.DataFrame(predictions)

# Tahmin sonuçlarını CSV dosyasına kaydetme
predictions_df.to_csv('../results/next_quarter_predictions.csv', index=False)

print("Predictions saved to ../results/next_quarter_predictions.csv")