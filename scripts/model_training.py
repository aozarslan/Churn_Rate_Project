import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Eğitim verisini yükleme
train_data_path = '../data/processed/train_data.csv'
train_data_df = pd.read_csv(train_data_path)

# Test verisini yükleme
test_data_path = '../data/processed/test_data.csv'
test_data_df = pd.read_csv(test_data_path)

# Zaman serisi tarih indeksini oluşturma
train_data_df['Quarter Start Date'] = pd.to_datetime(train_data_df['Quarter Start Date'])
test_data_df['Quarter Start Date'] = pd.to_datetime(test_data_df['Quarter Start Date'])
train_data_df.set_index('Quarter Start Date', inplace=True)
test_data_df.set_index('Quarter Start Date', inplace=True)

# Her müşteri için ayrı model eğitmek için müşteri ID'lerine göre gruplama
customer_groups = train_data_df.groupby('Müşteri Id')

# Modelleri kaydetmek için dizin oluşturma
model_dir = '../models/'
os.makedirs(model_dir, exist_ok=True)

# Her müşteri için bir model eğitme
for customer_id, group in customer_groups:
    group = group.sort_index()

    # Eğer müşteri verisi 2 tam sezona sahipse (8 çeyrek), Holt-Winters modelini kullan
    if len(group) >= 8:
        try:
            model = ExponentialSmoothing(
                group['Segment Score'],
                trend='add',
                seasonal='add',
                seasonal_periods=4,
                damped_trend=True
            ).fit(damping_trend=0.8)
            model_filename = os.path.join(model_dir, f'model_{customer_id}.pkl')
            joblib.dump(model, model_filename)
            print(f"Model for customer {customer_id} saved to {model_filename}")
        except Exception as e:
            print(f"Error training Holt-Winters model for customer {customer_id}: {e}")
            continue
    else:
        # Tam sezonları olmayan müşteriler için basit Exponential Smoothing modeli kullan
        try:
            model = SimpleExpSmoothing(group['Segment Score']).fit()
            model_filename = os.path.join(model_dir, f'model_{customer_id}.pkl')
            joblib.dump(model, model_filename)
            print(f"Model for customer {customer_id} saved to {model_filename}")
        except Exception as e:
            print(f"Error training Simple Exponential Smoothing model for customer {customer_id}: {e}")
            # Alternatif çözüm: Basit ortalama tahmini
            model = group['Segment Score'].mean()
            model_filename = os.path.join(model_dir, f'model_{customer_id}_fallback.pkl')
            joblib.dump(model, model_filename)
            print(f"Fallback model (mean) for customer {customer_id} saved to {model_filename}")

print("All models trained and saved successfully.")

# Performans değerlendirmesi için test verisinde tahminler yapma
predictions = []
actuals = []

for customer_id in test_data_df['Müşteri Id'].unique():
    model_filename = os.path.join(model_dir, f'model_{customer_id}.pkl')
    fallback_model_filename = os.path.join(model_dir, f'model_{customer_id}_fallback.pkl')
    
    if os.path.exists(model_filename):
        model = joblib.load(model_filename)
    elif os.path.exists(fallback_model_filename):
        model = joblib.load(fallback_model_filename)
    else:
        print(f"No model found for customer {customer_id}")
        continue
    
    test_customer_data = test_data_df[test_data_df['Müşteri Id'] == customer_id]
    actuals.extend(test_customer_data['Segment Score'])
    
    if isinstance(model, float):  # Model basit ortalama ise
        predictions.extend([model] * len(test_customer_data))
    else:
        try:
            predictions.extend(model.forecast(len(test_customer_data)).values)
        except KeyError as e:
            print(f"Error forecasting for customer {customer_id}: {e}")
            predictions.extend([model] * len(test_customer_data))  # Ortalamayı kullan

# Performans metriklerini hesaplama
mae = mean_absolute_error(actuals, predictions)
mse = mean_squared_error(actuals, predictions)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")