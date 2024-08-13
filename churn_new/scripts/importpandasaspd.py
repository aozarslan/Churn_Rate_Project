import pandas as pd # type: ignore
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Veriyi okuma
df = pd.read_csv('/Users/cornerback30/Desktop/churn_new/data/churn_data_with_quarterly_totals_and_scores-2.csv')

# Tarih sütununu datetime formatına çevirme
df['Quarter Start Date'] = pd.to_datetime(df['Quarter Start Date'])

# Tarih sütununa göre sıralama
df = df.sort_values(by='Quarter Start Date')


# Her çeyrek döneme ağırlık atama (çeyrek bazlı)

#her çeyrek döneme sıralı bir numara atanır.
df['Quarter Rank'] = df['Quarter Start Date'].rank(method='dense').astype(int)
#Bu, veri setindeki toplam çeyrek dönemi sayısını temsil eder.
quarter_max_rank = df['Quarter Rank'].max()
df['Quarter Weight'] = (df['Quarter Rank']) / (quarter_max_rank)



# Model için veriyi hazırlama
X = df.drop(['Segment Score', 'Müşteri Id', 'Quarter Start Date'], axis=1)
y = df['Segment Score']

# Veriyi eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hiperparametreler için grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Grid Search
grid_search = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror'),
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)


# En iyi hiperparametreler
best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

# En iyi model
best_model = grid_search.best_estimator_


# Eğitim seti üzerindeki tahminler
y_train_pred = best_model.predict(X_train)

# Test seti üzerindeki tahminler
y_test_pred = best_model.predict(X_test)

# Eğitim hatalarını hesaplama
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Test hatalarını hesaplama
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Sonuçları yazdırma
print(f'Çapraz Doğrulama Mean Squared Error (MSE): {train_mse}')
print(f'Çapraz Doğrulama R^2 Score: {train_r2}')
print(f'Eğitim Hatası - Mean Squared Error (MSE): {train_mse}')
print(f'Eğitim Hatası - R^2 Score: {train_r2}')
print(f'Test Hatası - Mean Squared Error (MSE): {test_mse}')
print(f'Test Hatası - R^2 Score: {test_r2}')


# Gelecek çeyreğin verilerini hazırlama
last_quarter_data = df[df['Quarter Start Date'] == df['Quarter Start Date'].max()].copy()
next_quarter_features = last_quarter_data.drop(['Segment Score', 'Quarter Start Date', 'Müşteri Id'], axis=1)

# Gelecek çeyrek için tahmin yapma
next_quarter_predictions = best_model.predict(next_quarter_features)


# Sonuçları Excel dosyasına kaydetme
last_quarter_data['Predicted Segment Score'] = next_quarter_predictions
last_quarter_data['Quarter Start Date'] = last_quarter_data['Quarter Start Date'] + pd.DateOffset(months=3)
output_df = last_quarter_data[['Müşteri Id', 'Quarter Start Date', 'Predicted Segment Score']]
output_path = 'agirlikli_son.xlsx'
output_df.to_excel(output_path, index=False)
print(f'Results saved to {output_path}')

###
# DataFrame'i Excel dosyasına kaydetme
data = 'veri_seti.xlsx'
df.to_excel(data, index=False)

print(f'Results saved to {data}')


####
# Son çeyreğin verilerini hazırlama
last_quarter_data = df[df['Quarter Start Date'] == df['Quarter Start Date'].max()].copy()
last_quarter_features = last_quarter_data.drop(['Segment Score', 'Quarter Start Date', 'Müşteri Id'], axis=1)

# Son çeyrek için tahmin yapma
last_quarter_predictions = best_model.predict(last_quarter_features)

# Tahmin sonuçlarını orijinal veriye ekleme
last_quarter_data['Predicted Segment Score'] = last_quarter_predictions

# Sonuçları Excel dosyasına kaydetme
output_df = last_quarter_data[['Müşteri Id', 'Quarter Start Date', 'Segment Score', 'Predicted Segment Score']]
son_ceyrek = 'son_ceyrek_tahmin.xlsx'
output_df.to_excel(son_ceyrek, index=False)

print(f'Results saved to {son_ceyrek}')
