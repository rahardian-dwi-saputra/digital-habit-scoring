import pandas as pd
import numpy as np
import onnxmltools

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skl2onnx.common.data_types import FloatTensorType

#load dataset
df = pd.read_csv('dataset/Mental_Health_and_Social_Media_Balance_Dataset.csv')


#cetak informasi dataset
print("5 Data teratas :")
print(df.head())
print("\nRingkasan Statistik data:")
print(df.describe())
print("\nInformasi kolom dataset:")
print(df.info())
print("\nTotal nilai duplikat")
print(df.duplicated().sum())
print("\nData kosong di tiap kolom")
print(df.isnull().sum())


# Data splitting
df_clean = df.drop(columns=['User_ID','Age','Days_Without_Social_Media', 'Exercise_Frequency(week)'])

X = df_clean.drop(columns=["Happiness_Index(1-10)"])
y = df_clean["Happiness_Index(1-10)"]

cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing Fitur (X)
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            num_cols,
        ),
        (
            "cat",
            OneHotEncoder(drop="first", handle_unknown="ignore"),
            cat_cols,
        ),
    ]
)

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

print(f"\nBentuk X_train setelah Preprocessing : {X_train_scaled.shape}")
print(f"Ukuran X_test setelah Preprocessing  : {X_test_scaled.shape}")


# Modelling
xgb_model = XGBRegressor(
    n_estimators=100,  
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
# Training
xgb_model.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False,
)
# Testing
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Evaluasi model
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print('\nHasil Evaluasi Model XGBOOST')
print(f"MAE  : {mae_xgb:.4f}")
print(f"RMSE : {rmse_xgb:.4f}")
print(f"R2   : {r2_xgb:.4f}")

df_result = pd.DataFrame(
    {
        "Nilai Aktual": y_test[:5], 
        "Hasil Prediksi XGBoost": y_pred_xgb[:5]
    }
)
print("\nPerbandingan 5 Data Pertama:")
print(df_result)


# Export Model
num_features = X_train_scaled.shape[1]
initial_types = [("float_input", FloatTensorType([None, num_features]))]

onnx_model = onnxmltools.convert_xgboost(
    xgb_model, 
    initial_types=initial_types
)

onnxmltools.utils.save_model(
    onnx_model, 
    "exports/predict_happiness_model.onnx"
)
print("Model berhasil dieksport ke exports/predict_happiness_model.onnx!")