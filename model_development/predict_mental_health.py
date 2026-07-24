import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import onnxmltools

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skl2onnx.common.data_types import FloatTensorType

#load dataset
df = pd.read_csv('dataset/Social_media_impact_on_life.csv')


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


#Visualisasi Data
# Boxplot untuk deteksi outlier
cols = [
    "Avg_Daily_Usage_Hours",
    "Sleep_Hours_Per_Night",
]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

for i, col in enumerate(cols):
    sns.boxplot(
        data=df,
        y=col,
        ax=axes[i],
        color="skyblue",
        width=0.4,
        flierprops=dict(
            marker='o', 
            markerfacecolor='red', 
            markersize=6
        ),
    )
    axes[i].set_title(
        f"Boxplot: {col}", 
        fontsize=12, 
        fontweight="bold"
    )
    axes[i].set_ylabel(
        "Nilai", 
        fontsize=10
    )

plt.tight_layout()
plt.show()

# Grafik distribusi overall impact
counts = df["Overall_Impact"].value_counts()
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    counts.index.astype(str),
    counts.values,
    color="skyblue",
    edgecolor="navy",
    alpha=0.8,
)

ax.bar_label(
    bars, 
    padding=3, 
    fontsize=10, 
    fontweight="bold"
)
ax.set_title(
    "Distribusi Jumlah per Overall Impact", 
    fontsize=14, 
    pad=15
)
ax.set_xlabel(
    "Overall Impact", 
    fontsize=11
)
ax.set_ylabel(
    "Jumlah Data", 
    fontsize=11
)
ax.grid(
    axis="y", 
    linestyle="--", 
    alpha=0.5
)

ax.set_ylim(0, max(counts.values) * 1.15)

plt.tight_layout()
plt.show()


# Headmat untu mengecek korelasi fitur
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8), dpi=75)

ax = sns.heatmap(
    corr_matrix, 
    annot=True, 
    cmap='coolwarm', 
    fmt='.2f', 
    linewidths=0.5,
    cbar_kws={'shrink': 0.8},
    annot_kws={'size': 9} 
)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

plt.xticks(rotation=45, ha='right')
plt.tight_layout(pad=2.0) 

plt.title(
    'Heatmap Korelasi Variabel Numerik', 
    fontweight='bold', 
    fontsize=12
)
plt.show()


# Data splitting
df_clean = df.drop(columns=['Student_ID', 'Age', 'Academic_Level','Country', 'Affects_Academic_Performance', 'Overall_Impact'])

X = df_clean.drop(columns=["Mental_Health_Score"])
y = df_clean["Mental_Health_Score"]

categorical_ohe = ["Gender", "Most_Used_Platform"]
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "ohe", 
            OneHotEncoder(
                drop="first", 
                handle_unknown="ignore"
            ), 
            categorical_ohe
        ),
        (
            "scaler", 
            StandardScaler(), 
            numeric_features
        ),
    ],
    remainder="passthrough",
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

print(f"\nBentuk X_train setelah Preprocessing : {X_train_preprocessed.shape}")
print(f"Ukuran X_test setelah Preprocessing  : {X_test_preprocessed.shape}")

# Eksport konfigurasi fitur
scaler_obj = preprocessor.named_transformers_["scaler"]
num_info = {}
for col_name, mean_val, std_val in zip(
    numeric_features, 
    scaler_obj.mean_, 
    scaler_obj.scale_
):
    num_info[col_name] = {
        "mean": float(mean_val), 
        "std": float(std_val)
    }

ohe_obj = preprocessor.named_transformers_["ohe"]
cat_info = {}
for col_name, categories in zip(
    categorical_ohe, 
    ohe_obj.categories_
):
    
    dropped_cat = categories[0]
    kept_categories = categories[1:].tolist()

    cat_info[col_name] = {
        "dropped_category": str(dropped_cat),
        "encoded_categories": kept_categories,
    }

feature_names_out = preprocessor.get_feature_names_out().tolist()

pipeline_config = {
    "feature_names_out": feature_names_out,
    "numeric_features": num_info,
    "categorical_features": cat_info,
}

with open("exports/predict_mental_health_config.json", "w") as f:
    json.dump(pipeline_config, f, indent=2)

print("\n[+] Berhasil menyimpan konfigurasi ke 'exports/predict_mental_health_config.json'")
print(f"[+] Total fitur hasil preprocessing: {len(feature_names_out)}")


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
    X_train_preprocessed,
    y_train,
    eval_set=[(X_test_preprocessed, y_test)],
    verbose=False,
)
# Testing
y_pred_xgb = xgb_model.predict(X_test_preprocessed)

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
num_features = X_train_preprocessed.shape[1]
initial_types = [("float_input", FloatTensorType([None, num_features]))]

onnx_model = onnxmltools.convert_xgboost(
    xgb_model, 
    initial_types=initial_types
)

onnxmltools.utils.save_model(
    onnx_model, 
    "exports/predict_mental_health_model.onnx"
)
print("Model berhasil dieksport ke exports/predict_mental_health_model.onnx!")