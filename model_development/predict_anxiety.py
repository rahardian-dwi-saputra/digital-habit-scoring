import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import model_pipeline as pipe

from xgboost import XGBRegressor

# Data loading
df = pd.read_csv('data/raw/Teen_Mental_Health_Dataset.csv')
pipe.print_dataset_info(df)


#Visualisasi Data
# Boxplot untuk deteksi outlier
cols = [
    "daily_social_media_hours",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "stress_level",
    "anxiety_level",
]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 7), dpi=80)

axes = axes.flatten()

for i, col in enumerate(cols):
    sns.boxplot(
        y=df[col], 
        ax=axes[i], 
        color="skyblue", 
        width=0.4
    )
    axes[i].set_title(
        col, 
        fontsize=10, 
        fontweight="bold"
    )
    axes[i].set_ylabel("")
    axes[i].grid(
        axis="y", 
        linestyle="--", 
        alpha=0.5
    )

plt.tight_layout(pad=2.0)
plt.show()

# Grafik distribusi Depression Risk
counts = df["depression_risk"].value_counts()

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
    "Distribusi Jumlah per Depression Risk", 
    fontsize=14, 
    pad=15
)
ax.set_xlabel(
    "Depression Risk", 
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
ax.set_ylim(
    0, 
    max(counts.values) * 1.15
)

plt.tight_layout()
plt.show()

# Heatmap untuk mengecek korelasi fitur
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 7))
sns.heatmap(
    corr_matrix, 
    annot=True, 
    cmap='coolwarm', 
    fmt='.2f', 
    linewidths=0.5,
    annot_kws={"size": 9}
)

plt.xticks(
    rotation=45, 
    ha='right', 
    fontsize=9
)
plt.yticks(
    rotation=0, 
    fontsize=9
)

plt.title(
    'Heatmap Korelasi Variabel Numerik', 
    fontweight='bold', 
    pad=15
)

plt.tight_layout()
plt.show()

df_clean = df.drop(columns=[
    'age',
    'screen_time_before_sleep',
    'academic_performance',
    'depression_risk',
    'social_interaction_level'
])
df_clean = pipe.standardize_dataframe(df_clean)


# Data splitting
X_train, X_test, y_train, y_test, preprocessor = pipe.split_and_preprocess_data(
    df=df_clean, 
    target_column="anxiety_level",
    export_json_path="exports/predict_anxiety_config.json",
    export_csv_path="data/processed/anxiety_processed.csv",
)

print(f"\nBentuk X_train setelah Preprocessing : {X_train.shape}")
print(f"Ukuran X_test setelah Preprocessing  : {X_test.shape}")


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
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
# Testing
y_pred_xgb = xgb_model.predict(X_test)

# Evaluasi model
print("\n")
metrics = pipe.evaluate_regression_model(
    y_true=y_test, 
    y_pred=y_pred_xgb, 
    model_name="XGBoost"
)


# Eksport model
pipe.export_model_to_onnx(
    model=xgb_model,
    num_features=X_train.shape[1],
    output_path="exports/predict_anxiety_model.onnx"
)