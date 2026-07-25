import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import model_pipeline as pipe

from xgboost import XGBRegressor

# Data loading
df = pd.read_csv('data/raw/Mental_Health_and_Social_Media_Balance_Dataset.csv')
pipe.print_dataset_info(df)


#Visualisasi Data
# Boxplot untuk deteksi outlier
columns = [
    'Daily_Screen_Time(hrs)',
    'Days_Without_Social_Media',
    'Exercise_Frequency(week)',
]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))

for idx, col in enumerate(columns):
    sns.boxplot(
        data=df,
        x=col,
        ax=axes[idx],
        color='skyblue',
        flierprops=dict(
            marker='o', 
            markerfacecolor='red', 
            markersize=6
        ),
    )
    axes[idx].set_title(
        f'Boxplot\n{col}', 
        fontsize=11, 
        fontweight='bold'
    )
    axes[idx].set_xlabel(
        col, 
        fontsize=10
    )

plt.tight_layout()
plt.show()

# Grafik distribusi target
happiness_counts = (
    df['Happiness_Index(1-10)'].value_counts().sort_index().reset_index()
)
happiness_counts.columns = ['Happiness_Index', 'Jumlah']

plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=happiness_counts,
    x='Happiness_Index',
    y='Jumlah',
    hue='Happiness_Index',
    legend=False,
    palette='Blues_d',
)

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(
            f'{int(height)}',
            (p.get_x() + p.get_width() / 2.0, height),
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            xytext=(0, 3),
            textcoords='offset points',
        )

plt.title(
    'Distribusi Frekuensi Happiness Index (1-10)', 
    fontsize=14, 
    fontweight='bold'
)
plt.xlabel('Happiness Index', fontsize=12)
plt.ylabel('Jumlah (Frekuensi)', fontsize=12)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()

# Heatmap untuk mengecek korelasi fitur
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(7, 5))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    annot_kws={"size": 10},  
    cbar_kws={"shrink": 0.9},
)

plt.title(
    "Heatmap Korelasi Variabel Numerik", 
    fontweight="bold", 
    fontsize=11
)

plt.tight_layout()
plt.show()

df_clean = df.drop(columns=[
    'User_ID',
    'Age',
    'Days_Without_Social_Media',
    'Exercise_Frequency(week)'
])
df_clean = pipe.standardize_dataframe(df_clean)

# Data splitting
X_train, X_test, y_train, y_test, preprocessor = pipe.split_and_preprocess_data(
    df=df_clean, 
    target_column="happiness_index(1-10)",
    export_json_path="exports/predict_happiness_config.json",
    export_csv_path="data/processed/happiness_processed.csv",
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
    output_path="exports/predict_happiness_model.onnx"
)