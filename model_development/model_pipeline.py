import pandas as pd
import json
import os
import numpy as np
import onnxmltools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skl2onnx.common.data_types import FloatTensorType

def print_dataset_info(df: pd.DataFrame) -> None:

    print("=" * 50)
    print("INFORMASI DATASET")
    print("=" * 50)
    
    print("\n[1] 5 Data Teratas:")
    print(df.head())
    
    print("\n[2] Ringkasan Statistik Data:")
    print(df.describe(include='all'))
    
    print("\n[3] Informasi Kolom Dataset:")
    df.info()
    
    print("\n[4] Total Nilai Duplikat:")
    print(f"{df.duplicated().sum()} baris")
    
    print("\n[5] Data Kosong di Tiap Kolom:")
    print(df.isnull().sum())
    
    print("\n" + "=" * 50)


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    
    df_clean = df.copy()
    df_clean.columns = (
        df_clean.columns
        .str.lower()
        .str.strip()
        .str.replace(' ', '_')
    )

    # Menyeragamkan nama kolom durasi waktu tidur
    if 'sleep_hours' in df_clean.columns:
        df_clean = df_clean.rename(columns={'sleep_hours': 'sleep_hours_per_night'})

    # Menyeragamkan nama kolom sosial media
    if 'platform_usage' in df_clean.columns:
        df_clean = df_clean.rename(columns={'platform_usage': 'social_media_platform'})
    if 'most_used_platform' in df_clean.columns:
        df_clean = df_clean.rename(columns={'most_used_platform': 'social_media_platform'})

    # Menyeragamkan nama kolom waktu penggunaan sosial media
    if 'avg_daily_usage_hours' in df_clean.columns:
        df_clean = df_clean.rename(columns={'avg_daily_usage_hours': 'daily_social_media_hours'})
    if 'daily_screen_time(hrs)' in df_clean.columns:
        df_clean = df_clean.rename(columns={'daily_screen_time(hrs)': 'daily_social_media_hours'})

    # Menyeragamkan nama kolom stress level
    if 'stress_level' in df_clean.columns:
        df_clean = df_clean.rename(columns={'stress_level': 'stress_level(1-10)'})

    if 'gender' in df_clean.columns:
        df_clean['gender'] = df_clean['gender'].astype(str).str.lower().str.strip()

    if 'social_media_platform' in df_clean.columns:
        df_clean['social_media_platform'] = df_clean['social_media_platform'].replace({
            'Twitter': 'X (Twitter)'
        })

    return df_clean


def split_and_preprocess_data(
    df: pd.DataFrame, 
    target_column: str, 
    export_json_path: str,
    export_csv_path: str,
    test_size: float = 0.2, 
    random_state: int = 42
):
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ]
    )

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Ekstraksi Konfigurasi Preprocessor untuk Node.js
    scaler_obj = preprocessor.named_transformers_["num"]
    num_info = {}
    for col_name, mean_val, std_val in zip(
        num_cols, 
        scaler_obj.mean_, 
        scaler_obj.scale_
    ):
        num_info[col_name] = {
            "mean": float(mean_val), 
            "std": float(std_val)
        }

    ohe_obj = preprocessor.named_transformers_["cat"]
    cat_info = {}
    for col_name, categories in zip(cat_cols, ohe_obj.categories_):
        dropped_cat = categories[0]
        kept_categories = categories[1:].tolist()
        cat_info[col_name] = {
            "dropped_category": str(dropped_cat),
            "encoded_categories": kept_categories,
        }

    feature_names = preprocessor.get_feature_names_out().tolist()

    pipeline_config = {
        "feature_names_out": feature_names,
        "numeric_features": num_info,
        "categorical_features": cat_info,
    }

    if export_json_path:
        output_dir = os.path.dirname(export_json_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(export_json_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_config, f, indent=2)
            
        print(f"\n[+] Konfigurasi preprocessor berhasil diekspor ke: {export_json_path}")

    # Ekspor Seluruh Hasil Preprocessing Dataset ke File CSV
    if export_csv_path:
        X_all_scaled = preprocessor.transform(X)
        
        df_processed = pd.DataFrame(
            X_all_scaled.toarray() if hasattr(X_all_scaled, "toarray") else X_all_scaled, 
            columns=feature_names,
            index=df.index
        )
        
        df_processed[target_column] = y.values
        
        csv_dir = os.path.dirname(export_csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        
        df_processed.to_csv(export_csv_path, index=False)
        print(f"[+] Dataset hasil preprocessing berhasil diekspor ke: {export_csv_path}")


    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor


def evaluate_regression_model(
    y_true, 
    y_pred, 
    model_name: str, 
    n_samples: int = 5
) -> dict:
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("=" * 45)
    print(f"Hasil Evaluasi Model: {model_name}")
    print("=" * 45)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    df_result = pd.DataFrame({
        "Nilai Aktual": np.array(y_true)[:n_samples],
        "Hasil Prediksi": np.array(y_pred)[:n_samples]
    })
    
    print(f"\nPerbandingan {n_samples} Data Pertama:")
    print(df_result)
    print("=" * 45 + "\n")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


def export_model_to_onnx(
    model, 
    num_features: int, 
    output_path: str
) -> None:
    
    initial_types = [("float_input", FloatTensorType([None, num_features]))]

    onnx_model = onnxmltools.convert_xgboost(
        model, 
        initial_types=initial_types
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    onnxmltools.utils.save_model(onnx_model, output_path)
    print(f"\nModel berhasil dieksport ke: {output_path}")