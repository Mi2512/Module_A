import os
import json
import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP не установлен. Установите с помощью: pip install shap")

try:
    from sklearn.inspection import permutation_importance
    PERM_IMPORTANCE_AVAILABLE = True
except ImportError:
    PERM_IMPORTANCE_AVAILABLE = False


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.significant_features = []
        
    def preprocess(self, dataset: pd.DataFrame) -> pd.DataFrame:
        df = dataset.copy()
        
        df = self._handle_missing_values(df)
        
        df = self._encode_categorical_variables(df)
        
        df = self._detect_handle_outliers(df)
        
        df = self._feature_engineering(df)
        
        df, significant_attrs = self._identify_significant_attributes(df)
        self.significant_features = significant_attrs
        
        df = self._normalize_features(df)
        
        df = self._add_seasonal_completeness(df)
        
        os.makedirs('data/processed', exist_ok=True)
        df.to_csv(os.path.join('data', 'processed', 'preprocessed_dataset.csv'), index=False)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if col != 'surrounding_objects':
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col != 'surrounding_objects']
        
        for col in categorical_cols:
            le = LabelEncoder()
            non_null_vals = df[col].dropna().astype(str)
            le.fit(non_null_vals)
            
            df[col] = df[col].apply(lambda x: le.transform([str(x)])[0] if pd.notna(x) and str(x) in le.classes_ else -1)
            
            self.label_encoders[col] = le
        
        return df
    
    def _detect_handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != 'point_index':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['hour'] = df['date'].dt.hour
            df['day_of_year'] = df['date'].dt.dayofyear
            df['season'] = df['month'].map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})
        
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['distance_from_equator'] = df['latitude'].abs()
            df['distance_from_prime_meridian'] = df['longitude'].abs()
        
        if 'altitude' in df.columns:
            if 'track_id' in df.columns:
                df['altitude_variation'] = df.groupby('track_id')['altitude'].transform(lambda x: x.max() - x.min())
                df['altitude_change_rate'] = df.groupby('track_id')['altitude'].pct_change().fillna(0)
            else:
                df['altitude_variation'] = df['altitude'].max() - df['altitude'].min()
                df['altitude_change_rate'] = df['altitude'].pct_change().fillna(0)
        
        if 'surrounding_objects' in df.columns:
            df['num_nearby_objects'] = df['surrounding_objects'].apply(
                lambda x: len(json.loads(x)) if pd.notna(x) and isinstance(x, str) else 0
            )
        
        return df
    
    def _identify_significant_attributes(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        feature_cols = [col for col in df.columns if col not in ['track_id', 'date', 'surrounding_objects']]
        
        X = df[feature_cols].select_dtypes(include=[np.number])
        
        non_constant_cols = X.columns[X.std() != 0]
        X = X[non_constant_cols]
        
        if X.empty:
            print("Предупреждение: Не найдено признаков с изменением для анализа значимости.")
            return df, []
        
        y = df['temperature'].fillna(df['temperature'].mean())
        
        significant_features = []
        
        correlations = {}
        for col in X.columns:
            if len(X[col].dropna()) > 1:
                try:
                    corr, _ = stats.pearsonr(X[col].fillna(X[col].mean()), y)
                    correlations[col] = abs(corr)
                except ValueError:
                    correlations[col] = 0
        
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        high_corr_features = [item[0] for item in sorted_corr if item[1] > 0.1]
        
        try:
            selector = SelectKBest(score_func=f_regression, k=min(10, len(X.columns)))
            X_selected = selector.fit_transform(X.fillna(X.mean()), y)
            anova_features = [X.columns[i] for i in selector.get_support(indices=True)]
        except ValueError:
            anova_features = []
        
        perm_importance_features = []
        if PERM_IMPORTANCE_AVAILABLE:
            try:
                from sklearn.ensemble import RandomForestRegressor
                
                rf_model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=3)
                rf_model.fit(X.fillna(X.mean()), y)
                
                perm_importance = permutation_importance(
                    rf_model, 
                    X.fillna(X.mean()), 
                    y, 
                    n_repeats=5, 
                    random_state=42,
                    n_jobs=-1
                )
                perm_indices = np.argsort(perm_importance.importances_mean)[::-1][:10]
                perm_importance_features = [X.columns[i] for i in perm_indices if perm_importance.importances_mean[i] > 0.001]
            except Exception as e:
                print(f"Вычисление permutation importance не удалось: {e}")
        
        shap_features = []
        if SHAP_AVAILABLE:
            try:
                from sklearn.ensemble import RandomForestRegressor
                
                rf_model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=3)
                rf_model.fit(X.fillna(X.mean()), y)
                
                sample_size = min(100, len(X))
                X_sample = X.fillna(X.mean()).iloc[:sample_size]
                
                explainer = shap.TreeExplainer(rf_model, feature_perturbation="tree_path_dependent")
                shap_values = explainer.shap_values(X_sample)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                mean_shap_values = np.mean(np.abs(shap_values), axis=0)
                shap_indices = np.argsort(mean_shap_values)[::-1][:10]
                shap_features = [X.columns[i] for i in shap_indices if mean_shap_values[i] > 0.001]
            except Exception as e:
                print(f"Вычисление SHAP не удалось: {e}")
        
        all_significant = set(high_corr_features + anova_features + perm_importance_features + shap_features)
        significant_features = list(all_significant)
        
        print(f"Выделено {len(significant_features)} значимых признаков: {significant_features}")
        
        return df, significant_features
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        return df
    
    def _add_seasonal_completeness(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'season' in df.columns:
            season_counts = df['season'].value_counts()
            print(f"Распределение по сезонам: {season_counts.to_dict()}")
            
            total_samples = len(df)
            min_season_samples = int(0.2 * total_samples)
            
            for season in range(4):
                current_count = season_counts.get(season, 0)
                
                if current_count < min_season_samples:
                    samples_needed = min_season_samples - current_count
                    synthetic_data = self._generate_synthetic_seasonal_data(df, season, samples_needed)
                    
                    df = pd.concat([df, synthetic_data], ignore_index=True)
        
        return df
    
    def _generate_synthetic_seasonal_data(self, df: pd.DataFrame, season: int, count: int) -> pd.DataFrame:
        if len(df) == 0:
            print("Невозможно генерировать синтетические данные из пустого датасета")
            return pd.DataFrame()
        
        if len(df) >= count:
            synthetic_df = df.iloc[:count].copy()
        else:
            reps = count // len(df) + 1
            synthetic_df = pd.concat([df] * reps, ignore_index=True)
            synthetic_df = synthetic_df.iloc[:count]
        
        synthetic_df['season'] = season
        
        month_mapping = {0: [12, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8], 3: [9, 10, 11]}
        months = month_mapping[season]
        synthetic_df['month'] = np.random.choice(months, size=len(synthetic_df))
        
        temp_adjustments = {0: (-10, 5), 1: (5, 15), 2: (15, 30), 3: (5, 15)}
        min_temp, max_temp = temp_adjustments[season]
        synthetic_df['temperature'] = np.random.uniform(min_temp, max_temp, size=len(synthetic_df))
        
        numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['season', 'month', 'temperature']:
                std_val = synthetic_df[col].std()
                if not pd.isna(std_val) and std_val != 0:
                    noise = np.random.normal(0, std_val * 0.1, size=len(synthetic_df))
                    synthetic_df[col] += noise
        
        return synthetic_df