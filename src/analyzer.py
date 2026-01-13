import os
import json
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, shapiro, anderson
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class Analyzer:    
    def __init__(self):
        self.distribution_analysis = {}
        
    def analyze(self, dataset: pd.DataFrame) -> Dict:
        analysis_results = {
            "structure_description": self._describe_structure(dataset),
            "distribution_analysis": self._analyze_distributions(dataset),
            "correlation_analysis": self._analyze_correlations(dataset),
            "multimodal_analysis": self._analyze_multimodal_distributions(dataset)
        }
        
        self._generate_visualizations(dataset, analysis_results)
        
        os.makedirs('reports', exist_ok=True)
        with open(os.path.join('reports', 'analysis_results.json'), 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        return analysis_results
    
    def _describe_structure(self, dataset: pd.DataFrame) -> Dict:
        structure = {
            "shape": dataset.shape,
            "columns": list(dataset.columns),
            "data_types": {col: str(dtype) for col, dtype in dataset.dtypes.items()},
            "missing_values": {col: int(dataset[col].isnull().sum()) for col in dataset.columns},
            "memory_usage": dataset.memory_usage(deep=True).sum(),
            "attribute_descriptions": self._get_attribute_descriptions(dataset)
        }
        
        return structure
    
    def _get_attribute_descriptions(self, dataset: pd.DataFrame) -> Dict:
        descriptions = {}
        
        for col in dataset.columns:
            col_info = {
                "name": col,
                "type": str(dataset[col].dtype),
                "description": self._get_attribute_meaning(col),
                "units": self._get_attribute_units(col),
                "valid_range": self._get_valid_range(dataset[col]),
                "purpose": self._get_attribute_purpose(col)
            }
            descriptions[col] = col_info
        
        return descriptions
    
    def _get_attribute_meaning(self, col_name: str) -> str:
        meanings = {
            "track_id": "Уникальный идентификатор для каждого маршрута",
            "point_index": "Последовательный индекс точки в маршруте",
            "date": "Временная метка когда была записана GPS точка",
            "region": "Географический регион где был записан маршрут",
            "latitude": "Широта координат в системе WGS84",
            "longitude": "Долгота координат в системе WGS84",
            "altitude": "Высота над уровнем моря в метрах",
            "frequency_steps": "Оценочная частота шагов в минуту",
            "temperature": "Температура воздуха в месте и время в Цельсиях",
            "terrain_type": "Классификация преобладающего типа местности в области",
            "surrounding_objects": "JSON строка с ближайшими географическими объектами в радиусе 500м",
            "year": "Год из даты",
            "month": "Месяц из даты",
            "day": "День месяца из даты",
            "hour": "Час дня из даты",
            "day_of_year": "День года (1-366) из даты",
            "season": "Классификация сезона (0=Зима, 1=Весна, 2=Лето, 3=Осень)",
            "distance_from_equator": "Расстояние от экватора в градусах широты",
            "distance_from_prime_meridian": "Расстояние от начального меридиана в градусах долготы",
            "altitude_variation": "Изменение высоты внутри маршрута в метрах",
            "altitude_change_rate": "Скорость изменения высоты между последовательными точками",
            "num_nearby_objects": "Количество ближайших географических объектов в радиусе 500м"
        }
        
        return meanings.get(col_name, f"Атрибут {col_name} в датасете")
    
    def _get_attribute_units(self, col_name: str) -> str:
        units = {
            "latitude": "градусы",
            "longitude": "градусы",
            "altitude": "метры",
            "frequency_steps": "шаги в минуту",
            "temperature": "Цельсии",
            "distance_from_equator": "градусы широты",
            "distance_from_prime_meridian": "градусы долготы",
            "altitude_variation": "метры",
            "altitude_change_rate": "процентное изменение",
            "num_nearby_objects": "количество"
        }
        
        return units.get(col_name, "безразмерный" if col_name in ['point_index', 'season', 'month', 'year'] else "неизвестно")
    
    def _get_valid_range(self, series) -> Dict:
        if series.dtype in ['int64', 'float64']:
            return {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std())
            }
        else:
            return {
                "unique_values": series.nunique(),
                "most_common": series.mode().iloc[0] if not series.mode().empty else "N/A"
            }
    
    def _get_attribute_purpose(self, col_name: str) -> str:
        purposes = {
            "track_id": "Идентификация и группировка точек, принадлежащих одному маршруту",
            "point_index": "Поддержание последовательного порядка GPS точек в маршруте",
            "date": "Временной контекст для погодных условий и сезонного анализа",
            "region": "Географический контекст и региональный анализ",
            "latitude": "Пространственное позиционирование и визуализация маршрута",
            "longitude": "Пространственное позиционирование и визуализация маршрута",
            "altitude": "Анализ рельефа и оценка сложности маршрута",
            "frequency_steps": "Анализ физической активности и оценка темпа",
            "temperature": "Оценка условий окружающей среды, влияющих на сложность маршрута",
            "terrain_type": "Классификация окружающей среды для планирования маршрута",
            "surrounding_objects": "Контекстные географические особенности для навигации",
            "year": "Долгосрочный анализ по годам",
            "month": "Сезонный анализ и выявление тенденций",
            "day": "Анализ ежедневных закономерностей",
            "hour": "Анализ по времени суток для планирования маршрута",
            "day_of_year": "Анализ сезонных закономерностей",
            "season": "Сезонный анализ характеристик маршрута",
            "distance_from_equator": "Анализ по широте для климатических и рельефных особенностей",
            "distance_from_prime_meridian": "Анализ по долготе",
            "altitude_variation": "Оценка топографической сложности маршрута",
            "altitude_change_rate": "Анализ крутизны и сложности подъема",
            "num_nearby_objects": "Плотность географических особенностей вокруг маршрута"
        }
        
        return purposes.get(col_name, "Общий признак для анализа маршрута")
    
    def _analyze_distributions(self, dataset: pd.DataFrame) -> Dict:
        results = {}
        
        numerical_cols = dataset.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            series = dataset[col].dropna()
            if len(series) < 3:
                continue
                
            normality_tests = self._perform_normality_tests(series)
            
            stats_summary = {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "variance": float(series.var()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurt()),
                "min": float(series.min()),
                "max": float(series.max()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
                "iqr": float(series.quantile(0.75) - series.quantile(0.25))
            }
            
            distribution_type = self._classify_distribution_type(series)
            
            results[col] = {
                "normality_tests": normality_tests,
                "statistics": stats_summary,
                "distribution_type": distribution_type,
                "needs_transformation": self._needs_transformation(stats_summary)
            }
        
        return results
    
    def _perform_normality_tests(self, series) -> Dict:
        dagostino_stat, dagostino_p = normaltest(series)
        
        shapiro_stat, shapiro_p = shapiro(series[:5000])
        
        anderson_result = anderson(series, dist='norm')
        
        return {
            "dagostino_pearson": {
                "statistic": float(dagostino_stat),
                "p_value": float(dagostino_p),
                "is_normal": dagostino_p > 0.05
            },
            "shapiro_wilk": {
                "statistic": float(shapiro_stat),
                "p_value": float(shapiro_p),
                "is_normal": shapiro_p > 0.05
            },
            "anderson_darling": {
                "statistic": float(anderson_result.statistic),
                "critical_values": [float(cv) for cv in anderson_result.critical_values],
                "significance_levels": [float(sl) for sl in anderson_result.significance_level]
            }
        }
    
    def _classify_distribution_type(self, series) -> str:
        skewness = series.skew()
        kurtosis = series.kurt()
        
        if abs(skewness) < 0.5:
            skew_desc = "симметричное"
        elif skewness > 0.5:
            skew_desc = "смещенное вправо"
        else:
            skew_desc = "смещенное влево"
        
        if abs(kurtosis) < 0.5:
            kurt_desc = "мезокуртичное (нормально-подобное)"
        elif kurtosis > 0.5:
            kurt_desc = "лептокуртичное (тяжелые хвосты)"
        else:
            kurt_desc = "платикуртичное (легкие хвосты)"
        
        return f"{skew_desc}, {kurt_desc}"
    
    def _needs_transformation(self, stats_summary: Dict) -> bool:
        skewness = stats_summary["skewness"]
        kurtosis = stats_summary["kurtosis"]
        
        return abs(skewness) > 1 or abs(kurtosis) > 1
    
    def _analyze_correlations(self, dataset: pd.DataFrame) -> Dict:
        numerical_cols = dataset.select_dtypes(include=[np.number]).columns
        corr_matrix = dataset[numerical_cols].corr()
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": float(corr_val)
                    })
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlation_pairs": high_corr_pairs,
            "plot_path": os.path.join('reports', 'correlation_matrix.png')
        }
    
    def _analyze_multimodal_distributions(self, dataset: pd.DataFrame) -> Dict:
        results = {}
        
        numerical_cols = dataset.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            series = dataset[col].dropna()
            if len(series) < 10:
                continue
                
            hist, bins = np.histogram(series, bins=30)
            peaks = []
            
            for i in range(1, len(hist) - 1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                    peaks.append({
                        "bin_center": (bins[i] + bins[i+1]) / 2,
                        "frequency": int(hist[i])
                    })
            
            results[col] = {
                "number_of_modes": len(peaks),
                "peaks": peaks,
                "is_multimodal": len(peaks) > 1
            }
        
        return results
    
    def _generate_visualizations(self, dataset: pd.DataFrame, analysis_results: Dict):
        os.makedirs('reports', exist_ok=True)
        
        numerical_cols = dataset.select_dtypes(include=[np.number]).columns
        plt.figure(figsize=(12, 10))
        sns.heatmap(dataset[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Матрица корреляций признаков')
        plt.tight_layout()
        plt.savefig(os.path.join('reports', 'correlation_matrix.png'))
        plt.close()
        
        numerical_cols = dataset.select_dtypes(include=[np.number]).columns[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(numerical_cols):
            if i >= len(axes):
                break
            axes[i].hist(dataset[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'Распределение {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Частота')
        
        plt.tight_layout()
        plt.savefig(os.path.join('reports', 'distributions.png'))
        plt.close()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        skewness_vals = [analysis_results['distribution_analysis'][col]['statistics']['skewness'] 
                         for col in analysis_results['distribution_analysis'].keys() 
                         if 'statistics' in analysis_results['distribution_analysis'][col]]
        cols = [col for col in analysis_results['distribution_analysis'].keys() 
                if 'statistics' in analysis_results['distribution_analysis'][col]]
        
        ax1.bar(range(len(cols)), skewness_vals)
        ax1.set_xticks(range(len(cols)))
        ax1.set_xticklabels(cols, rotation=45, ha='right')
        ax1.set_title('Асимметрия признаков')
        ax1.set_ylabel('Асимметрия')
        ax1.axhline(y=0, color='r', linestyle='--', label='Норма (0)')
        ax1.legend()
        
        kurtosis_vals = [analysis_results['distribution_analysis'][col]['statistics']['kurtosis'] 
                         for col in analysis_results['distribution_analysis'].keys() 
                         if 'statistics' in analysis_results['distribution_analysis'][col]]
        
        ax2.bar(range(len(cols)), kurtosis_vals)
        ax2.set_xticks(range(len(cols)))
        ax2.set_xticklabels(cols, rotation=45, ha='right')
        ax2.set_title('Эксцесс признаков')
        ax2.set_ylabel('Эксцесс')
        ax2.axhline(y=0, color='r', linestyle='--', label='Норма (0)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join('reports', 'skewness_kurtosis.png'))
        plt.close()