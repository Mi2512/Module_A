import os
import sys
import cleanup

def main():
    print("Запуск ракеты A: ")
    
    # Очищаем старые данные и файлы при каждой генерации
    cleanup.cleanup_old_files()
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

    from src.data_loader import DataLoader
    from src.dataset_former import DatasetFormer
    from src.preprocessors import Preprocessor
    from src.analyzer import Analyzer
    from src.augmenter import Augmenter
    from src.reporter import Reporter

    loader = DataLoader()
    former = DatasetFormer()
    processor = Preprocessor()
    analyzer = Analyzer()
    augmenter = Augmenter()
    reporter = Reporter()
    
    print("Шаг 1: Загрузка данных с гео. коррекциями...")
    tracks_data = loader.load_tracks_with_corrections()
    
    print("Шаг 2: Формирование датасета с декодированием карт...")
    dataset = former.form_dataset(tracks_data)
    
    print("Шаг 3: Предобработка и извлечение признаков...")
    processed_dataset = processor.preprocess(dataset)
    
    print("Шаг 4: Анализ структуры датасета и распределений...")
    analysis_results = analyzer.analyze(processed_dataset)
    
    print("Шаг 5: Расширение датасета с гео. точностью...")
    expanded_dataset = augmenter.expand_dataset(processed_dataset)
    
    print("Шаг 6: Генерация финального отчета...")
    reporter.generate_report(expanded_dataset, analysis_results)
    
    print("Цепочка выполнена успешно!")

if __name__ == "__main__":
    main()