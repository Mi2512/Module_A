import os
import shutil

def cleanup_old_files():
    data_processed_dir = os.path.join('data', 'processed')
    if os.path.exists(data_processed_dir):
        for file in os.listdir(data_processed_dir):
            file_path = os.path.join(data_processed_dir, file)
            if os.path.isfile(file_path) and (file.endswith('.csv') or file.endswith('.json')):
                os.remove(file_path)
                print(f"Удален старый файл: {file_path}")
    
    images_dir = os.path.join('data', 'processed', 'images')
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
        print(f"Удалена старая директория изображений: {images_dir}")
    
    reports_dir = 'reports'
    if os.path.exists(reports_dir):
        for file in os.listdir(reports_dir):
            file_path = os.path.join(reports_dir, file)
            if os.path.isfile(file_path) and file.endswith('.md'):
                os.remove(file_path)
                print(f"Удален старый файл отчета: {file_path}")
    
    db_path = os.path.join('data', 'tracks.db')
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Удалена старая база данных: {db_path}")
    
    print("Очистка завершена")


if __name__ == "__main__":
    cleanup_old_files()