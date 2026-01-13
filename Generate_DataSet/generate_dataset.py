import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os
import uuid

def generate_large_dataset(num_tracks=1000, points_per_track=500):
    # Регионы
    regions = [
        "Москва и область", "Ленинградская область", "Краснодарский край", 
        "Республика Алтай", "Республика Карелия", "Сахалинская область",
        "Камчатский край", "Республика Бурятия", "Архангельская область",
        "Республика Коми", "Иркутская область", "Хабаровский край",
        "Ямало-Ненецкий АО", "Тюменская область", "Новосибирская область",
        "Омская область", "Томская область", "Кемеровская область",
        "Новгородская область", "Псковская область", "Вологодская область",
        "Мурманская область", "Республика Саха (Якутия)", "Магаданская область",
        "Чукотский АО", "Республика Тыва", "Республика Хакасия",
        "Алтайский край", "Красноярский край", "Республика Татарстан",
        "Удмуртская Республика", "Пермский край", "Республика Башкортостан"
    ]
    
    # Типы местности
    terrain_types = ["лес", "болото", "дорога", "река", "озеро", "горы", "степь", "тундра", "тайга", "ледник"]
    
    # Возможные объекты в радиусе 500м
    objects = [
        "источник воды", "перевал", "посёлок", "дорога", "река", "озеро", 
        "лес", "скалы", "лагерь", "горная вершина", "обзорная площадка", 
        "маяк", "радиовышка", "железнодорожная станция", "автобусная остановка"
    ]
    
    print("Начинаем генерацию объемного датасета...")
    print(f"Количество треков: {num_tracks}")
    print(f"Точек на трек: {points_per_track}")
    print(f"Общий объем записей: {num_tracks * points_per_track:,}")
    
    dataset = []
    
    for track_id in range(1, num_tracks+1):
        if track_id % 100 == 0:
            print(f"Обработано {track_id}/{num_tracks} треков...")
        
        region = random.choice(regions)
        
        start_date = datetime(2004, 1, 1) + timedelta(days=random.randint(0, 20*365))
        
        base_lat = random.uniform(45.0, 70.0)  # Россия
        base_lon = random.uniform(20.0, 180.0)
        
        for point_idx in range(points_per_track):
            # Добавляем небольшие изменения к координатам
            lat = base_lat + random.uniform(-0.1, 0.1)
            lon = base_lon + random.uniform(-0.1, 0.1)
            
            # Высота над уровнем моря
            altitude = random.uniform(-5, 5000)
            
            # Частота шагов
            frequency_steps = random.uniform(60, 120)
            
            # Температура в зависимости от региона и времени года
            current_date = start_date + timedelta(days=point_idx)
            month = current_date.month
            
            # Учитываем сезонность и региональные особенности
            if region in ["Республика Алтай", "Камчатский край", "Хабаровский край", "Республика Саха (Якутия)", "Магаданская область", "Чукотский АО"]:
                # Сибирские/далековосточные регионы
                seasonal_temp = [-35, -30, -20, -5, 8, 15, 18, 16, 8, -5, -20, -30][month-1]
            elif region in ["Краснодарский край", "Республика Крым", "Астраханская область", "Ростовская область"]:
                # Южные регионы
                seasonal_temp = [2, 4, 8, 15, 20, 25, 30, 28, 22, 14, 6, 2][month-1]
            elif region in ["Ленинградская область", "Республика Карелия", "Архангельская область", "Мурманская область"]:
                # Северные регионы
                seasonal_temp = [-10, -8, -2, 6, 14, 18, 20, 18, 12, 4, -2, -8][month-1]
            else:
                # Умеренный климат остальных регионов
                seasonal_temp = [-12, -10, -5, 5, 12, 18, 20, 18, 12, 5, -2, -10][month-1]
                
            # Добавляем немного случайности к температуре
            temperature = seasonal_temp + random.uniform(-5, 5)
            
            # Тип местности
            terrain_type = random.choice(terrain_types)
            
            # Объекты в радиусе 500м (случайное количество от 1 до 7)
            num_objects = random.randint(1, 7)
            surrounding_objects = "; ".join([random.choice(objects) for _ in range(num_objects)])
            
            # Создаем запись
            record = {
                "track_id": f"TRACK_{track_id:04d}",
                "point_index": point_idx,
                "date": current_date.strftime("%Y-%m-%d"),
                "region": region,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "altitude": round(altitude, 2),
                "frequency_steps": round(frequency_steps, 2),
                "temperature": round(temperature, 2),
                "terrain_type": terrain_type,
                "surrounding_objects": surrounding_objects
            }
            
            dataset.append(record)
    
    return pd.DataFrame(dataset)


def save_large_dataset_to_csv(df, filename):
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Датасет сохранён в {filename}")
    print(f"Размер датасета: {df.shape[0]:,} записей, {df.shape[1]} атрибутов")
    print(f"Период данных: {df['date'].min()} - {df['date'].max()}")
    
    # Статистика по регионам
    region_stats = df['region'].value_counts()
    print(f"\nТоп-10 регионов по количеству точек:")
    print(region_stats.head(10))


if __name__ == "__main__":
    random.seed(datetime.now().timestamp())
    np.random.seed(int(datetime.now().timestamp()))
    
    print("Период: 2004-2024")
    print("=" * 60)
    
    dataset = generate_large_dataset(num_tracks=1000, points_per_track=500)
    
    # Проверяем уникальность треков
    unique_tracks = dataset['track_id'].nunique()
    total_points = len(dataset)
    
    print("=" * 60)
    print(f"Генерация завершена!")
    print(f"Сгенерировано треков: {unique_tracks:,}")
    print(f"Всего точек: {total_points:,}")
    print(f"Среднее количество точек на трек: {total_points/unique_tracks:.1f}")
    
    # Генерируем уникальное имя файла на основе текущего времени по таймсмапу
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"large_route_dataset_500k_{timestamp}.csv"
    
    save_large_dataset_to_csv(dataset, filename)