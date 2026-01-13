## Структура базы данных

### 1. Таблица tracks

Хранит информацию на уровне трека:

- **id** (INTEGER, PRIMARY KEY, AUTOINCREMENT)
- **track_id** (TEXT, UNIQUE)
- **name** (TEXT)
- **region** (TEXT)
- **date_created** (TEXT)
- **source_url** (TEXT)
- **geodetic_correction_applied** (BOOLEAN)
- **anomalies_detected** (INTEGER)
- **created_at** (TIMESTAMP)
- **updated_at** (TIMESTAMP)

### 2. Таблица track_points

Хранит отдельные GPS точки:

- **id** (INTEGER, PRIMARY KEY, AUTOINCREMENT)
- **track_id** (TEXT)
- **point_index** (INTEGER)
- **latitude** (REAL)
- **longitude** (REAL)
- **altitude** (REAL)
- **timestamp** (TEXT)
- **corrected_latitude** (REAL)
- **corrected_longitude** (REAL)
- **hash_value** (TEXT, UNIQUE)

### 3. Таблица anomalies

Хранит обнаруженные аномалии:

- **id** (INTEGER, PRIMARY KEY, AUTOINCREMENT)
- **track_id** (TEXT)
- **point_index** (INTEGER)
- **anomaly_type** (TEXT)
- **original_lat** (REAL)
- **original_lon** (REAL)
- **corrected_lat** (REAL)
- **corrected_lon** (REAL)
- **severity** (REAL)
- **detected_at** (TIMESTAMP)