import os
import json
import sqlite3
import requests
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
from pyproj import Transformer
from geopy.distance import geodesic
import pandas as pd
import hashlib
import uuid


class DataLoader:
    def __init__(self):
        self.db_path = os.path.join('data', 'tracks.db')
        self.transformer_wgs84_to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self.transformer_mercator_to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        
        os.makedirs('data', exist_ok=True)
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('data/external', exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id TEXT UNIQUE,
                name TEXT,
                region TEXT,
                date_created TEXT,
                source_url TEXT,
                geodetic_correction_applied BOOLEAN DEFAULT FALSE,
                anomalies_detected INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS track_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id TEXT,
                point_index INTEGER,
                latitude REAL,
                longitude REAL,
                altitude REAL,
                timestamp TEXT,
                corrected_latitude REAL,
                corrected_longitude REAL,
                hash_value TEXT UNIQUE,
                FOREIGN KEY (track_id) REFERENCES tracks (track_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id TEXT,
                point_index INTEGER,
                anomaly_type TEXT,
                original_lat REAL,
                original_lon REAL,
                corrected_lat REAL,
                corrected_lon REAL,
                severity REAL,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (track_id) REFERENCES tracks (track_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_tracks_with_corrections(self, track_urls: List[str] = None) -> Dict:
        if track_urls is None:
            track_urls = [
                "https://example.com/track1.gpx",
                "https://example.com/track2.kml"
            ]
        
        all_tracks = {}
        
        for url in track_urls:
            try:
                track_data = self._download_and_parse_track(url)
                
                corrected_track = self._apply_geodetic_corrections(track_data)
                
                corrected_track = self._detect_and_fix_anomalies(corrected_track)
                
                existing_track_id = self._check_track_exists(url, corrected_track)
                
                if existing_track_id:
                    print(f"Трек из {url} уже существует в базе данных с ID: {existing_track_id}")
                    all_tracks[existing_track_id] = corrected_track
                else:
                    track_id = self._store_track_in_db(corrected_track, url)
                    all_tracks[track_id] = corrected_track
                    print(f"Новый трек добавлен с ID: {track_id}")
                
                self._get_topographic_map_image(corrected_track, track_id if not existing_track_id else existing_track_id)
                
            except Exception as e:
                print(f"Ошибка при обработке трека из {url}: {str(e)}")
                continue
        
        return all_tracks
    
    def _check_track_exists(self, source_url: str, track_data: Dict) -> str:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT track_id FROM tracks WHERE source_url = ?", (source_url,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return result[0]
        return None
    
    def _download_and_parse_track(self, url: str) -> Dict:
        sample_points = [
            {"latitude": 55.7558, "longitude": 37.6173, "altitude": 156.0, "timestamp": "2023-09-15T10:00:00Z"},
            {"latitude": 55.7560, "longitude": 37.6180, "altitude": 157.2, "timestamp": "2023-09-15T10:05:00Z"},
            {"latitude": 65.7565, "longitude": 77.6185, "altitude": 158.5, "timestamp": "2023-09-15T10:10:00Z"},
            {"latitude": 55.7570, "longitude": 37.6190, "altitude": 160.0, "timestamp": "2023-09-15T10:15:00Z"},
            {"latitude": 55.7575, "longitude": 37.6195, "altitude": 162.3, "timestamp": "2023-09-15T10:20:00Z"},
        ]
        
        return {
            "name": f"Трек из {url.split('/')[-1]}",
            "region": "Москва",
            "date_created": "2023-09-15",
            "points": sample_points
        }
    
    def _apply_geodetic_corrections(self, track_data: Dict) -> Dict:
        corrected_points = []
        
        for point in track_data["points"]:
            lat, lon = point["latitude"], point["longitude"]
            
            x, y = self.transformer_wgs84_to_mercator.transform(lon, lat)
            corrected_lon, corrected_lat = self.transformer_mercator_to_wgs84.transform(x, y)
            
            corrected_point = point.copy()
            corrected_point["corrected_latitude"] = corrected_lat
            corrected_point["corrected_longitude"] = corrected_lon
            
            corrected_points.append(corrected_point)
        
        track_data["points"] = corrected_points
        return track_data
    
    def _detect_and_fix_anomalies(self, track_data: Dict) -> Dict:
        points = track_data["points"]
        anomalies = []
        
        for i, point in enumerate(points):
            if i > 0:
                prev_point = points[i-1]
                
                dist = geodesic(
                    (prev_point.get("corrected_latitude", prev_point["latitude"]), 
                     prev_point.get("corrected_longitude", prev_point["longitude"])),
                    (point.get("corrected_latitude", point["latitude"]), 
                     point.get("corrected_longitude", point["longitude"]))
                ).meters
                
                curr_time = datetime.fromisoformat(point["timestamp"].replace('Z', '+00:00'))
                prev_time = datetime.fromisoformat(prev_point["timestamp"].replace('Z', '+00:00'))
                time_diff = abs((curr_time - prev_time).total_seconds())
                
                if time_diff > 0:
                    speed = dist / time_diff
                    
                    if speed > 27.78:
                        anomalies.append({
                            "index": i,
                            "original_lat": point["latitude"],
                            "original_lon": point["longitude"],
                            "distance": dist,
                            "time_diff": time_diff,
                            "speed": speed,
                            "type": "high_speed_anomaly"
                        })
                        
                        if i < len(points) - 1:
                            next_point = points[i+1]
                            
                            interpolated_lat = (prev_point["latitude"] + next_point["latitude"]) / 2
                            interpolated_lon = (prev_point["longitude"] + next_point["longitude"]) / 2
                            
                            point["corrected_latitude"] = interpolated_lat
                            point["corrected_longitude"] = interpolated_lon
        
        track_data["anomalies"] = anomalies
        return track_data
    
    def _store_track_in_db(self, track_data: Dict, source_url: str) -> str:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        track_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_url}_{datetime.now().isoformat()}"))
        
        cursor.execute('''
            INSERT OR REPLACE INTO tracks (track_id, name, region, date_created, source_url, geodetic_correction_applied, anomalies_detected)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            track_id,
            track_data["name"],
            track_data["region"],
            track_data["date_created"],
            source_url,
            True,
            len(track_data.get("anomalies", []))
        ))
        
        for idx, point in enumerate(track_data["points"]):
            point_data_str = f"{track_id}_{idx}_{point['latitude']}_{point['longitude']}_{point['timestamp']}"
            hash_value = hashlib.sha256(point_data_str.encode()).hexdigest()
            
            cursor.execute('''
                INSERT OR IGNORE INTO track_points (track_id, point_index, latitude, longitude, altitude, timestamp, corrected_latitude, corrected_longitude, hash_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                track_id,
                idx,
                point["latitude"],
                point["longitude"],
                point.get("altitude", 0),
                point["timestamp"],
                point.get("corrected_latitude", point["latitude"]),
                point.get("corrected_longitude", point["longitude"]),
                hash_value
            ))
        
        for anomaly in track_data.get("anomalies", []):
            cursor.execute('''
                INSERT OR IGNORE INTO anomalies (track_id, point_index, anomaly_type, original_lat, original_lon, corrected_lat, corrected_lon)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                track_id,
                anomaly["index"],
                anomaly["type"],
                anomaly["original_lat"],
                anomaly["original_lon"],
                track_data["points"][anomaly["index"]]["corrected_latitude"],
                track_data["points"][anomaly["index"]]["corrected_longitude"]
            ))
        
        conn.commit()
        conn.close()
        
        return track_id
    
    def _get_topographic_map_image(self, track_data: Dict, track_id: str):
        print(f"Получение топографической карты для трека {track_id}")
        print(f"Найдено {len(track_data.get('anomalies', []))} аномалий в треке {track_id}")
        
        coords_file = os.path.join('data', 'raw', f'{track_id}_coords.json')
        with open(coords_file, 'w') as f:
            json.dump(track_data["points"], f)
    
    def _get_weather_data(self, lat: float, lon: float, timestamp: str) -> Dict:
        date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        month = date.month
        base_temp = 20
        
        seasonal_factor = 10 * np.cos((month - 1) * (2 * np.pi / 12) - np.pi/6)
        temp = base_temp + seasonal_factor + np.random.uniform(-5, 5)
        
        return {
            "temperature": round(temp, 2),
            "humidity": np.random.randint(30, 90),
            "pressure": np.random.randint(980, 1040),
            "wind_speed": round(np.random.uniform(0, 15), 2)
        }