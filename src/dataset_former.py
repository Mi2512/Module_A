import os
import json
import sqlite3
import requests
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from geopy.distance import geodesic
import rasterio
from rasterio.features import bounds as feature_bounds
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from datetime import datetime


class DatasetFormer:
    def __init__(self):
        self.db_path = os.path.join('data', 'tracks.db')
        self.map_legend_mapping = self._create_legend_mapping()
        
    def _create_legend_mapping(self) -> Dict:
        return {
            "google_maps": {
                "#FF0000": "road_primary",
                "#FFA500": "road_secondary",
                "#00FF00": "forest",
                "#0000FF": "water",
                "#800080": "urban_area",
            },
            "yandex_maps": {
                "#FF0000": "road_primary",
                "#FF8C00": "road_secondary",
                "#228B22": "forest",
                "#1E90FF": "water",
                "#800080": "urban_area",
            },
            "openstreetmap": {
                "#F2D0A4": "road_primary",
                "#F2E5BF": "road_secondary",
                "#B5EAAA": "forest",
                "#ADD8E6": "water",
                "#D3D3D3": "urban_area",
            }
        }
    
    def form_dataset(self, tracks_data: Dict) -> pd.DataFrame:
        dataset_rows = []
        
        for track_id, track_info in tracks_data.items():
            points = track_info["points"]
            
            for i, point in enumerate(points):
                lat = point.get("corrected_latitude", point["latitude"])
                lon = point.get("corrected_longitude", point["longitude"])
                
                env_data = self._extract_environmental_data(lat, lon)
                
                weather_data = self._get_weather_data(lat, lon, point["timestamp"])
                
                frequency_steps = self._calculate_frequency_steps(i, points)
                
                row = {
                    "track_id": track_id,
                    "point_index": i,
                    "date": point["timestamp"],
                    "region": track_info["region"],
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": point.get("altitude", 0),
                    "frequency_steps": frequency_steps,
                    "temperature": weather_data.get("temperature", 0),
                    "terrain_type": env_data.get("terrain_type", "unknown"),
                    "surrounding_objects": json.dumps(env_data.get("objects", [])),
                    "environment_radius_500m": json.dumps(env_data.get("features", {})),
                }
                
                dataset_rows.append(row)
        
        df = pd.DataFrame(dataset_rows)
        
        os.makedirs('data/processed', exist_ok=True)
        df.to_csv(os.path.join('data', 'processed', 'complete_dataset.csv'), index=False)
        
        return df
    
    def _extract_environmental_data(self, lat: float, lon: float) -> Dict:
        center_point = Point(lon, lat)
        
        nearby_features = []
        
        for _ in range(5):
            angle = np.random.uniform(0, 2*np.pi)
            distance = np.random.uniform(0, 500)
            
            new_lat = lat + (distance / 111000) * np.cos(angle)
            new_lon = lon + (distance / (111000 * np.cos(np.radians(lat)))) * np.sin(angle)
            
            feature_types = ["forest", "road", "water", "urban", "meadow", "mountain"]
            feature_type = np.random.choice(feature_types)
            
            nearby_features.append({
                "type": feature_type,
                "coordinates": [new_lat, new_lon],
                "distance": distance
            })
        
        terrain_counts = {}
        for feature in nearby_features:
            terrain_type = feature["type"]
            terrain_counts[terrain_type] = terrain_counts.get(terrain_type, 0) + 1
        
        dominant_terrain = max(terrain_counts, key=terrain_counts.get) if terrain_counts else "unknown"
        
        return {
            "terrain_type": dominant_terrain,
            "objects": nearby_features,
            "features": {
                "radius": 500,
                "center": [lat, lon],
                "nearby_features_count": len(nearby_features)
            }
        }
    
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
    
    def _calculate_frequency_steps(self, point_idx: int, all_points: List[Dict]) -> float:
        if point_idx == 0:
            return 80.0
        
        prev_point = all_points[point_idx - 1]
        curr_point = all_points[point_idx]
        
        dist = geodesic(
            (prev_point.get("corrected_latitude", prev_point["latitude"]), 
             prev_point.get("corrected_longitude", prev_point["longitude"])),
            (curr_point.get("corrected_latitude", curr_point["latitude"]), 
             curr_point.get("corrected_longitude", curr_point["longitude"]))
        ).meters
        
        curr_time = datetime.fromisoformat(curr_point["timestamp"].replace('Z', '+00:00'))
        prev_time = datetime.fromisoformat(prev_point["timestamp"].replace('Z', '+00:00'))
        time_diff = abs((curr_time - prev_time).total_seconds()) / 60
        
        if time_diff > 0:
            speed = dist / time_diff
            steps_per_min = speed / 0.7
            return float(min(max(steps_per_min, 40), 140))
        else:
            return 80.0