import os
import json
import sqlite3
import pandas as pd
import numpy as np
import requests
from typing import List, Dict, Tuple
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pyproj import Transformer
from geopy.distance import geodesic
import rasterio
from shapely.geometry import LineString, Point
from shapely.ops import transform


class Augmenter:
    def __init__(self):
        self.transformer_wgs84_to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self.transformer_mercator_to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        
    def expand_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        additional_tracks = self._get_tracks_from_public_platforms()
        
        formatted_tracks = self._format_additional_tracks(additional_tracks)
        
        expanded_dataset = pd.concat([dataset, formatted_tracks], ignore_index=True)
        
        self._augment_route_images()
        
        os.makedirs('data/processed', exist_ok=True)
        expanded_dataset.to_csv(os.path.join('data', 'processed', 'expanded_dataset.csv'), index=False)
        
        return expanded_dataset
    
    def _get_tracks_from_public_platforms(self) -> List[Dict]:
        sample_tracks = []
        
        platforms = ["alltrails", "komoot", "openstreetmap"]
        
        for platform in platforms:
            for i in range(3):
                base_lat = 55.7558 + np.random.uniform(-1, 1)
                base_lon = 37.6173 + np.random.uniform(-1, 1)
                
                track_points = []
                for j in range(20):
                    lat_offset = np.random.uniform(-0.01, 0.01)
                    lon_offset = np.random.uniform(-0.01, 0.01)
                    
                    point = {
                        "latitude": base_lat + lat_offset,
                        "longitude": base_lon + lon_offset,
                        "altitude": np.random.uniform(100, 300),
                        "timestamp": f"2023-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}T{np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}:00Z"
                    }
                    track_points.append(point)
                
                track = {
                    "track_id": f"{platform}_track_{i}",
                    "name": f"{platform.capitalize()} Track {i}",
                    "region": "Simulated Region",
                    "source": platform,
                    "points": track_points
                }
                
                sample_tracks.append(track)
        
        return sample_tracks
    
    def _format_additional_tracks(self, additional_tracks: List[Dict]) -> pd.DataFrame:
        dataset_rows = []
        
        for track in additional_tracks:
            for i, point in enumerate(track["points"]):
                lat = point.get("corrected_latitude", point["latitude"])
                lon = point.get("corrected_longitude", point["longitude"])
                
                env_data = self._extract_environmental_data(lat, lon)
                
                weather_data = self._get_weather_data(lat, lon, point["timestamp"])
                
                frequency_steps = self._estimate_frequency_steps(i, track["points"])
                
                row = {
                    "track_id": track["track_id"],
                    "point_index": i,
                    "date": point["timestamp"],
                    "region": track["region"],
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
        
        return pd.DataFrame(dataset_rows)
    
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
        import random
        from datetime import datetime
        
        date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        month = date.month
        base_temp = 20
        
        seasonal_factor = 10 * np.cos((month - 1) * (2 * np.pi / 12) - np.pi/6)
        temp = base_temp + seasonal_factor + random.uniform(-5, 5)
        
        return {
            "temperature": round(temp, 2),
            "humidity": random.randint(30, 90),
            "pressure": random.randint(980, 1040),
            "wind_speed": round(random.uniform(0, 15), 2)
        }
    
    def _estimate_frequency_steps(self, point_idx: int, all_points: List[Dict]) -> float:
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
        
        from datetime import datetime
        curr_time = datetime.fromisoformat(curr_point["timestamp"].replace('Z', '+00:00'))
        prev_time = datetime.fromisoformat(prev_point["timestamp"].replace('Z', '+00:00'))
        time_diff = abs((curr_time - prev_time).total_seconds()) / 60
        
        if time_diff > 0:
            speed = dist / time_diff
            steps_per_min = speed / 0.7
            return min(max(steps_per_min, 40), 140)
        else:
            return 80.0
    
    def _augment_route_images(self):
        os.makedirs('data/processed/images', exist_ok=True)
        
        for i in range(3):
            img_size = (800, 600)
            img = np.ones((*img_size[::-1], 3), dtype=np.uint8) * 240
            
            route_points = []
            for pt_idx in range(10):
                x = 100 + pt_idx * 50 + np.random.uniform(-10, 10)
                y = 300 + np.random.uniform(-50, 50) * np.sin(pt_idx / 2)
                route_points.append((int(x), int(y)))
            
            for j in range(len(route_points) - 1):
                pt1 = route_points[j]
                pt2 = route_points[j + 1]
                cv2.line(img, pt1, pt2, (0, 0, 255), 3)
            
            for pt in route_points:
                cv2.circle(img, pt, 5, (0, 255, 0), -1)
            
            original_path = os.path.join('data/processed/images', f'route_original_{i}.png')
            cv2.imwrite(original_path, img)
            
            augmented_imgs = self._apply_geometric_augmentation(img)
            
            for aug_idx, aug_img in enumerate(augmented_imgs):
                bright_img = self._adjust_brightness(aug_img, factor=np.random.uniform(0.8, 1.2))
                contrast_img = self._adjust_contrast(bright_img, factor=np.random.uniform(0.8, 1.2))
                saturation_img = self._adjust_saturation(contrast_img, factor=np.random.uniform(0.8, 1.2))
                
                aug_path = os.path.join('data/processed/images', f'route_augmented_{i}_{aug_idx}.png')
                cv2.imwrite(aug_path, saturation_img)
        
        print("Аугментация маршрутных изображений завершена")
    
    def _apply_geometric_augmentation(self, img: np.ndarray) -> List[np.ndarray]:
        augmented_images = [img]
        
        h, w = img.shape[:2]
        
        for angle in [-15, 15]:
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
            rotated = cv2.warpAffine(img, M, (w, h))
            augmented_images.append(rotated)
        
        pts1 = np.float32([[50, 50], [w-50, 50], [50, h-50], [w-50, h-50]])
        pts2 = np.float32([
            [50 + np.random.uniform(-20, 20), 50 + np.random.uniform(-20, 20)],
            [w-50 + np.random.uniform(-20, 20), 50 + np.random.uniform(-20, 20)],
            [50 + np.random.uniform(-20, 20), h-50 + np.random.uniform(-20, 20)],
            [w-50 + np.random.uniform(-20, 20), h-50 + np.random.uniform(-20, 20)]
        ])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(img, M, (w, h))
        augmented_images.append(warped)
        
        tx, ty = np.random.uniform(-30, 30, 2)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(img, M, (w, h))
        augmented_images.append(translated)
        
        return augmented_images
    
    def _adjust_brightness(self, img: np.ndarray, factor: float) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float64)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def _adjust_contrast(self, img: np.ndarray, factor: float) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        adjusted = cv2.merge((l, a, b))
        return cv2.cvtColor(adjusted, cv2.COLOR_LAB2BGR)
    
    def _adjust_saturation(self, img: np.ndarray, factor: float) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float64)
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)