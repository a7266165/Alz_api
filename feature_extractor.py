# api/feature_extractor.py
"""
VGGFace特徵提取模組
"""
import numpy as np
from deepface import DeepFace
import json
import logging
from typing import Optional, List, Tuple
import os
from datetime import datetime


logger = logging.getLogger(__name__)

class VGGFaceExtractor:
    """VGGFace特徵提取器"""
    
    def __init__(self, feature_selection_path: Optional[str] = None):
        """
        Args:
            feature_selection_path: 特徵選擇配置檔路徑
        """
        self.feature_selection = None
        
        if feature_selection_path:
            try:
                with open(feature_selection_path, 'r') as f:
                    self.feature_selection = json.load(f)
                print(f"✓ 載入特徵選擇: {self.feature_selection['original_dim']} -> {self.feature_selection['selected_dim']}維")
            except Exception as e:
                print(f"✗ 無法載入特徵選擇: {e}")
    
    def extract_vggface(self, img_array: np.ndarray) -> Optional[np.ndarray]:
        """
        提取VGGFace特徵
        
        Args:
            img_array: numpy array格式的圖片
            
        Returns:
            4096維特徵向量
        """
        try:
            result = DeepFace.represent(
                img_path=img_array,
                model_name='VGG-Face',
                enforce_detection=False,
                detector_backend='opencv',  # 已經預處理過了
                align=True  # 已經對齊過了
            )
            
            if result and len(result) > 0:
                embedding = np.array(result[0]['embedding'])
                return embedding
                
        except Exception as e:
            logger.error(f"VGGFace提取失敗: {e}")
        
        return None
    
    def calculate_difference(self, left_features: np.ndarray, right_features: np.ndarray) -> np.ndarray:
        """
        計算左右臉差異特徵（絕對值）
        
        Args:
            left_features: 左臉特徵
            right_features: 右臉特徵
            
        Returns:
            差異特徵向量
        """
        return np.abs(left_features - right_features)
    
    def process_image_pair(self, left_img: np.ndarray, right_img: np.ndarray) -> Optional[np.ndarray]:
        """
        處理一對左右臉圖片，提取差異特徵
        
        Args:
            left_img: 左臉鏡射圖
            right_img: 右臉鏡射圖
            
        Returns:
            差異特徵向量 (4096維)
        """
        # 提取VGGFace特徵
        left_features = self.extract_vggface(left_img)
        right_features = self.extract_vggface(right_img)
        
        if left_features is None or right_features is None:
            return None
        
        # 計算差異
        diff_features = self.calculate_difference(left_features, right_features)
        
        return diff_features
    
    def process_with_demographics(
        self, 
        mirror_pairs: List[Tuple[np.ndarray, np.ndarray]], 
        age: int, 
        gender: int
    ) -> np.ndarray:
        """
        處理多對鏡射圖片並加入人口學特徵，最後應用特徵選擇
        
        Args:
            mirror_pairs: [(左臉鏡射, 右臉鏡射), ...]
            age: 年齡
            gender: 性別 (0=女, 1=男)
            
        Returns:
            篩選後的特徵向量
        """
        # 1. 提取所有差異特徵
        all_differences = []
        for left_mirror, right_mirror in mirror_pairs:
            diff_features = self.process_image_pair(left_mirror, right_mirror)
            if diff_features is not None:
                all_differences.append(diff_features)
        
        if not all_differences:
            raise ValueError("無法提取有效特徵")
        
        # ========== 調試開始：儲存平均前特徵 ==========
        os.makedirs("temp", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"temp/features_before_average_{timestamp}.json", 'w') as f:
            json.dump({
                "stage": "before_average",
                "count": len(all_differences),
                "features": [feat.tolist() for feat in all_differences],
                "dimensions": [len(feat) for feat in all_differences],
                "age": age,
                "gender": gender
            }, f, indent=2)
        print(f"  [調試] 已儲存平均前特徵: temp/features_before_average_{timestamp}.json")
        # ========== 調試結束：儲存平均前特徵 ==========

        # 2. 平均多張照片的特徵 (4096維)
        avg_features = np.mean(all_differences, axis=0)
        
        # 3. 加入人口學特徵 (變成4098維)
        combined_features = np.concatenate([
            avg_features,      # 4096維 VGGFace差異特徵
            [age, gender]      # 2維 人口學特徵
        ])

        # ========== 調試開始：儲存平均後特徵 ==========
        with open(f"temp/features_after_average_{timestamp}.json", 'w') as f:
            json.dump({
                "stage": "after_average",
                "feature_vector": combined_features.tolist(),
                "dimensions": len(combined_features),
                "vggface_dims": len(avg_features),
                "age": age,
                "gender": gender
            }, f, indent=2)
        print(f"  [調試] 已儲存平均後特徵: temp/features_after_average_{timestamp}.json")
        # ========== 調試結束：儲存平均後特徵 ==========

        # 4. 應用特徵選擇
        selected_features = self.apply_feature_selection(combined_features)
        
        # ========== 調試開始：儲存篩選後特徵 ==========
        with open(f"temp/features_after_selection_{timestamp}.json", 'w') as f:
            json.dump({
                "stage": "after_selection",
                "feature_vector": selected_features.tolist(),
                "dimensions": len(selected_features),
                "original_dims": len(combined_features),
                "age": age,
                "gender": gender
            }, f, indent=2)
        print(f"  [調試] 已儲存篩選後特徵: temp/features_after_selection_{timestamp}.json")
        # ========== 調試結束：儲存篩選後特徵 ==========

        return selected_features
    
    def apply_feature_selection(self, features: np.ndarray) -> np.ndarray:
        """
        篩選特徵
        
        Args:
            features: 完整特徵向量（4098維 = 4096 VGGFace + 2 demographics）
            
        Returns:
            篩選後的特徵（166維）
        """
        if self.feature_selection is None:
            # 沒有特徵選擇配置，返回原始特徵
            return features
        
        selected_indices = self.feature_selection['selected_indices']
        
        # 確保索引在範圍內
        valid_indices = [i for i in selected_indices if i < len(features)]
        
        if len(valid_indices) != len(selected_indices):
            logger.warning(
                f"部分索引超出範圍：{len(selected_indices)} -> {len(valid_indices)}"
            )
        
        return features[valid_indices]