# api/preprocess.py
"""
預處理模組
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple

class FacePreprocessor:
    """預處理管線"""
    
    def __init__(self):
        # 初始化MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 定義人臉中軸線
        self.FACEMESH_MID_LINE = frozenset([
            (10, 151), (151, 9), (9, 8), (8, 168),
            (168, 6), (6, 197), (197, 195), (195, 5), (5, 4),
            (4, 1), (1, 19), (19, 94), (94, 2)
        ])
        
        # CLAHE參數
        self.CLIP_LIMIT = 2.0
        self.TILE_GRID_SIZE = 8
        
        # 鏡射參數
        self.FEATHER_PX = 2  # 邊緣羽化像素
        self.ERODE_PX = 0    # 邊緣收縮像素
    
    def process_images(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        完整的預處理管線
        1. 角度校正
        2. 鏡射生成
        3. 直方圖校正
        
        Returns:
            List of (left_mirror, right_mirror) tuples
        """
        if not images:
            return []
        
        mirror_pairs = []
        
        for img in images:
            try:
                # 1. 角度校正（對齊）
                aligned = self.align_face(img)
                
                # 2. 生成鏡射對
                left_mirror, right_mirror = self.create_mirror_images(aligned)
                
                # 3. 直方圖校正（CLAHE）
                left_corrected = self.histogram_matching_clahe(left_mirror)
                right_corrected = self.histogram_matching_clahe(right_mirror)
                
                mirror_pairs.append((left_corrected, right_corrected))
                
            except Exception as e:
                print(f"處理圖片時發生錯誤: {e}")
                # 如果處理失敗，使用原圖
                mirror_pairs.append((img, img))
        
        return mirror_pairs
    
    def align_face(self, image: np.ndarray) -> np.ndarray:
        """
        使用中軸線所有點計算平均角度並旋轉
        """
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return image
        
        # 計算中軸線平均角度
        angle = self._mid_line_angle_all_points(results, h, w)
        
        # 旋轉圖片
        M = cv2.getRotationMatrix2D((w//2, h//2), -angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        
        return rotated
    
    def _mid_line_angle_all_points(self, results, height: int, width: int) -> float:
        """
        計算所有中軸線段的平均角度
        """
        angles = []
        
        for pair in self.FACEMESH_MID_LINE:
            point1 = results.multi_face_landmarks[0].landmark[pair[0]]
            point2 = results.multi_face_landmarks[0].landmark[pair[1]]
            
            dot1 = np.array([point1.x * width, point1.y * height, 0])
            dot2 = np.array([point2.x * width, point2.y * height, 0])
            
            vector1 = dot2 - dot1
            
            if vector1[1] == 0:
                angle1_deg = 90.0
            else:
                angle1 = np.arctan(vector1[0] / vector1[1])
                angle1_deg = np.degrees(angle1)
            
            angles.append(angle1_deg)
        
        avg_angle = sum(angles) / len(angles) if angles else 0
        return avg_angle
    
    def create_mirror_images(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成左臉和右臉的完整鏡射
        """
        h, w = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        
        if not results.multi_face_landmarks:
            # 簡單的左右分割
            mid = w // 2
            left_half = image[:, :mid]
            right_half = image[:, mid:]
            left_mirror = np.hstack([left_half, cv2.flip(left_half, 1)])
            right_mirror = np.hstack([cv2.flip(right_half, 1), right_half])
            return cv2.resize(left_mirror, (w, h)), cv2.resize(right_mirror, (w, h))
        
        # 獲取臉部標記點
        face_landmarks = results.multi_face_landmarks[0].landmark
        pts_xy = self._landmarks_to_xy(face_landmarks, image.shape)
        
        # 估計臉部中線
        p0, n = self._estimate_midline_from_landmarks(pts_xy)
        
        # 計算每個像素到中線的有號距離
        Y, X = np.mgrid[:h, :w]
        XY = np.stack([X, Y], axis=-1).reshape(-1, 2)
        d = ((XY - p0) @ n).reshape(h, w)
        
        # 生成左臉鏡射（保留左半邊）
        left_mirror = image.copy()
        for y in range(h):
            for x in range(w):
                if d[y, x] > 0:  # 右半邊
                    # 找到對應的左半邊像素
                    reflected_x = int(p0[0] - (x - p0[0]))
                    if 0 <= reflected_x < w:
                        left_mirror[y, x] = image[y, reflected_x]
        
        # 生成右臉鏡射（保留右半邊）
        right_mirror = image.copy()
        for y in range(h):
            for x in range(w):
                if d[y, x] < 0:  # 左半邊
                    # 找到對應的右半邊像素
                    reflected_x = int(p0[0] + (p0[0] - x))
                    if 0 <= reflected_x < w:
                        right_mirror[y, x] = image[y, reflected_x]
        
        return left_mirror, right_mirror
    
    def _landmarks_to_xy(self, landmarks, img_shape) -> np.ndarray:
        """將FaceMesh相對座標轉為像素座標"""
        h, w = img_shape[:2]
        pts = []
        for lm in landmarks:
            x = float(lm.x * w)
            y = float(lm.y * h)
            pts.append([x, y])
        return np.array(pts, dtype=np.float64)
    
    def _estimate_midline_from_landmarks(self, face_points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用PCA估計臉部中線
        返回：中線上一點p0和單位法向量n
        """
        # 使用中線關鍵點
        midline_idx = (10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2)
        
        # 過濾有效索引
        idx = np.array(midline_idx, dtype=int)
        idx = idx[(idx >= 0) & (idx < face_points_xy.shape[0])]
        
        if idx.size == 0:
            # 使用整臉點
            ml_pts = face_points_xy
        else:
            idx = np.unique(idx)
            ml_pts = face_points_xy[idx, :]
        
        # PCA計算主方向
        p0 = ml_pts.mean(axis=0)
        X = ml_pts - p0
        
        if not np.isfinite(X).all() or np.allclose(X, 0):
            # 退化情況：使用垂直中線
            xs = face_points_xy[:, 0]
            mid_x = 0.5 * (xs.min() + xs.max())
            p0 = np.array([mid_x, face_points_xy[:, 1].mean()], dtype=np.float64)
            n = np.array([1.0, 0.0], dtype=np.float64)
            return p0, n
        
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        u = Vt[0]
        u = u / (np.linalg.norm(u) + 1e-12)
        
        # 法向量
        n = np.array([-u[1], u[0]], dtype=np.float64)
        if n[0] < 0:  # 確保指向右側
            n = -n
        
        return p0, n
    
    def histogram_matching_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        在Lab色彩空間的L通道進行自適應直方圖均衡化
        """
        # 轉換到Lab色彩空間
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 應用CLAHE到L通道
        clahe = cv2.createCLAHE(
            clipLimit=self.CLIP_LIMIT,
            tileGridSize=(self.TILE_GRID_SIZE, self.TILE_GRID_SIZE)
        )
        l_eq = clahe.apply(l)
        
        # 合併通道
        lab_eq = cv2.merge([l_eq, a, b])
        result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        
        return result