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
        self.FEATHER_PX = 2      # 邊緣羽化像素
        self.ERODE_PX = 0        # 邊緣收縮像素
        self.MIRROR_SIZE = (512, 512)  # 輸出尺寸
        self.MARGIN = 0.08       # 畫布邊緣留白比例
    
    def process_images(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        完整的預處理管線
        1. 鏡射生成  # 原本是角度校正
        2. 直方圖校正
        3. 角度校正  # 原本是鏡射生成
        
        Returns:
            List of (left_mirror, right_mirror) tuples
        """
        if not images:
            return []
        
        mirror_pairs = []
        
        for img in images:
            try:
                # 1. 生成鏡射對（先鏡射，不先校正）
                left_mirror, right_mirror = self.create_mirror_images(img)
                
                # 2. 直方圖校正（CLAHE）
                left_corrected = self.apply_clahe(left_mirror)
                right_corrected = self.apply_clahe(right_mirror)
                
                # 3. 角度校正（最後才對齊）
                left_aligned = self.align_face(left_corrected)
                right_aligned = self.align_face(right_corrected)
                
                mirror_pairs.append((left_aligned, right_aligned))
                
            except Exception as e:
                print(f"處理圖片時發生錯誤: {e}")
                # 如果處理失敗，使用原圖
                mirror_pairs.append((img, img))
        
        # DEBUG: 儲存處理後的圖片
        for idx, (left_img, right_img) in enumerate(mirror_pairs):
            cv2.imwrite(f"temp/preprocess_pic/left_mirror_{idx}.png", left_img)
            cv2.imwrite(f"temp/preprocess_pic/right_mirror_{idx}.png", right_img)

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
        angle = self._calculate_face_angle(results, h, w)
        
        # 旋轉圖片
        M = cv2.getRotationMatrix2D((w//2, h//2), -angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        
        return rotated
    
    def _calculate_face_angle(self, results, height: int, width: int) -> float:
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
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            # 沒有偵測到臉，返回原圖的複製
            return image.copy(), image.copy()
        
        # 取得臉部標點
        face_landmarks = results.multi_face_landmarks[0].landmark
        pts_xy = self._landmarks_to_xy(face_landmarks, image.shape)
        
        # 建立臉部遮罩
        mask = self._build_face_mask(image.shape, pts_xy)
        
        # 估計中線
        p0, n = self._estimate_midline(pts_xy)
        
        # 建立鏡射影像 - 使用與4080版本相同的方法
        left_mirror = self._align_to_canvas_premul(
            image, mask, p0, n,
            side='left',
            out_size=self.MIRROR_SIZE,
            margin=self.MARGIN
        )
        
        right_mirror = self._align_to_canvas_premul(
            image, mask, p0, n,
            side='right',
            out_size=self.MIRROR_SIZE,
            margin=self.MARGIN
        )
        
        return left_mirror, right_mirror
    
    def _landmarks_to_xy(self, landmarks, img_shape: tuple) -> np.ndarray:
        """將 FaceMesh 的相對座標轉為像素座標"""
        h, w = img_shape[:2]
        pts = []
        for lm in landmarks:
            x = float(lm.x * w)
            y = float(lm.y * h)
            pts.append([x, y])
        return np.array(pts, dtype=np.float64)
    
    def _build_face_mask(self, img_shape: tuple, face_points_xy: np.ndarray) -> np.ndarray:
        """建立臉部遮罩"""
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        if face_points_xy.shape[0] == 0:
            return mask
        hull = cv2.convexHull(face_points_xy.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        return mask
    
    def _estimate_midline(
        self,
        face_points_xy: np.ndarray,
        midline_idx: tuple = (10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """估計臉部中線 - 與4080版本對齊"""
        idx = np.array(midline_idx, dtype=int)
        idx = idx[(idx >= 0) & (idx < face_points_xy.shape[0])]
        
        if idx.size == 0:
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
    
    def _make_half_alpha(
        self,
        mask_u8: np.ndarray,
        d: np.ndarray,
        side: str,
        feather_px: int = 2,
        erode_px: int = 0
    ) -> np.ndarray:
        """生成半臉 alpha mask with feathering"""
        h, w = mask_u8.shape
        a_mask = mask_u8.astype(np.float32) / 255.0
        
        # 根據 side 生成基本 alpha
        if side == 'left':
            alpha = np.where(d <= 0, 1.0, 0.0)
        else:
            alpha = np.where(d >= 0, 1.0, 0.0)
        
        # 邊緣羽化
        if feather_px > 0:
            if side == 'left':
                # 左半臉：在 d=0 附近羽化
                alpha = np.where(
                    np.abs(d) < feather_px,
                    0.5 - 0.5 * d / feather_px,
                    alpha
                )
            else:
                # 右半臉：在 d=0 附近羽化
                alpha = np.where(
                    np.abs(d) < feather_px,
                    0.5 + 0.5 * d / feather_px,
                    alpha
                )
        
        # 與臉部遮罩結合
        alpha = alpha * a_mask
        
        # 可選的邊緣收縮
        if erode_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2*erode_px+1, 2*erode_px+1)
            )
            alpha = cv2.erode(alpha, kernel)
        
        return alpha.astype(np.float32)
    
    def _remap_premultiplied(
        self,
        img_bgr: np.ndarray,
        alpha: np.ndarray,
        Xr: np.ndarray,
        Yr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """使用預乘 alpha 進行重映射"""
        img_f = img_bgr.astype(np.float32) / 255.0
        premul = img_f * alpha[..., None]
        
        # 重映射
        premul_ref = cv2.remap(
            premul, Xr, Yr,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        a_ref = cv2.remap(
            alpha, Xr, Yr,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        eps = 1e-6
        rgb_ref = np.where(
            a_ref[..., None] > eps,
            premul_ref / a_ref[..., None],
            0
        )
        
        rgb_ref_u8 = np.clip(rgb_ref * 255.0, 0, 255).astype(np.uint8)
        return rgb_ref_u8, a_ref
    
    def _align_to_canvas_premul(
        self,
        img_bgr: np.ndarray,
        mask_u8: np.ndarray,
        p0: np.ndarray,
        n: np.ndarray,
        side: str,
        out_size: Tuple[int, int] = (512, 512),
        margin: float = 0.08
    ) -> np.ndarray:
        """
        對齊到畫布並使用預乘 alpha - 與4080版本對齊
        """
        H, W = out_size
        h, w = img_bgr.shape[:2]
        
        # 計算有號距離
        X, Y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32)
        )
        d = (X - p0[0]) * n[0] + (Y - p0[1]) * n[1]
        
        # 反射座標
        Xr = X - 2.0 * d * n[0]
        Yr = Y - 2.0 * d * n[1]
        
        # 建立半臉 alpha
        alpha_half = self._make_half_alpha(mask_u8, d, side, self.FEATHER_PX, self.ERODE_PX)
        
        # 反射半臉
        reflect_rgb, reflect_a = self._remap_premultiplied(img_bgr, alpha_half, Xr, Yr)
        
        # 合成
        img_f = img_bgr.astype(np.float32) / 255.0
        eps = 1e-6
        
        premul = img_f * alpha_half[..., None] + \
                 (reflect_rgb.astype(np.float32)/255.0) * reflect_a[..., None]
        alpha_total = np.clip(alpha_half + reflect_a, 0.0, 1.0)
        
        composite = np.where(
            alpha_total[..., None] > eps,
            premul / alpha_total[..., None],
            0
        )
        composite = (np.clip(composite, 0, 1) * 255.0).astype(np.uint8)
        
        # 更新遮罩
        mask_composite = np.clip(alpha_total * 255.0, 0, 255).astype(np.uint8)
        
        # 旋轉對齊並縮放到輸出尺寸
        result = self._rotate_and_scale_to_canvas(
            composite, mask_composite, p0, n, out_size, margin
        )
        
        return result
    
    def _rotate_and_scale_to_canvas(
        self,
        img_bgr: np.ndarray,
        mask_u8: np.ndarray,
        p0: np.ndarray,
        n: np.ndarray,
        out_size: Tuple[int, int],
        margin: float
    ) -> np.ndarray:
        """旋轉、縮放並置中到畫布"""
        H, W = out_size
        
        # 計算旋轉角度（讓中線垂直）
        u = np.array([n[1], -n[0]], dtype=np.float64)
        u /= (np.linalg.norm(u) + 1e-12)
        if u[1] < 0:
            u = -u
        angle = np.arctan2(u[0], u[1])
        cos, sin = np.cos(angle), np.sin(angle)
        
        # 旋轉矩陣（繞 p0）
        R = np.array([
            [cos, -sin, (1 - cos) * p0[0] + sin * p0[1]],
            [sin,  cos, (1 - cos) * p0[1] - sin * p0[0]]
        ], dtype=np.float32)
        
        # 先旋轉遮罩以計算邊界框
        m_rot = cv2.warpAffine(
            mask_u8, R, (img_bgr.shape[1], img_bgr.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        ys, xs = np.where(m_rot > 0)
        if xs.size == 0 or ys.size == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)
        
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
        
        # 計算縮放比例
        Wfit = int(round(W * (1 - 2 * margin)))
        Hfit = int(round(H * (1 - 2 * margin)))
        Wfit = max(Wfit, 1)
        Hfit = max(Hfit, 1)
        
        s = min(Wfit / max(bw, 1), Hfit / max(bh, 1))
        
        # 組合最終變換矩陣
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        
        # 先旋轉，再縮放，最後平移到畫布中心
        R3 = np.vstack([R, [0, 0, 1]]).astype(np.float32)
        S3 = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float32)
        
        center_vec = np.array([cx, cy, 1.0], dtype=np.float32)
        center_rot = S3 @ (R3 @ center_vec)
        target = np.array([W / 2.0, H / 2.0, 1.0], dtype=np.float32)
        
        tx, ty = (target - center_rot)[:2]
        T3 = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
        
        M3 = T3 @ S3 @ R3
        M = M3[:2, :]
        
        # 應用變換
        result = cv2.warpAffine(
            img_bgr, M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return result

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        在Lab色彩空間的L通道進行自適應直方圖均衡化
        """
        # 確保影像格式正確
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
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