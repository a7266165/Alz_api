"""
人臉分析與認知評估API
支援多種壓縮格式，提供6QDS認知評估和人臉不對稱性分析
"""

import os
import cv2
import numpy as np
import pandas as pd
import zipfile
import tempfile
from typing import Dict, List, Optional
import mediapipe as mp
import json
import xgboost as xgb
import base64
from fastapi import FastAPI, File, UploadFile
import uvicorn
from pydantic import BaseModel

# 多格式壓縮檔支援
try:
    import py7zr
    HAS_7Z_SUPPORT = True
except ImportError:
    HAS_7Z_SUPPORT = False

try:
    import rarfile
    HAS_RAR_SUPPORT = True
except ImportError:
    HAS_RAR_SUPPORT = False


class AnalysisResponse(BaseModel):
    """API 回應模型"""
    success: bool
    error: Optional[str] = None
    q6ds_classification_result: Optional[float] = None
    asymmetry_classification_result: Optional[float] = None
    marked_figure: Optional[str] = None


class QuestionnaireData(BaseModel):
    """問卷資料模型"""
    age: int
    gender: int  # 0: 女性, 1: 男性
    education_years: int
    q1: int
    q2: int
    q3: int
    q4: int
    q5: int
    q6: int
    q7: int
    q8: int
    q9: int
    q10: int


class FaceAnalysisAPI:
    """人臉分析API類別"""
    
    def __init__(self, symmetry_csv_path: str = None, asymmetry_model_path: str = None, q6ds_model_path: str = None):
        # 初始化MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.symmetry_csv_path = symmetry_csv_path
        self.asymmetry_model_path = asymmetry_model_path
        self.q6ds_model_path = q6ds_model_path
        
        # 載入不對稱性XGBoost模型
        self.asymmetry_model = None
        if asymmetry_model_path and os.path.exists(asymmetry_model_path):
            try:
                self.asymmetry_model = xgb.Booster()
                self.asymmetry_model.load_model(asymmetry_model_path)
            except Exception as e:
                print(f"載入不對稱性XGBoost模型失敗: {e}")
        
        # 載入6QDS XGBoost模型
        self.q6ds_model = None
        if q6ds_model_path and os.path.exists(q6ds_model_path):
            try:
                self.q6ds_model = xgb.Booster()
                self.q6ds_model.load_model(q6ds_model_path)
            except Exception as e:
                print(f"載入6QDS XGBoost模型失敗: {e}")
        
        # 定義人臉中軸線
        self.FACEMESH_MID_LINE = [
            (10, 151), (151, 9), (9, 8), (8, 168), (168, 6),
            (6, 197), (197, 195), (195, 5), (5, 4), (4, 1),
            (1, 19), (19, 94), (94, 2),
        ]

    def analyze_from_archive_and_questionnaire(self, archive_file_path: str, questionnaire_data: QuestionnaireData, skip_face_selection: bool) -> Dict:
        """從壓縮檔和問卷資料分析"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # 解壓縮檔案
                extracted_dir = self._extract_archive_file(archive_file_path, temp_dir)
                
                # 分析人臉
                result = self._analyze_face_from_folder(extracted_dir, questionnaire_data, skip_face_selection)
                
                return result
                
            except Exception as e:
                return {
                    "success": False,
                    "error": f"分析失敗: {str(e)}",
                    "q6ds_classification_result": None,
                    "asymmetry_classification_result": None,
                    "marked_figure": None
                }

    def _extract_archive_file(self, archive_file_path: str, extract_to: str) -> str:
        """解壓縮檔案 - 支援 ZIP, 7Z, RAR 格式"""
        file_extension = os.path.splitext(archive_file_path)[1].lower()
        
        try:
            if file_extension == '.zip':
                with zipfile.ZipFile(archive_file_path, 'r') as archive_ref:
                    archive_ref.extractall(extract_to)
                    
            elif file_extension == '.7z':
                if not HAS_7Z_SUPPORT:
                    raise ValueError("需要安裝 py7zr 套件才能支援 7Z 格式：pip install py7zr")
                with py7zr.SevenZipFile(archive_file_path, mode='r') as archive_ref:
                    archive_ref.extractall(extract_to)
                    
            elif file_extension == '.rar':
                if not HAS_RAR_SUPPORT:
                    raise ValueError("需要安裝 rarfile 套件才能支援 RAR 格式：pip install rarfile")
                with rarfile.RarFile(archive_file_path, 'r') as archive_ref:
                    archive_ref.extractall(extract_to)
                    
            else:
                raise ValueError(f"不支援的壓縮格式：{file_extension}")
                
        except Exception as e:
            raise ValueError(f"解壓縮失敗：{str(e)}")
        
        # 找到包含圖片的資料夾
        for root, dirs, files in os.walk(extract_to):
            jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            if jpg_files:
                return root
        
        raise ValueError("壓縮檔中未找到圖片檔案")

    def _analyze_face_from_folder(self, folder_path: str, questionnaire_data: QuestionnaireData = None, skip_face_selection: bool = False) -> Dict:
        """從資料夾分析人臉"""
        try:
            # 步驟1: 決定是否選擇最正面的照片並轉正
            if not skip_face_selection:
                rotated_images = self._align_and_select_faces(folder_path)
            else:
                # 直接載入資料夾中所有影像，假設已經是正面
                rotated_images = self._load_images(folder_path)
            
            if not rotated_images:
                return {
                    "success": False,
                    "error": "未找到有效的人臉圖片",
                    "q6ds_classification_result": None,
                    "asymmetry_classification_result": None,
                    "marked_figure": None
                }
            
            # 步驟2: 提取正規化特徵點座標
            landmarks = self._extract_normalized_landmark_coordinates(rotated_images)
            
            if landmarks is None:
                return {
                    "success": False,
                    "error": "無法提取特徵點",
                    "q6ds_classification_result": None,
                    "asymmetry_classification_result": None,
                    "marked_figure": None
                }
            
            # 步驟3: 計算不對稱性指標和預測
            asymmetry_classification = None
            if self.symmetry_csv_path:
                symmetry_metrics = self._calculate_symmetry_metrics(landmarks)
                if symmetry_metrics and self.asymmetry_model:
                    asymmetry_classification = self._predict_asymmetry(symmetry_metrics)
            
            # 步驟4: 6QDS問卷分類預測
            q6ds_classification = None
            if questionnaire_data and self.q6ds_model:
                q6ds_classification = self._predict_6qds(questionnaire_data)
            
            # 步驟5: 生成標記圖片
            marked_figure = self._generate_marked_figure(rotated_images[0])
            
            return {
                "success": True,
                "error": None,
                "q6ds_classification_result": q6ds_classification,
                "asymmetry_classification_result": asymmetry_classification,
                "marked_figure": marked_figure
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"分析過程發生錯誤: {str(e)}",
                "q6ds_classification_result": None,
                "asymmetry_classification_result": None,
                "marked_figure": None
            }

    def _get_point(self, results, index: int, width: int, height: int) -> np.ndarray:
        """將 landmark 轉換為經過寬高縮放的二維點"""
        pt = results.multi_face_landmarks[0].landmark[index]
        return np.array([pt.x * width, pt.y * height])

    def _angle_between(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """計算兩向量之間的夾角（單位：度）"""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免數值誤差
        return np.degrees(np.arccos(cos_angle))

    def _calc_intermediate_angle_sum(self, results, height: int, width: int) -> float:
        """計算中間兩個夾角的總和"""
        p1, p2, p3, p4 = [self._get_point(results, i, width, height) for i in (10, 168, 4, 2)]
        angle1 = self._angle_between(p2 - p1, p3 - p2)
        angle2 = self._angle_between(p3 - p2, p4 - p3)
        return angle1 + angle2

    def _mid_line_angle_all_points(self, results, height: int, width: int) -> float:
        """計算人臉中軸線各段的角度，並回傳平均角度"""
        angles = []
        for i, j in self.FACEMESH_MID_LINE:
            p1, p2 = self._get_point(results, i, width, height), self._get_point(results, j, width, height)
            angles.append(np.degrees(np.arctan2(p2[0] - p1[0], p2[1] - p1[1])))
        return sum(angles) / len(angles)

    def _rotate_image(self, image: np.ndarray) -> np.ndarray:
        """根據人臉中軸角度，將圖片旋轉調整至正立"""
        height, width = image.shape[:2]
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return image
        
        angle = self._mid_line_angle_all_points(results, height, width)
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        return cv2.warpAffine(image, M, (width, height))

    def _align_and_select_faces(self, face_pic_folder: str) -> List[np.ndarray]:
        """選取夾角總和最小的10張圖片並旋轉"""
        angle_dict = {}
        
        # 支援多種圖片格式
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        for file in os.listdir(face_pic_folder):
            if not file.lower().endswith(valid_extensions):
                continue
                
            path = os.path.join(face_pic_folder, file)
            image = cv2.imread(path)
            if image is None:
                continue
                
            height, width = image.shape[:2]
            results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                continue
                
            angle_dict[file] = self._calc_intermediate_angle_sum(results, height, width)

        if not angle_dict:
            return []

        # 選出夾角總和最小的10張圖（代表正對相機）
        selected_files = sorted(angle_dict, key=lambda x: angle_dict[x])[:10]

        # 將選取的圖片轉正
        rotated_images = []
        for file in selected_files:
            image = cv2.imread(os.path.join(face_pic_folder, file))
            if image is not None:
                rotated_image = self._rotate_image(image)
                rotated_images.append(rotated_image)
                
        return rotated_images

    def _load_images(self, folder_path: str) -> List:
        """讀取資料夾中所有圖片檔案並回傳影像列表"""
        images = []
        for fname in os.listdir(folder_path):
            path = os.path.join(folder_path, fname)
            try:
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
            except Exception:
                continue
        return images

    def _extract_normalized_landmark_coordinates(self, rotated_face_images: List[np.ndarray]) -> Optional[np.ndarray]:
        """提取正規化的特徵點座標"""
        landmarks_all = []

        for image in rotated_face_images:
            height, width = image.shape[:2]
            results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                continue
                
            landmarks = results.multi_face_landmarks[0].landmark

            # 取出臉部最左邊和最上面的點
            left_x = landmarks[234].x * width
            top_y = landmarks[10].y * height
            
            pts = []
            for i in range(468):
                pt = landmarks[i]
                x = (pt.x * width) - left_x
                y = (pt.y * height) - top_y
                pts.append((x, y))
            pts = np.array(pts)

            # 依據臉部最右側點固定臉的寬度為500
            right_x = pts[454][0]
            if right_x <= 0:
                continue
                
            scale_factor = 500 / right_x
            pts = pts * scale_factor
            pts = pts.T  # 轉置成 (2, 468)
            landmarks_all.append(pts)

        if not landmarks_all:
            return None

        landmarks_all = np.array(landmarks_all)
        # 取平均
        landmarks_all = np.mean(landmarks_all, axis=0).reshape(1, 2, 468)

        # 新增z軸座標（設為0）
        z_coords = np.zeros((1, 1, 468))
        landmarks_all = np.concatenate([landmarks_all, z_coords], axis=1)

        return landmarks_all

    def _parse_idxs(self, s: str) -> List[int]:
        """解析索引字串"""
        return list(map(int, s.split(",")))

    def _line_len(self, x: Dict, y: Dict, idxs: List[int]) -> float:
        """計算線段長度"""
        i, j = idxs
        return np.hypot(x[i] - x[j], y[i] - y[j])

    def _tri_area(self, x: Dict, y: Dict, idxs: List[int]) -> float:
        """計算三角形面積"""
        i, j, k = idxs
        x1, y1 = x[i], y[i]
        x2, y2 = x[j], y[j]
        x3, y3 = x[k], y[k]
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

    def _calculate_symmetry_metrics(self, landmarks: np.ndarray) -> Optional[Dict]:
        """計算對稱性指標"""
        if not self.symmetry_csv_path or not os.path.exists(self.symmetry_csv_path):
            return None
            
        try:
            # 讀取對稱配對資料
            df_pairs = pd.read_csv(self.symmetry_csv_path)

            # 拆分三種類型的配對
            df_pts = df_pairs[df_pairs["pair_type"].str.startswith("point")].copy()
            df_lines = df_pairs[df_pairs["pair_type"].str.startswith("line")].copy()
            df_tri = df_pairs[df_pairs["pair_type"].str.startswith("triangle")].copy()

            # 解析線段和三角形的索引
            if not df_lines.empty:
                df_lines["left_idx"] = df_lines["left"].apply(self._parse_idxs)
                df_lines["right_idx"] = df_lines["right"].apply(self._parse_idxs)
            if not df_tri.empty:
                df_tri["left_idx"] = df_tri["left"].apply(self._parse_idxs)
                df_tri["right_idx"] = df_tri["right"].apply(self._parse_idxs)

            # 從landmarks提取x, y座標
            x_coords = landmarks[0, 0, :]
            y_coords = landmarks[0, 1, :]

            # 建立座標字典
            x = {i: x_coords[i] for i in range(468)}
            y = {i: y_coords[i] for i in range(468)}

            # 計算基準線
            baseline_x = abs(x[234] - x[454])
            baseline_y = abs(y[10] - y[152])

            # 避免除以零
            if baseline_x == 0:
                baseline_x = 1
            if baseline_y == 0:
                baseline_y = 1

            # 初始化累加器
            total_pt_x = 0.0
            total_pt_y = 0.0
            total_line = 0.0
            total_tri = 0.0

            # 計算點對稱 X 差值
            for _, r in df_pts.iterrows():
                idx_l = int(r["left"])
                idx_r = int(r["right"])
                if idx_l < 468 and idx_r < 468:
                    diff_x = abs(abs(x[idx_l] - 250) - abs(x[idx_r] - 250)) / baseline_x
                    total_pt_x += diff_x

            # 計算點對稱 Y 差值
            for _, r in df_pts.iterrows():
                idx_l = int(r["left"])
                idx_r = int(r["right"])
                if idx_l < 468 and idx_r < 468:
                    diff_y = abs(y[idx_l] - y[idx_r]) / baseline_y
                    total_pt_y += diff_y

            # 計算線段對稱差值
            for _, r in df_lines.iterrows():
                if all(idx < 468 for idx in r["left_idx"]) and all(idx < 468 for idx in r["right_idx"]):
                    ld = self._line_len(x, y, r["left_idx"])
                    rd = self._line_len(x, y, r["right_idx"])
                    if ld + rd > 0:
                        diff_line = abs(ld - rd) / (ld + rd)
                        total_line += diff_line

            # 計算三角形面積對稱差值
            for _, r in df_tri.iterrows():
                if all(idx < 468 for idx in r["left_idx"]) and all(idx < 468 for idx in r["right_idx"]):
                    la = self._tri_area(x, y, r["left_idx"])
                    ra = self._tri_area(x, y, r["right_idx"])
                    if la + ra > 0:
                        diff_tri = abs(la - ra) / (la + ra)
                        total_tri += diff_tri

            return {
                "sum_point_x_diff": float(total_pt_x),
                "sum_point_y_diff": float(total_pt_y),
                "sum_line_diff": float(total_line),
                "sum_triangle_area_diff": float(total_tri),
            }

        except Exception as e:
            print(f"計算對稱性指標錯誤: {str(e)}")
            return None

    def _predict_asymmetry(self, symmetry_metrics: Dict) -> Optional[float]:
        """使用XGBoost模型預測不對稱性分類結果"""
        if not self.asymmetry_model or not symmetry_metrics:
            return None
            
        try:
            # 準備輸入特徵
            features = [
                symmetry_metrics["sum_point_x_diff"],
                symmetry_metrics["sum_point_y_diff"], 
                symmetry_metrics["sum_line_diff"],
                symmetry_metrics["sum_triangle_area_diff"]
            ]
            
            # 轉換為XGBoost DMatrix格式
            dmatrix = xgb.DMatrix([features])
            
            # 預測
            prediction = self.asymmetry_model.predict(dmatrix)
            
            # 回傳預測結果（通常是機率值或分類結果）
            return float(prediction[0])
            
        except Exception as e:
            print(f"不對稱性XGBoost預測錯誤: {str(e)}")
            return None

    def _predict_6qds(self, questionnaire_data: QuestionnaireData) -> Optional[float]:
        """使用XGBoost模型預測6QDS分類結果"""
        if not self.q6ds_model:
            return None
            
        try:
            # 準備輸入特徵：年紀 性別 教育程度yr q1 q2 q3 q4 q5 q6 q7 q8 q9 q10
            features = [
                questionnaire_data.age,
                questionnaire_data.gender,
                questionnaire_data.education_years,
                questionnaire_data.q1,
                questionnaire_data.q2,
                questionnaire_data.q3,
                questionnaire_data.q4,
                questionnaire_data.q5,
                questionnaire_data.q6,
                questionnaire_data.q7,
                questionnaire_data.q8,
                questionnaire_data.q9,
                questionnaire_data.q10
            ]
            
            # 定義特徵名稱（與訓練時相同）
            feature_names = [
                '年紀', '性別', '教育程度yr', 
                'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10'
            ]
            
            # 轉換為XGBoost DMatrix格式，包含特徵名稱
            dmatrix = xgb.DMatrix([features], feature_names=feature_names)
            
            # 預測
            prediction = self.q6ds_model.predict(dmatrix)
            
            # 回傳預測結果
            return float(prediction[0])
            
        except Exception as e:
            print(f"6QDS XGBoost預測錯誤: {str(e)}")
            return None

    def _generate_marked_figure(self, image: np.ndarray) -> Optional[str]:
        """生成帶有特徵點標記的圖片，並轉換為base64字串"""
        try:
            marked_image = self._get_face_with_landmarks_from_image(image)
            if marked_image is None:
                return None
                
            # 將圖片編碼為JPEG格式
            _, buffer = cv2.imencode('.jpg', marked_image)
            
            # 轉換為base64字串
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return img_base64
            
        except Exception as e:
            print(f"生成標記圖片錯誤: {str(e)}")
            return None

    def _get_face_with_landmarks_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """從已經轉正的圖片獲取帶有特徵點標記的人臉圖片（只截取人臉部分）"""
        height, width = image.shape[:2]
        
        # 檢測特徵點
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return image
        
        # 獲取原始特徵點位置
        original_landmarks = results.multi_face_landmarks[0].landmark
        
        # 根據特定的特徵點來確定裁剪範圍
        left_x = original_landmarks[234].x * width
        right_x = original_landmarks[454].x * width
        top_y = original_landmarks[10].y * height
        bottom_y = original_landmarks[152].y * height
        
        # 添加一些邊距
        margin = 5
        left = max(0, int(left_x - margin))
        right = min(width, int(right_x + margin))
        top = max(0, int(top_y - margin))
        bottom = min(height, int(bottom_y + margin))
        
        # 裁剪圖片
        cropped_image = image[top:bottom, left:right]
        cropped_height, cropped_width = cropped_image.shape[:2]
        
        # 計算縮放比例，使臉寬為500像素
        face_width = right_x - left_x
        if face_width <= 0:
            return image
            
        scale_factor = 500 / face_width
        
        # 調整裁剪後圖片的大小
        new_width = int(cropped_width * scale_factor)
        new_height = int(cropped_height * scale_factor)
        resized_image = cv2.resize(cropped_image, (new_width, new_height))
        
        # 在調整大小後的圖片上繪製特徵點和線段
        image_with_landmarks = resized_image.copy()
        
        # 繪製特徵點
        for i in range(468):
            pt = original_landmarks[i]
            x = int((pt.x * width - left) * scale_factor)
            y = int((pt.y * height - top) * scale_factor)
            
            if 0 <= x < new_width and 0 <= y < new_height:
                cv2.circle(image_with_landmarks, (x, y), 2, (0, 0, 255), -1)  # 紅色點
        
        # 繪製中線
        mid_x = new_width // 2
        cv2.line(image_with_landmarks, (mid_x, 0), (mid_x, new_height), (0, 255, 255), 2)  # 黃色線
        
        # 如果有對稱性CSV，繪製線段
        if self.symmetry_csv_path and os.path.exists(self.symmetry_csv_path):
            try:
                df_pairs = pd.read_csv(self.symmetry_csv_path, encoding='utf-8-sig')
                df_lines = df_pairs[df_pairs['pair_type'].str.startswith('line')]
                
                for _, row_pair in df_lines.iterrows():
                    for side in ('left', 'right'):
                        try:
                            idx0, idx1 = map(int, row_pair[side].split(','))
                            if idx0 < 468 and idx1 < 468:
                                pt0 = original_landmarks[idx0]
                                pt1 = original_landmarks[idx1]
                                
                                x0 = int((pt0.x * width - left) * scale_factor)
                                y0 = int((pt0.y * height - top) * scale_factor)
                                x1 = int((pt1.x * width - left) * scale_factor)
                                y1 = int((pt1.y * height - top) * scale_factor)
                                
                                if (0 <= x0 < new_width and 0 <= y0 < new_height and 
                                    0 <= x1 < new_width and 0 <= y1 < new_height):
                                    cv2.line(image_with_landmarks, (x0, y0), (x1, y1), (0, 255, 0), 1)  # 綠色線
                        except:
                            continue
            except Exception as e:
                print(f"繪製線段時發生錯誤: {e}")
        
        return image_with_landmarks


class FaceAnalysisFastAPI:
    """FastAPI 服務封裝"""
    
    def __init__(self, symmetry_csv_path: str = None, asymmetry_model_path: str = None, q6ds_model_path: str = None):
        self.app = FastAPI(
            title="人臉分析與認知評估API",
            description="上傳人臉相片壓縮檔和問卷資料，回傳6QDS認知評估、不對稱性分類結果和標記圖片",
            version="1.0.0"
        )
        self.analyzer = FaceAnalysisAPI(symmetry_csv_path, asymmetry_model_path, q6ds_model_path)
        self._setup_routes()
    
    def _setup_routes(self):
        """設定API路由"""
        
        @self.app.post("/analyze_1200_pics", response_model=AnalysisResponse, summary="分析人臉不對稱性和6QDS認知評估(1200張未篩選相片)")
        async def analyze_face_and_questionnaire(
            file: UploadFile = File(...),
            age: int = 65,
            gender: int = 1,  # 0: 女性, 1: 男性
            education_years: int = 6,
            q1: int = 0,
            q2: int = 0,
            q3: int = 2,
            q4: int = 1,
            q5: int = 1,
            q6: int = 1,
            q7: int = 1,
            q8: int = 1,
            q9: int = 1,
            q10: int = 1
        ):
            """
            分析人臉不對稱性和6QDS認知評估的API端點
            
            - **file**: 包含未篩選人臉相片的壓縮檔（支援 ZIP, 7Z, RAR 格式）
            - **age**: 年齡
            - **gender**: 性別 (0: 女性, 1: 男性)
            - **education_years**: 教育年數
            - **q1-q10**: 問卷題目1-10的答案
            
            回傳:
            - **success**: 分析是否成功
            - **error**: 錯誤訊息（如有）
            - **q6ds_classification_result**: 6QDS認知評估XGBoost模型預測結果
            - **asymmetry_classification_result**: 不對稱性XGBoost模型預測結果
            - **marked_figure**: base64編碼的標記圖片
            """
            skip_face_selection = False
            # 檢查檔案格式
            supported_formats = self._get_supported_formats()
            if not any(file.filename.lower().endswith(fmt) for fmt in supported_formats):
                available_formats = ', '.join(supported_formats)
                return AnalysisResponse(
                    success=False,
                    error=f"支援的格式：{available_formats}。如需其他格式請安裝對應套件。",
                    q6ds_classification_result=None,
                    asymmetry_classification_result=None,
                    marked_figure=None
                )
            
            # 檢查檔案大小
            content = await file.read()
            if len(content) > 50 * 1024 * 1024:  # 50MB限制
                return AnalysisResponse(
                    success=False,
                    error="檔案大小超過50MB限制",
                    q6ds_classification_result=None,
                    asymmetry_classification_result=None,
                    marked_figure=None
                )
            
            # 建立問卷資料
            questionnaire_data = QuestionnaireData(
                age=age,
                gender=gender,
                education_years=education_years,
                q1=q1, q2=q2, q3=q3, q4=q4, q5=q5,
                q6=q6, q7=q7, q8=q8, q9=q9, q10=q10
            )
            
            # 處理檔案
            return await self._process_uploaded_file(file.filename, content, questionnaire_data, skip_face_selection)
        
        @self.app.post("/analyze_n_pics", response_model=AnalysisResponse, summary="分析人臉不對稱性和6QDS認知評估(n張已篩選相片)")
        async def analyze_face_and_questionnaire(
            file: UploadFile = File(...),
            age: int = 65,
            gender: int = 1,  # 0: 女性, 1: 男性
            education_years: int = 6,
            q1: int = 0,
            q2: int = 0,
            q3: int = 2,
            q4: int = 1,
            q5: int = 1,
            q6: int = 1,
            q7: int = 1,
            q8: int = 1,
            q9: int = 1,
            q10: int = 1
        ):
            """
            分析人臉不對稱性和6QDS認知評估的API端點
            
            - **file**: 包含已篩選人臉相片的壓縮檔（支援 ZIP, 7Z, RAR 格式）
            - **age**: 年齡
            - **gender**: 性別 (0: 女性, 1: 男性)
            - **education_years**: 教育年數
            - **q1-q10**: 問卷題目1-10的答案
            
            回傳:
            - **success**: 分析是否成功
            - **error**: 錯誤訊息（如有）
            - **q6ds_classification_result**: 6QDS認知評估XGBoost模型預測結果
            - **asymmetry_classification_result**: 不對稱性XGBoost模型預測結果
            - **marked_figure**: base64編碼的標記圖片
            """
            skip_face_selection = True
            # 檢查檔案格式
            supported_formats = self._get_supported_formats()
            if not any(file.filename.lower().endswith(fmt) for fmt in supported_formats):
                available_formats = ', '.join(supported_formats)
                return AnalysisResponse(
                    success=False,
                    error=f"支援的格式：{available_formats}。如需其他格式請安裝對應套件。",
                    q6ds_classification_result=None,
                    asymmetry_classification_result=None,
                    marked_figure=None
                )
            
            # 檢查檔案大小
            content = await file.read()
            if len(content) > 50 * 1024 * 1024:  # 50MB限制
                return AnalysisResponse(
                    success=False,
                    error="檔案大小超過50MB限制",
                    q6ds_classification_result=None,
                    asymmetry_classification_result=None,
                    marked_figure=None
                )
            
            # 建立問卷資料
            questionnaire_data = QuestionnaireData(
                age=age,
                gender=gender,
                education_years=education_years,
                q1=q1, q2=q2, q3=q3, q4=q4, q5=q5,
                q6=q6, q7=q7, q8=q8, q9=q9, q10=q10
            )
            
            # 處理檔案
            return await self._process_uploaded_file(file.filename, content, questionnaire_data, skip_face_selection)


        @self.app.get("/health", summary="健康檢查")
        async def health_check():
            """健康檢查端點"""
            return {
                "status": "healthy",
                "service": "Face Analysis & Cognitive Assessment API",
                "version": "1.0.0",
                "supported_formats": self._get_supported_formats()
            }
        
        @self.app.get("/", summary="API資訊")
        async def root():
            """API根路徑，回傳基本資訊"""
            return {
                "message": "人臉分析與認知評估API",
                "docs": "/docs",
                "health": "/health",
                "analyze": "/analyze (POST with archive file + questionnaire data)",
                "supported_formats": self._get_supported_formats(),
                "installation_notes": self._get_installation_notes(),
                "questionnaire_fields": {
                    "age": "年齡",
                    "gender": "性別 (0: 女性, 1: 男性)",
                    "education_years": "教育年數",
                    "q1-q10": "問卷題目1-10的答案"
                }
            }
    
    def _get_supported_formats(self) -> List[str]:
        """取得支援的檔案格式"""
        formats = ['.zip']
        if HAS_7Z_SUPPORT:
            formats.append('.7z')
        if HAS_RAR_SUPPORT:
            formats.append('.rar')
        return formats
    
    def _get_installation_notes(self) -> Dict[str, str]:
        """取得安裝說明"""
        return {
            "7z_support": "已安裝" if HAS_7Z_SUPPORT else "pip install py7zr",
            "rar_support": "已安裝" if HAS_RAR_SUPPORT else "pip install rarfile"
        }
    
    async def _process_uploaded_file(self, filename: str, content: bytes, questionnaire_data: QuestionnaireData, skip_face_selection: bool) -> AnalysisResponse:
        """處理上傳的檔案和問卷資料"""
        # 確定檔案副檔名
        file_ext = self._get_file_extension(filename)
        
        # 創建臨時檔案
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        temp_file_path = temp_file.name
        
        try:
            # 寫入檔案內容
            temp_file.write(content)
            temp_file.flush()
            temp_file.close()
            
            # 分析人臉和問卷
            result = self.analyzer.analyze_from_archive_and_questionnaire(temp_file_path, questionnaire_data, skip_face_selection)
            return AnalysisResponse(**result)
            
        except Exception as e:
            return AnalysisResponse(
                success=False,
                error=f"處理檔案時發生錯誤: {str(e)}",
                q6ds_classification_result=None,
                asymmetry_classification_result=None,
                marked_figure=None
            )
        finally:
            # 清理臨時檔案
            self._cleanup_temp_file(temp_file_path)
    
    def _get_file_extension(self, filename: str) -> str:
        """取得檔案副檔名"""
        if filename.lower().endswith('.7z'):
            return '.7z'
        elif filename.lower().endswith('.rar'):
            return '.rar'
        else:
            return '.zip'
    
    def _cleanup_temp_file(self, file_path: str):
        """清理臨時檔案"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"清理臨時檔案時發生錯誤: {e}")


# 應用程式配置和啟動
def get_file_path(relative_path: str) -> str:
    """取得檔案的絕對路徑"""
    if os.path.isabs(relative_path):
        return relative_path
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)

def print_startup_info():
    """顯示啟動資訊"""
    print("🚀 啟動人臉分析與認知評估FastAPI服務...")
    print("\n📋 API端點:")
    print("  POST /analyze - 上傳壓縮檔案和問卷資料進行綜合分析")
    print("  GET /health - 健康檢查")
    print("  GET / - API資訊")
    print("  GET /docs - Swagger文檔")
    print("  GET /redoc - ReDoc文檔")
    
    print("\n📊 回傳格式:")
    print("  - success: 布林值，是否成功")
    print("  - error: 錯誤訊息（如有）")
    print("  - q6ds_classification_result: 6QDS認知評估XGBoost模型預測結果")
    print("  - asymmetry_classification_result: 不對稱性XGBoost模型預測結果")
    print("  - marked_figure: base64編碼的標記圖片")
    
    print("\n📝 問卷輸入:")
    print("  - age: 年齡")
    print("  - gender: 性別 (0: 女性, 1: 男性)")
    print("  - education_years: 教育年數")
    print("  - q1-q10: 問卷題目1-10的答案")
    
    print("\n📁 支援格式:")
    formats = ['.zip']
    if HAS_7Z_SUPPORT:
        formats.append('.7z')
    if HAS_RAR_SUPPORT:
        formats.append('.rar')
    print(f"  {', '.join(formats)}")
    
    print("\n🌐 服務將在 http://localhost:8000 啟動")
    if not HAS_7Z_SUPPORT:
        print("💡 提示: 安裝 py7zr 以支援 7Z 格式")
    if not HAS_RAR_SUPPORT:
        print("💡 提示: 安裝 rarfile 以支援 RAR 格式")

# 直接在模組層級創建app實例
print_startup_info()

# 設定檔案路徑
symmetry_csv_path = get_file_path("./data/symmetry_all_pairs.csv")
asymmetry_model_path = get_file_path("./data/xgb_face_asym_model.json")
q6ds_model_path = get_file_path("./data/xgb_6qds_model.json")

# 檢查檔案存在性
if symmetry_csv_path and not os.path.exists(symmetry_csv_path):
    print(f"警告: 找不到對稱性CSV檔案: {symmetry_csv_path}")
if asymmetry_model_path and not os.path.exists(asymmetry_model_path):
    print(f"警告: 找不到不對稱性XGBoost模型檔案: {asymmetry_model_path}")
if q6ds_model_path and not os.path.exists(q6ds_model_path):
    print(f"警告: 找不到6QDS XGBoost模型檔案: {q6ds_model_path}")

# 直接創建全域app實例
api_server = FaceAnalysisFastAPI(symmetry_csv_path, asymmetry_model_path, q6ds_model_path)
app = api_server.app

# 主程式入口
if __name__ == "__main__":
    # 啟動服務
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)