import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="SelarasSehat - Ergonomic Assessment",
    page_icon="🏥",
    layout="wide"
)

# Translations
TRANSLATIONS = {
    'en': {
        'title': '🏥 SelarasSehat - Ergonomic Assessment App',
        'subtitle': 'Computer Vision-Based RULA Scoring using MediaPipe',
        'lang_label': 'Language / Bahasa',
        'upload_label': 'Upload Video',
        'upload_help': 'Upload a video (max 30 seconds recommended) showing work activity from side view',
        'processing': 'Processing video... Please wait',
        'analysis_complete': 'Analysis Complete!',
        'results_title': 'RULA Assessment Results',
        'avg_score': 'Average RULA Score',
        'max_score': 'Maximum RULA Score',
        'min_score': 'Minimum RULA Score',
        'risk_level': 'Risk Level',
        'recommendation': 'Recommendation',
        'score_timeline': 'RULA Score Timeline',
        'annotated_video': 'Annotated Video with Pose Detection',
        'download_csv': 'Download Detailed Results (CSV)',
        'download_video': 'Download Annotated Video',
        'about_title': 'About SelarasSehat',
        'about_text': '''
        SelarasSehat is an automated ergonomic assessment tool that uses computer vision 
        to calculate RULA (Rapid Upper Limb Assessment) scores from video recordings.
        
        **How it works:**
        1. Upload a video of work activity
        2. MediaPipe detects body pose landmarks
        3. Joint angles are calculated automatically
        4. RULA score is computed for each frame
        5. Get comprehensive assessment and recommendations
        
        **Best practices for video recording:**
        - Record from side view (90° angle) for best accuracy
        - Ensure good lighting and clear visibility
        - Keep camera stable
        - 15-30 seconds duration is optimal
        ''',
        'risk_levels': {
            1: 'Acceptable - No action required',
            2: 'Low Risk - Further investigation, change may be needed',
            3: 'Medium Risk - Investigation and changes required soon',
            4: 'High Risk - Investigation and changes required immediately'
        },
        'error_no_pose': 'Error: Could not detect pose in video. Please ensure person is clearly visible.',
        'error_processing': 'Error processing video. Please try again.',
        'adjustments_title': 'Manual Adjustments',
        'adjustments_help': 'Check boxes that apply to adjust RULA score',
        'upper_arm_raised': 'Shoulder raised',
        'upper_arm_abducted': 'Arm abducted (>20° away from body)',
        'lower_arm_midline': 'Working across midline or out to side',
        'wrist_deviated': 'Wrist deviated (radial/ulnar)',
        'neck_twisted': 'Neck twisted',
        'neck_bent': 'Neck side bent',
        'trunk_twisted': 'Trunk twisted',
        'trunk_bent': 'Trunk side bent',
        'wrist_twist_label': 'Wrist Twist',
        'wrist_twist_mid': 'Mid-range (default)',
        'wrist_twist_extreme': 'At or near end of range',
        'legs_label': 'Legs/Feet',
        'legs_supported': 'Supported and balanced',
        'legs_not_supported': 'Not supported',
        'muscle_label': 'Muscle Use',
        'muscle_static': 'Static (held >1 min) or repeated (>4x/min)',
        'force_label': 'Force/Load',
        'force_none': 'None or <2 kg intermittent',
        'force_light': '2-10 kg intermittent',
        'force_heavy': '2-10 kg static/repeated, or >10 kg intermittent',
        'force_shock': 'Shock or rapid force increase',
    },
    'id': {
        'title': '🏥 SelarasSehat - Aplikasi Penilaian Ergonomis',
        'subtitle': 'Penilaian RULA Berbasis Computer Vision menggunakan MediaPipe',
        'lang_label': 'Language / Bahasa',
        'upload_label': 'Unggah Video',
        'upload_help': 'Unggah video (maksimal 30 detik direkomendasikan) yang menunjukkan aktivitas kerja dari tampak samping',
        'processing': 'Memproses video... Mohon tunggu',
        'analysis_complete': 'Analisis Selesai!',
        'results_title': 'Hasil Penilaian RULA',
        'avg_score': 'Skor RULA Rata-rata',
        'max_score': 'Skor RULA Maksimum',
        'min_score': 'Skor RULA Minimum',
        'risk_level': 'Tingkat Risiko',
        'recommendation': 'Rekomendasi',
        'score_timeline': 'Timeline Skor RULA',
        'annotated_video': 'Video Teranotasi dengan Deteksi Pose',
        'download_csv': 'Unduh Hasil Lengkap (CSV)',
        'download_video': 'Unduh Video Teranotasi',
        'about_title': 'Tentang SelarasSehat',
        'about_text': '''
        SelarasSehat adalah alat penilaian ergonomis otomatis yang menggunakan computer vision 
        untuk menghitung skor RULA (Rapid Upper Limb Assessment) dari rekaman video.
        
        **Cara kerja:**
        1. Unggah video aktivitas kerja
        2. MediaPipe mendeteksi landmark pose tubuh
        3. Sudut sendi dihitung secara otomatis
        4. Skor RULA dihitung untuk setiap frame
        5. Dapatkan penilaian komprehensif dan rekomendasi
        
        **Praktik terbaik untuk perekaman video:**
        - Rekam dari tampak samping (sudut 90°) untuk akurasi terbaik
        - Pastikan pencahayaan baik dan visibilitas jelas
        - Jaga kamera tetap stabil
        - Durasi 15-30 detik adalah optimal
        ''',
        'risk_levels': {
            1: 'Dapat Diterima - Tidak perlu tindakan',
            2: 'Risiko Rendah - Perlu investigasi lebih lanjut, perubahan mungkin diperlukan',
            3: 'Risiko Sedang - Investigasi dan perubahan diperlukan segera',
            4: 'Risiko Tinggi - Investigasi dan perubahan diperlukan dengan segera'
        },
        'error_no_pose': 'Error: Tidak dapat mendeteksi pose dalam video. Pastikan orang terlihat dengan jelas.',
        'error_processing': 'Error memproses video. Silakan coba lagi.',
        'adjustments_title': 'Penyesuaian Manual',
        'adjustments_help': 'Centang kotak yang sesuai untuk menyesuaikan skor RULA',
        'upper_arm_raised': 'Bahu terangkat',
        'upper_arm_abducted': 'Lengan abduksi (>20° dari tubuh)',
        'lower_arm_midline': 'Bekerja melintasi garis tengah atau ke samping',
        'wrist_deviated': 'Pergelangan tangan menyimpang (radial/ulnar)',
        'neck_twisted': 'Leher berputar',
        'neck_bent': 'Leher miring ke samping',
        'trunk_twisted': 'Batang tubuh berputar',
        'trunk_bent': 'Batang tubuh miring',
        'wrist_twist_label': 'Putaran Pergelangan Tangan',
        'wrist_twist_mid': 'Rentang tengah (default)',
        'wrist_twist_extreme': 'Di atau dekat akhir rentang',
        'legs_label': 'Kaki/Telapak Kaki',
        'legs_supported': 'Didukung dan seimbang',
        'legs_not_supported': 'Tidak didukung',
        'muscle_label': 'Penggunaan Otot',
        'muscle_static': 'Statis (>1 menit) atau berulang (>4x/menit)',
        'force_label': 'Gaya/Beban',
        'force_none': 'Tidak ada atau <2 kg intermiten',
        'force_light': '2-10 kg intermiten',
        'force_heavy': '2-10 kg statis/berulang, atau >10 kg intermiten',
        'force_shock': 'Kejutan atau peningkatan gaya cepat',
    }
}

class RULACalculator:
    """Calculate RULA scores from MediaPipe pose landmarks"""
    
    @staticmethod
    def calculate_angle(point1, point2, point3):
        """Calculate angle between three points"""
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle
    
    @staticmethod
    def detect_shoulder_raised(landmarks):
        """Detect if shoulder is raised (shoulder higher than neutral)"""
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Calculate shoulder-hip distance (normalized)
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        
        # If shoulder is elevated (smaller y value = higher on screen)
        shoulder_hip_ratio = abs(shoulder_y - hip_y)
        
        # Shoulder raised if ratio is smaller than normal (shoulder elevated)
        return shoulder_hip_ratio < 0.25  # Threshold for raised shoulder
    
    @staticmethod
    def detect_arm_abducted(landmarks):
        """Detect if arm is abducted (>20° away from body)"""
        right_shoulder = landmarks[12]
        right_elbow = landmarks[14]
        right_hip = landmarks[24]
        
        # Calculate angle between shoulder-elbow and vertical
        vertical_point = [right_shoulder.x, right_shoulder.y - 0.1, right_shoulder.z]
        
        # Vector from shoulder to elbow
        elbow_vector_x = abs(right_elbow.x - right_shoulder.x)
        
        # If elbow is far from body horizontally (abducted)
        return elbow_vector_x > 0.15  # Threshold for abduction
    
    @staticmethod
    def detect_working_across_midline(landmarks):
        """Detect if arm crosses body midline"""
        nose = landmarks[0]
        right_shoulder = landmarks[12]
        right_wrist = landmarks[16]
        
        # Body midline is at nose x-coordinate
        midline_x = nose.x
        shoulder_x = right_shoulder.x
        wrist_x = right_wrist.x
        
        # If right arm crosses to left side of body
        if shoulder_x > midline_x and wrist_x < midline_x:
            return True
        # If working out to extreme side
        if abs(wrist_x - shoulder_x) > 0.3:
            return True
        
        return False
    
    @staticmethod
    def detect_wrist_deviation(landmarks):
        """Detect wrist radial/ulnar deviation"""
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]
        
        # Simple detection: if wrist is offset laterally from forearm line
        elbow_wrist_x_diff = abs(right_elbow.x - right_wrist.x)
        elbow_wrist_y_diff = abs(right_elbow.y - right_wrist.y)
        
        if elbow_wrist_y_diff > 0:
            deviation_ratio = elbow_wrist_x_diff / (elbow_wrist_y_diff + 1e-6)
            return deviation_ratio > 0.3  # Threshold for deviation
        
        return False
    
    @staticmethod
    def detect_neck_twisted(landmarks):
        """Detect if neck is twisted (rotation)"""
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        # Calculate shoulder midpoint
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        
        # If nose is significantly offset from shoulder midpoint
        nose_offset = abs(nose.x - shoulder_mid_x)
        
        return nose_offset > 0.05  # Threshold for twist
    
    @staticmethod
    def detect_neck_side_bent(landmarks):
        """Detect if neck is bent to side"""
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        # Check if one shoulder is significantly higher than other
        shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)
        
        return shoulder_tilt > 0.08  # Threshold for side bend
    
    @staticmethod
    def detect_trunk_twisted(landmarks):
        """Detect if trunk is twisted"""
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Calculate shoulder and hip orientations
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        hip_width = abs(left_hip.x - right_hip.x)
        
        # If widths differ significantly, trunk is twisted
        width_ratio = abs(shoulder_width - hip_width) / (hip_width + 1e-6)
        
        return width_ratio > 0.3  # Threshold for twist
    
    @staticmethod
    def detect_trunk_side_bent(landmarks):
        """Detect if trunk is bent to side"""
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Check alignment between shoulders and hips
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        hip_mid_x = (left_hip.x + right_hip.x) / 2
        
        lateral_shift = abs(shoulder_mid_x - hip_mid_x)
        
        return lateral_shift > 0.08  # Threshold for side bend
    
    @staticmethod
    def get_upper_arm_score(angle, raised=False, abducted=False):
        """Calculate upper arm RULA score"""
        if angle < 20:
            score = 1
        elif angle <= 45:
            score = 2
        elif angle <= 90:
            score = 3
        else:
            score = 4
        
        if raised:
            score += 1
        if abducted:
            score += 1
            
        return score
    
    @staticmethod
    def recalculate_rula(upper_arm_angle, lower_arm_angle, wrist_angle, neck_angle, trunk_angle,
                         upper_arm_raised, upper_arm_abducted, lower_arm_midline, wrist_deviated,
                         neck_twisted, neck_bent, trunk_twisted, trunk_bent, wrist_twist, legs_score,
                         muscle_use, force_load):
        """Recalculate RULA with manual adjustments"""
        # Get component scores with adjustments
        upper_arm_score = RULACalculator.get_upper_arm_score(upper_arm_angle, upper_arm_raised, upper_arm_abducted)
        lower_arm_score = RULACalculator.get_lower_arm_score(lower_arm_angle, lower_arm_midline)
        wrist_score = RULACalculator.get_wrist_score(wrist_angle, wrist_deviated)
        
        neck_score = RULACalculator.get_neck_score(neck_angle, neck_twisted, neck_bent)
        trunk_score = RULACalculator.get_trunk_score(trunk_angle, trunk_twisted, trunk_bent)
        
        # Get posture scores
        score_a = RULACalculator.get_posture_score_a(upper_arm_score, lower_arm_score, wrist_score, wrist_twist)
        score_b = RULACalculator.get_posture_score_b(neck_score, trunk_score, legs_score)
        
        # Final RULA score
        final_score = RULACalculator.get_final_score(score_a, score_b, muscle_use, force_load)
        
        return final_score, score_a, score_b
    
    @staticmethod
    def get_lower_arm_score(angle, working_across_midline=False):
        """Calculate lower arm RULA score"""
        if 60 <= angle <= 100:
            score = 1
        else:
            score = 2
        
        if working_across_midline:
            score += 1
            
        return score
    
    @staticmethod
    def get_wrist_score(angle, deviated=False):
        """Calculate wrist RULA score"""
        # RULA Wrist: Score 1 (0-15°), Score 2 (>15°), Score 3 (if deviated from midline)
        if abs(angle) <= 15:
            score = 1
        else:
            score = 2
        
        # Add 1 if wrist is deviated (radial/ulnar deviation)
        if deviated:
            score += 1
            
        return min(score, 4)  # Max wrist score is 4
    
    @staticmethod
    def get_neck_score(angle, twisted=False, side_bent=False):
        """Calculate neck RULA score"""
        if 0 <= angle < 10:
            score = 1
        elif 10 <= angle <= 20:
            score = 2
        elif angle > 20:
            score = 3
        else:  # extension
            score = 4
        
        if twisted or side_bent:
            score += 1
            
        return score
    
    @staticmethod
    def get_trunk_score(angle, twisted=False, side_bent=False):
        """Calculate trunk RULA score"""
        if 0 <= angle < 10:
            score = 1
        elif 10 <= angle <= 20:
            score = 2
        elif 20 < angle <= 60:
            score = 3
        else:
            score = 4
        
        if twisted or side_bent:
            score += 1
            
        return score
    
    @staticmethod
    def get_posture_score_a(upper_arm, lower_arm, wrist, wrist_twist):
        """Get posture score A from official RULA Table A"""
        # Official RULA Table A: Upper Arm (rows) × Lower Arm (cols) for each Wrist position
        # Wrist Position 1
        table_a_wrist1 = {
            (1, 1): 1, (1, 2): 2, (1, 3): 2,
            (2, 1): 2, (2, 2): 2, (2, 3): 2,
            (3, 1): 2, (3, 2): 3, (3, 3): 3,
            (4, 1): 2, (4, 2): 3, (4, 3): 3,
            (5, 1): 3, (5, 2): 4, (5, 3): 4,
            (6, 1): 3, (6, 2): 4, (6, 3): 4,
        }
        # Wrist Position 2
        table_a_wrist2 = {
            (1, 1): 2, (1, 2): 2, (1, 3): 3,
            (2, 1): 2, (2, 2): 2, (2, 3): 3,
            (3, 1): 3, (3, 2): 3, (3, 3): 3,
            (4, 1): 3, (4, 2): 3, (4, 3): 4,
            (5, 1): 4, (5, 2): 4, (5, 3): 4,
            (6, 1): 4, (6, 2): 4, (6, 3): 4,
        }
        # Wrist Position 3
        table_a_wrist3 = {
            (1, 1): 2, (1, 2): 3, (1, 3): 3,
            (2, 1): 3, (2, 2): 3, (2, 3): 3,
            (3, 1): 3, (3, 2): 4, (3, 3): 4,
            (4, 1): 4, (4, 2): 4, (4, 3): 4,
            (5, 1): 4, (5, 2): 4, (5, 3): 5,
            (6, 1): 4, (6, 2): 4, (6, 3): 5,
        }
        # Wrist Position 4
        table_a_wrist4 = {
            (1, 1): 3, (1, 2): 3, (1, 3): 4,
            (2, 1): 3, (2, 2): 3, (2, 3): 4,
            (3, 1): 3, (3, 2): 4, (3, 3): 4,
            (4, 1): 4, (4, 2): 4, (4, 3): 4,
            (5, 1): 4, (5, 2): 4, (5, 3): 5,
            (6, 1): 4, (6, 2): 4, (6, 3): 5,
        }
        
        # Select correct table based on wrist position
        tables = {1: table_a_wrist1, 2: table_a_wrist2, 3: table_a_wrist3, 4: table_a_wrist4}
        table = tables.get(min(wrist, 4), table_a_wrist4)
        
        # Look up score
        key = (min(upper_arm, 6), min(lower_arm, 3))
        score_a = table.get(key, 4)
        
        # Add wrist twist adjustment (+1 if at end of range)
        if wrist_twist == 2:
            score_a += 1
            
        return score_a
    
    @staticmethod
    def get_posture_score_b(neck, trunk, legs):
        """Get posture score B from official RULA Table B"""
        # Official RULA Table B: Neck (rows) × Trunk (cols)
        # Legs = 1 (supported and balanced)
        table_b_legs1 = {
            (1, 1): 1, (1, 2): 2, (1, 3): 3, (1, 4): 5, (1, 5): 6, (1, 6): 7,
            (2, 1): 2, (2, 2): 2, (2, 3): 4, (2, 4): 5, (2, 5): 6, (2, 6): 7,
            (3, 1): 3, (3, 2): 3, (3, 3): 4, (3, 4): 5, (3, 5): 6, (3, 6): 7,
            (4, 1): 5, (4, 2): 5, (4, 3): 6, (4, 4): 7, (4, 5): 7, (4, 6): 7,
            (5, 1): 7, (5, 2): 7, (5, 3): 7, (5, 4): 7, (5, 5): 7, (5, 6): 8,
            (6, 1): 8, (6, 2): 8, (6, 3): 8, (6, 4): 8, (6, 5): 8, (6, 6): 8,
        }
        # Legs = 2 (not supported)
        table_b_legs2 = {
            (1, 1): 1, (1, 2): 3, (1, 3): 4, (1, 4): 6, (1, 5): 7, (1, 6): 7,
            (2, 1): 2, (2, 2): 3, (2, 3): 5, (2, 4): 6, (2, 5): 7, (2, 6): 7,
            (3, 1): 3, (3, 2): 4, (3, 3): 5, (3, 4): 6, (3, 5): 7, (3, 6): 7,
            (4, 1): 5, (4, 2): 6, (4, 3): 7, (4, 4): 7, (4, 5): 7, (4, 6): 8,
            (5, 1): 7, (5, 2): 7, (5, 3): 7, (5, 4): 8, (5, 5): 8, (5, 6): 8,
            (6, 1): 8, (6, 2): 8, (6, 3): 8, (6, 4): 8, (6, 5): 8, (6, 6): 8,
        }
        
        # Select correct table based on legs
        table = table_b_legs1 if legs == 1 else table_b_legs2
        
        # Look up score
        key = (min(neck, 6), min(trunk, 6))
        score_b = table.get(key, 6)
            
        return score_b
    
    @staticmethod
    def get_final_score(score_a, score_b, muscle_use=0, force_load=0):
        """Get final RULA score from official RULA Table C"""
        # Add muscle use and force to Score A and Score B first (official RULA method)
        final_score_a = score_a + muscle_use + force_load
        final_score_b = score_b + muscle_use + force_load
        
        # Official RULA Table C: Score A × Score B → Grand Score
        table_c = {
            (1, 1): 1, (1, 2): 2, (1, 3): 3, (1, 4): 3, (1, 5): 4, (1, 6): 5, (1, 7): 5,
            (2, 1): 2, (2, 2): 2, (2, 3): 3, (2, 4): 4, (2, 5): 4, (2, 6): 5, (2, 7): 5,
            (3, 1): 3, (3, 2): 3, (3, 3): 3, (3, 4): 4, (3, 5): 4, (3, 6): 6, (3, 7): 6,
            (4, 1): 3, (4, 2): 3, (4, 3): 3, (4, 4): 4, (4, 5): 5, (4, 6): 6, (4, 7): 6,
            (5, 1): 4, (5, 2): 4, (5, 3): 4, (5, 4): 5, (5, 5): 6, (5, 6): 7, (5, 7): 7,
            (6, 1): 4, (6, 2): 4, (6, 3): 5, (6, 4): 6, (6, 5): 6, (6, 6): 7, (6, 7): 7,
            (7, 1): 5, (7, 2): 5, (7, 3): 6, (7, 4): 6, (7, 5): 7, (7, 6): 7, (7, 7): 7,
            (8, 1): 5, (8, 2): 5, (8, 3): 6, (8, 4): 7, (8, 5): 7, (8, 6): 7, (8, 7): 7,
        }
        
        key = (min(final_score_a, 8), min(final_score_b, 8))
        grand_score = table_c.get(key, 7)
        
        return min(grand_score, 7)  # RULA max is 7
    
    @classmethod
    def calculate_rula_from_landmarks(cls, landmarks):
        """Calculate RULA score from MediaPipe landmarks with automatic adjustments"""
        try:
            # Extract key points (using MediaPipe pose landmark indices)
            left_shoulder = [landmarks[11].x, landmarks[11].y, landmarks[11].z]
            right_shoulder = [landmarks[12].x, landmarks[12].y, landmarks[12].z]
            left_elbow = [landmarks[13].x, landmarks[13].y, landmarks[13].z]
            right_elbow = [landmarks[14].x, landmarks[14].y, landmarks[14].z]
            left_wrist = [landmarks[15].x, landmarks[15].y, landmarks[15].z]
            right_wrist = [landmarks[16].x, landmarks[16].y, landmarks[16].z]
            left_hip = [landmarks[23].x, landmarks[23].y, landmarks[23].z]
            right_hip = [landmarks[24].x, landmarks[24].y, landmarks[24].z]
            nose = [landmarks[0].x, landmarks[0].y, landmarks[0].z]
            
            # Calculate angles - using right side
            shoulder = right_shoulder
            elbow = right_elbow
            wrist = right_wrist
            hip = right_hip
            
            # UPPER ARM ANGLE: Angle from vertical (0° = arm hanging down, 90° = horizontal)
            # Calculate angle between vertical and upper arm
            vertical_down = [shoulder[0], shoulder[1] + 0.2, shoulder[2]]  # Point below shoulder
            upper_arm_angle = cls.calculate_angle(vertical_down, shoulder, elbow)
            
            # If arm is behind body, angle should be negative (extension)
            if elbow[0] < shoulder[0]:  # Elbow behind shoulder
                upper_arm_angle = -upper_arm_angle
            
            # Convert to RULA scale (0° = neutral hanging, positive = flexion, negative = extension)
            # For RULA: measure deviation from hanging (neutral) position
            upper_arm_angle = abs(upper_arm_angle)
            
            # LOWER ARM ANGLE: Elbow flexion angle (0° = straight, 180° = fully bent)
            lower_arm_angle = cls.calculate_angle(shoulder, elbow, wrist)
            
            # WRIST ANGLE: Flexion/extension from neutral
            # Calculate angle between forearm and hand
            # Create reference point along forearm direction
            forearm_direction_x = wrist[0] - elbow[0]
            forearm_direction_y = wrist[1] - elbow[1]
            forearm_extension = [wrist[0] + forearm_direction_x * 0.1, 
                                 wrist[1] + forearm_direction_y * 0.1, 
                                 wrist[2]]
            
            wrist_angle = cls.calculate_angle(elbow, wrist, forearm_extension)
            # Convert to deviation from neutral (0° = neutral/straight)
            wrist_angle = abs(wrist_angle - 180)  # 180° = straight, deviation from that
            
            # NECK ANGLE: Flexion/extension from neutral
            # Calculate angle between vertical and neck
            neck_base = [(left_shoulder[0] + right_shoulder[0])/2,
                        (left_shoulder[1] + right_shoulder[1])/2,
                        (left_shoulder[2] + right_shoulder[2])/2]
            vertical_up = [neck_base[0], neck_base[1] - 0.2, neck_base[2]]
            neck_angle = cls.calculate_angle(vertical_up, neck_base, nose)
            
            # For RULA: 0° = neutral, positive = flexion
            if nose[1] > neck_base[1]:  # Nose below shoulder (flexion)
                neck_angle = abs(neck_angle)
            else:  # Extension
                neck_angle = -abs(neck_angle)
            
            # TRUNK ANGLE: Flexion/extension from vertical
            trunk_midpoint_shoulder = [(left_shoulder[0] + right_shoulder[0])/2,
                                       (left_shoulder[1] + right_shoulder[1])/2,
                                       (left_shoulder[2] + right_shoulder[2])/2]
            trunk_midpoint_hip = [(left_hip[0] + right_hip[0])/2,
                                  (left_hip[1] + right_hip[1])/2,
                                  (left_hip[2] + right_hip[2])/2]
            
            # Vertical reference below hips
            vertical_ref = [trunk_midpoint_hip[0], trunk_midpoint_hip[1] + 0.2, trunk_midpoint_hip[2]]
            trunk_angle = cls.calculate_angle(vertical_ref, trunk_midpoint_hip, trunk_midpoint_shoulder)
            
            # For RULA: measure deviation from upright (vertical = 0°)
            trunk_angle = abs(trunk_angle)
            
            # Ensure angles are in reasonable ranges
            upper_arm_angle = min(upper_arm_angle, 180)
            lower_arm_angle = max(0, min(lower_arm_angle, 180))
            wrist_angle = min(abs(wrist_angle), 90)  # Cap at 90° deviation
            neck_angle = max(-45, min(neck_angle, 90))  # Reasonable neck range
            trunk_angle = min(trunk_angle, 90)  # Cap at 90° forward bend
            
            # AUTO-DETECT ADJUSTMENTS
            upper_arm_raised = cls.detect_shoulder_raised(landmarks)
            upper_arm_abducted = cls.detect_arm_abducted(landmarks)
            lower_arm_midline = cls.detect_working_across_midline(landmarks)
            wrist_deviated = cls.detect_wrist_deviation(landmarks)
            neck_twisted = cls.detect_neck_twisted(landmarks)
            neck_bent = cls.detect_neck_side_bent(landmarks)
            trunk_twisted = cls.detect_trunk_twisted(landmarks)
            trunk_bent = cls.detect_trunk_side_bent(landmarks)
            
            # Calculate RULA scores WITH automatic adjustments
            upper_arm_score = cls.get_upper_arm_score(upper_arm_angle, upper_arm_raised, upper_arm_abducted)
            lower_arm_score = cls.get_lower_arm_score(lower_arm_angle, lower_arm_midline)
            wrist_score = cls.get_wrist_score(wrist_angle, wrist_deviated)
            wrist_twist = 1  # Default mid-range (can't detect from single camera)
            
            neck_score = cls.get_neck_score(neck_angle, neck_twisted, neck_bent)
            trunk_score = cls.get_trunk_score(trunk_angle, trunk_twisted, trunk_bent)
            legs_score = 1  # Assume supported (can't reliably detect from video)
            
            # Get posture scores
            score_a = cls.get_posture_score_a(upper_arm_score, lower_arm_score, wrist_score, wrist_twist)
            score_b = cls.get_posture_score_b(neck_score, trunk_score, legs_score)
            
            # Final RULA score (no muscle/force - need manual input)
            final_score = cls.get_final_score(score_a, score_b, 0, 0)
            
            return {
                'rula_score': final_score,
                'upper_arm_angle': upper_arm_angle,
                'lower_arm_angle': lower_arm_angle,
                'wrist_angle': wrist_angle,
                'neck_angle': neck_angle,
                'trunk_angle': trunk_angle,
                'score_a': score_a,
                'score_b': score_b,
                # Store auto-detected adjustments
                'upper_arm_raised': upper_arm_raised,
                'upper_arm_abducted': upper_arm_abducted,
                'lower_arm_midline': lower_arm_midline,
                'wrist_deviated': wrist_deviated,
                'neck_twisted': neck_twisted,
                'neck_bent': neck_bent,
                'trunk_twisted': trunk_twisted,
                'trunk_bent': trunk_bent,
            }
        except Exception as e:
            return None


def process_video(video_path, progress_bar=None):
    """Process video and calculate RULA scores"""
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps = max(fps, 15)  # Ensure minimum fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Store frames in memory first, then write
    processed_frames = []
    
    results_data = []
    frame_count = 0
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if progress_bar:
                progress_bar.progress(frame_count / total_frames)
            
            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Process with MediaPipe
            results = pose.process(image)
            
            # Convert back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            rula_score = None
            if results.pose_landmarks:
                # Calculate RULA
                rula_data = RULACalculator.calculate_rula_from_landmarks(results.pose_landmarks.landmark)
                
                if rula_data:
                    rula_score = rula_data['rula_score']
                    
                    # Store results
                    results_data.append({
                        'frame': frame_count,
                        'time_sec': frame_count / fps,
                        'rula_score': rula_score,
                        **rula_data
                    })
                    
                    # DON'T draw score on frame - removed for cleaner video
                
                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
            
            # Store frame
            processed_frames.append(image.copy())
    
    cap.release()
    
    # Write frames to AVI file with MJPEG codec (browser-compatible)
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
    
    # Use MJPEG codec - works without FFmpeg
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise Exception("Could not create video output")
    
    # Write all frames
    for frame in processed_frames:
        out.write(frame)
    
    out.release()
    
    # Verify video was created
    if not os.path.exists(output_path):
        raise Exception("Video file was not created")
    
    return output_path, pd.DataFrame(results_data)


def create_score_timeline(df, lang='en'):
    """Create interactive timeline plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['time_sec'],
        y=df['rula_score'],
        mode='lines+markers',
        name='RULA Score',
        line=dict(color='rgb(0, 123, 255)', width=2),
        marker=dict(size=6)
    ))
    
    # Add risk level zones
    fig.add_hrect(y0=0, y1=2, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=2, y1=3, fillcolor="yellow", opacity=0.1, line_width=0)
    fig.add_hrect(y0=3, y1=5, fillcolor="orange", opacity=0.1, line_width=0)
    fig.add_hrect(y0=5, y1=7, fillcolor="red", opacity=0.1, line_width=0)
    
    fig.update_layout(
        title=TRANSLATIONS[lang]['score_timeline'],
        xaxis_title='Time (seconds)' if lang == 'en' else 'Waktu (detik)',
        yaxis_title='RULA Score' if lang == 'en' else 'Skor RULA',
        yaxis=dict(range=[0, 7.5], dtick=1),
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_score_timeline_comparison(df, lang='en'):
    """Create interactive timeline plot comparing original and adjusted scores"""
    fig = go.Figure()
    
    # Original scores
    fig.add_trace(go.Scatter(
        x=df['time_sec'],
        y=df['rula_score'],
        mode='lines+markers',
        name='Original RULA' if lang == 'en' else 'RULA Asli',
        line=dict(color='rgb(0, 123, 255)', width=2, dash='dash'),
        marker=dict(size=4),
        opacity=0.6
    ))
    
    # Adjusted scores
    fig.add_trace(go.Scatter(
        x=df['time_sec'],
        y=df['adjusted_rula_score'],
        mode='lines+markers',
        name='Adjusted RULA' if lang == 'en' else 'RULA Disesuaikan',
        line=dict(color='rgb(255, 0, 0)', width=2),
        marker=dict(size=6)
    ))
    
    # Add risk level zones
    fig.add_hrect(y0=0, y1=2, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=2, y1=3, fillcolor="yellow", opacity=0.1, line_width=0)
    fig.add_hrect(y0=3, y1=5, fillcolor="orange", opacity=0.1, line_width=0)
    fig.add_hrect(y0=5, y1=7, fillcolor="red", opacity=0.1, line_width=0)
    
    fig.update_layout(
        title='RULA Score Comparison' if lang == 'en' else 'Perbandingan Skor RULA',
        xaxis_title='Time (seconds)' if lang == 'en' else 'Waktu (detik)',
        yaxis_title='RULA Score' if lang == 'en' else 'Skor RULA',
        yaxis=dict(range=[0, 7.5], dtick=1),
        hovermode='x unified',
        height=400,
        showlegend=True
    )
    
    return fig


def get_risk_level(score):
    """Determine risk level from RULA score"""
    if score <= 2:
        return 1
    elif score <= 4:
        return 2
    elif score <= 6:
        return 3
    else:
        return 4


def main():
    # Language selector in sidebar
    lang = st.sidebar.selectbox(
        TRANSLATIONS['en']['lang_label'],
        options=['en', 'id'],
        format_func=lambda x: 'English' if x == 'en' else 'Bahasa Indonesia'
    )
    
    t = TRANSLATIONS[lang]
    
    # Title
    st.title(t['title'])
    st.markdown(f"**{t['subtitle']}**")
    
    # File upload
    uploaded_file = st.file_uploader(
        t['upload_label'],
        type=['mp4', 'avi', 'mov', 'mkv'],
        help=t['upload_help']
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Process button
        if st.button('🚀 ' + ('Analyze Video' if lang == 'en' else 'Analisis Video'), type='primary'):
            with st.spinner(t['processing']):
                progress_bar = st.progress(0)
                
                try:
                    # Process video
                    output_video_path, results_df = process_video(video_path, progress_bar)
                    
                    if len(results_df) == 0:
                        st.error(t['error_no_pose'])
                    else:
                        st.success(t['analysis_complete'])
                        
                        # Store in session state to persist across form submissions
                        st.session_state.output_video_path = output_video_path
                        st.session_state.results_df = results_df
                        st.session_state.video_processed = True
                        
                except Exception as e:
                    st.error(f"{t['error_processing']}: {str(e)}")
        
        # Display results if video has been processed (persists across form submissions)
        if 'video_processed' in st.session_state and st.session_state.video_processed:
            output_video_path = st.session_state.output_video_path
            results_df = st.session_state.results_df.copy()  # Make a copy to avoid modifying original
            
            # Calculate statistics
            avg_score = results_df['rula_score'].mean()
            max_score = results_df['rula_score'].max()
            min_score = results_df['rula_score'].min()
            risk_level = get_risk_level(avg_score)
            
            # Calculate average auto-detected adjustments for pre-filling
            auto_upper_arm_raised = results_df['upper_arm_raised'].mean() > 0.5
            auto_upper_arm_abducted = results_df['upper_arm_abducted'].mean() > 0.5
            auto_lower_arm_midline = results_df['lower_arm_midline'].mean() > 0.5
            auto_wrist_deviated = results_df['wrist_deviated'].mean() > 0.5
            auto_neck_twisted = results_df['neck_twisted'].mean() > 0.5
            auto_neck_bent = results_df['neck_bent'].mean() > 0.5
            auto_trunk_twisted = results_df['trunk_twisted'].mean() > 0.5
            auto_trunk_bent = results_df['trunk_bent'].mean() > 0.5
            
            # Display annotated video immediately (only once, doesn't reload)
            st.markdown(f"### {t['annotated_video']}")
            
            # TWO COLUMN LAYOUT: Video (left) + Adjustments (right)
            video_col, adjustment_col = st.columns([2, 3])
            
            # LEFT COLUMN: Video (smaller)
            with video_col:
                try:
                    st.video(output_video_path)
                except Exception as e:
                    st.warning("⚠️ " + ("Video preview not available. Download below." if lang == 'en' else "Pratinjau tidak tersedia. Unduh di bawah."))
                    with open(output_video_path, 'rb') as f:
                        video_bytes = f.read()
                        st.download_button(
                            label=f"📥 " + ("Download Video" if lang == 'en' else "Unduh Video"),
                            data=video_bytes,
                            file_name=f"selarassehat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                            mime='video/mp4',
                            type='primary'
                        )
            
            # RIGHT COLUMN: Manual Adjustments
            with adjustment_col:
                st.markdown(f"### {t['adjustments_title']}")
                st.caption(t['adjustments_help'])
                st.info("✨ " + ("Auto-detected adjustments are pre-checked" if lang == 'en' else "Penyesuaian terdeteksi otomatis sudah dicentang"))
                
                # Create form for adjustments
                with st.form(key='adjustment_form'):
                    # Group A
                    st.markdown("**Group A (Arms & Wrists)**")
                    upper_arm_raised = st.checkbox(t['upper_arm_raised'], value=auto_upper_arm_raised, key='ua_raised')
                    upper_arm_abducted = st.checkbox(t['upper_arm_abducted'], value=auto_upper_arm_abducted, key='ua_abd')
                    lower_arm_midline = st.checkbox(t['lower_arm_midline'], value=auto_lower_arm_midline, key='la_mid')
                    wrist_deviated = st.checkbox(t['wrist_deviated'], value=auto_wrist_deviated, key='w_dev')
                    wrist_twist = st.radio(
                        t['wrist_twist_label'],
                        options=[1, 2],
                        format_func=lambda x: t['wrist_twist_mid'] if x == 1 else t['wrist_twist_extreme'],
                        key='w_twist',
                        horizontal=True
                    )
                    
                    st.markdown("---")
                    
                    # Group B
                    st.markdown("**Group B (Neck, Trunk, Legs)**")
                    neck_twisted = st.checkbox(t['neck_twisted'], value=auto_neck_twisted, key='n_twist')
                    neck_bent = st.checkbox(t['neck_bent'], value=auto_neck_bent, key='n_bent')
                    trunk_twisted = st.checkbox(t['trunk_twisted'], value=auto_trunk_twisted, key='t_twist')
                    trunk_bent = st.checkbox(t['trunk_bent'], value=auto_trunk_bent, key='t_bent')
                    legs_score = st.radio(
                        t['legs_label'],
                        options=[1, 2],
                        format_func=lambda x: t['legs_supported'] if x == 1 else t['legs_not_supported'],
                        key='legs',
                        horizontal=True
                    )
                    
                    st.markdown("---")
                    
                    # Additional Factors
                    st.markdown("**Additional Factors**")
                    st.caption("⚠️ " + ("Manual input required" if lang == 'en' else "Input manual diperlukan"))
                    muscle_use = st.checkbox(t['muscle_label'], key='muscle') * 1
                    force_load = st.radio(
                        t['force_label'],
                        options=[0, 1, 2, 3],
                        format_func=lambda x: [t['force_none'], t['force_light'], t['force_heavy'], t['force_shock']][x],
                        key='force'
                    )
                    
                    # Submit button
                    submit_button = st.form_submit_button(
                        label='🔄 ' + ('Recalculate RULA' if lang == 'en' else 'Hitung Ulang RULA'),
                        type='primary',
                        use_container_width=True
                    )
            
            # BOTTOM: Summary Results
            st.markdown("---")
            st.markdown(f"## {t['results_title']}")
            
            # Original scores
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(t['avg_score'], f"{avg_score:.1f}")
            with col2:
                st.metric(t['max_score'], f"{max_score:.0f}")
            with col3:
                st.metric(t['min_score'], f"{min_score:.0f}")
            with col4:
                st.metric(t['risk_level'], f"{risk_level}")
            
            # Recommendation
            st.info(f"**{t['recommendation']}:** {t['risk_levels'][risk_level]}")
            
            # Show adjusted scores only after submit
            if submit_button:
                # Show what adjustments were applied
                st.markdown("---")
                st.markdown("### " + ("Applied Adjustments" if lang == 'en' else "Penyesuaian yang Diterapkan"))
                
                adj_summary = []
                if upper_arm_raised: adj_summary.append("✓ Shoulder raised")
                if upper_arm_abducted: adj_summary.append("✓ Arm abducted")
                if lower_arm_midline: adj_summary.append("✓ Working across midline")
                if wrist_deviated: adj_summary.append("✓ Wrist deviated")
                if neck_twisted: adj_summary.append("✓ Neck twisted")
                if neck_bent: adj_summary.append("✓ Neck side bent")
                if trunk_twisted: adj_summary.append("✓ Trunk twisted")
                if trunk_bent: adj_summary.append("✓ Trunk side bent")
                adj_summary.append(f"Wrist twist: {wrist_twist}")
                adj_summary.append(f"Legs: {'Supported' if legs_score == 1 else 'Not supported'}")
                if muscle_use: adj_summary.append("✓ Muscle use")
                if force_load > 0: adj_summary.append(f"Force: {force_load}")
                
                st.info(" | ".join(adj_summary))
                
                # Recalculate RULA with adjustments
                adjusted_scores = []
                for _, row in results_df.iterrows():
                    new_score, score_a, score_b = RULACalculator.recalculate_rula(
                        row['upper_arm_angle'], row['lower_arm_angle'], row['wrist_angle'],
                        row['neck_angle'], row['trunk_angle'],
                        upper_arm_raised, upper_arm_abducted, lower_arm_midline, wrist_deviated,
                        neck_twisted, neck_bent, trunk_twisted, trunk_bent,
                        wrist_twist, legs_score, muscle_use, force_load
                    )
                    adjusted_scores.append(new_score)
                
                results_df['adjusted_rula_score'] = adjusted_scores
                
                # Display adjusted statistics
                st.markdown("### " + ("Adjusted RULA Scores" if lang == 'en' else "Skor RULA Disesuaikan"))
                
                adj_avg = results_df['adjusted_rula_score'].mean()
                adj_max = results_df['adjusted_rula_score'].max()
                adj_min = results_df['adjusted_rula_score'].min()
                adj_risk = get_risk_level(adj_avg)
                
                acol1, acol2, acol3, acol4 = st.columns(4)
                with acol1:
                    st.metric(t['avg_score'], f"{adj_avg:.1f}", f"{adj_avg - avg_score:+.1f}")
                with acol2:
                    st.metric(t['max_score'], f"{adj_max:.0f}", f"{adj_max - max_score:+.0f}")
                with acol3:
                    st.metric(t['min_score'], f"{adj_min:.0f}", f"{adj_min - min_score:+.0f}")
                with acol4:
                    st.metric(t['risk_level'], f"{adj_risk}", f"{adj_risk - risk_level:+d}")
                
                st.info(f"**{t['recommendation']}:** {t['risk_levels'][adj_risk]}")
                
                # Show comparison timeline
                st.plotly_chart(create_score_timeline_comparison(results_df, lang), use_container_width=True)
            else:
                # Show original timeline only
                st.plotly_chart(create_score_timeline(results_df, lang), use_container_width=True)
            
            st.markdown("---")
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label=f"📊 {t['download_csv']}",
                    data=csv,
                    file_name=f"selarassehat_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv'
                )
            
            with col2:
                with open(output_video_path, 'rb') as f:
                    st.download_button(
                        label=f"🎥 {t['download_video']}",
                        data=f,
                        file_name=f"selarassehat_annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime='video/mp4'
                    )
        
        # Cleanup uploaded file
        try:
            os.unlink(video_path)
        except:
            pass


if __name__ == "__main__":
    main()
