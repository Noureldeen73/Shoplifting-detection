import os
import cv2
import numpy as np
import tensorflow as tf
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
IMG_SIZE = 128  # Resize frames to 128x128
MAX_FRAMES = 20  # Use a fixed number of frames from each video
MODEL_NAME = "conv3d.h5"  # Fixed model to use

class VideoPreprocessor:
    
    def __init__(self, img_size: int = IMG_SIZE, max_frames: int = MAX_FRAMES):
        self.img_size = img_size
        self.max_frames = max_frames
    
    def extract_frames_with_bias(self, video_path: str) -> Optional[np.ndarray]:
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return None
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Skip videos that are too short to sample from
            if total_frames < self.max_frames:
                logger.warning(f"Video too short ({total_frames} frames): {video_path}")
                cap.release()
                return None

            frames = []
            # Define how many frames to take from the middle vs. edges
            n_frames_middle = int(self.max_frames * 0.7)  # e.g., 14 frames
            n_frames_edges = self.max_frames - n_frames_middle  # e.g., 6 frames

            # Define the video sections
            middle_start_frame = int(total_frames * 0.25)
            middle_end_frame = int(total_frames * 0.75)

            # Generate indices for frames to extract
            middle_indices = np.linspace(middle_start_frame, middle_end_frame, n_frames_middle, dtype=int)
            edge_indices_start = np.linspace(0, middle_start_frame - 1, max(1, n_frames_edges // 2), dtype=int)
            edge_indices_end = np.linspace(middle_end_frame + 1, total_frames - 1, 
                                         max(1, n_frames_edges - (n_frames_edges // 2)), dtype=int)
            
            # Combine indices, ensure they are unique and sorted
            combined_indices = np.concatenate([edge_indices_start, middle_indices, edge_indices_end])
            unique_indices = np.unique(combined_indices)
            unique_indices.sort()

            # Extract the frames at the calculated indices
            for frame_idx in unique_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.img_size, self.img_size))
                    frames.append(frame)
            
            cap.release()

            # --- Padding / Truncating to ensure fixed length ---
            # Ensure every sequence has exactly MAX_FRAMES
            if len(frames) > self.max_frames:
                frames = frames[:self.max_frames]
            elif len(frames) < self.max_frames and len(frames) > 0:
                # Pad with the last frame if we have too few
                padding = [frames[-1]] * (self.max_frames - len(frames))
                frames.extend(padding)

            if len(frames) == self.max_frames:
                # Normalize pixel values to [0, 1]
                frames_array = np.array(frames, dtype=np.float32) / 255.0
                return frames_array
            else:
                logger.warning(f"Could not extract {self.max_frames} frames from: {video_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return None

class TheftDetectionModel:
    def __init__(self):
        self.model = None
        self.preprocessor = VideoPreprocessor()
        self.model_path = self._get_model_path()
        
        if self.model_path:
            self.load_model()
    
    def _get_model_path(self) -> str:
        # Construct path to model in project root using MODEL_NAME constant
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(project_root, MODEL_NAME)
        logger.info(f"Looking for 3D Convolution model at: {model_path}")
        return model_path
    
    def load_model(self) -> bool:
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"3D Convolution model file does not exist: {self.model_path}")
                return False
                
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"3D Convolution model loaded successfully from: {self.model_path}")
            
            # Log model input shape for debugging
            if hasattr(self.model, 'input_shape'):
                logger.info(f"3D Convolution model expected input shape: {self.model.input_shape}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load 3D Convolution model from {self.model_path}: {str(e)}")
            return False
    
    def predict_single_video(self, video_path: str) -> dict:
        if self.model is None:
            return {
                'error': '3D Convolution model not loaded',
                'theft_detected': False,
                'confidence': 0.0,
                'prediction_probability': 0.0
            }
        frames = self.preprocessor.extract_frames_with_bias(video_path)
        
        if frames is None:
            return {
                'error': 'Failed to process video',
                'theft_detected': False,
                'confidence': 0.0,
                'prediction_probability': 0.0
            }
        
        try:
            # Add batch dimension: from (20, 128, 128, 3) to (1, 20, 128, 128, 3)
            frames_batch = np.expand_dims(frames, axis=0)
            logger.info(f"Input shape to 3D Convolution model: {frames_batch.shape}")
            logger.info(f"Expected shape for 3D Convolution: (1, {MAX_FRAMES}, {IMG_SIZE}, {IMG_SIZE}, 3)")
            
            prediction = self.model.predict(frames_batch, verbose=0)
            logger.info(f"3D Convolution prediction output shape: {prediction.shape}")
            
            # Extract probability (assuming binary classification)
            if prediction.shape[-1] == 1:
                # Binary classification with sigmoid
                probability = float(prediction[0][0])
            else:
                # Multi-class classification, get probability of theft class
                probability = float(prediction[0][1]) if prediction.shape[-1] > 1 else float(prediction[0][0])
            
            theft_detected = probability > 0.5
            confidence = probability if theft_detected else 1 - probability
            
            return {
                'theft_detected': theft_detected,
                'confidence': confidence,
                'prediction_probability': probability,
                'frames_processed': len(frames),
                'model_used': os.path.basename(self.model_path).split('.')[0].upper(),
                'input_shape': str(frames_batch.shape)
            }
            
        except Exception as e:
            logger.error(f"3D Convolution model prediction failed: {str(e)}")
            return {
                'error': f'3D Convolution model prediction failed: {str(e)}',
                'theft_detected': False,
                'confidence': 0.0,
                'prediction_probability': 0.0
            }


def process_video_for_inference(video_path: str) -> dict:
    
    detector = TheftDetectionModel()
    return detector.predict_single_video(video_path)


def get_model_info() -> dict:

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(project_root, MODEL_NAME)
    
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path)
        return {
            'model_name': MODEL_NAME,
            'model_path': model_path,
            'size_mb': round(model_size / (1024 * 1024), 2),
            'exists': True
        }
    else:
        return {
            'model_name': MODEL_NAME,
            'model_path': model_path,
            'size_mb': 0,
            'exists': False,
            'error': f'Model file not found: {model_path}'
        }


# Example usage
if __name__ == "__main__":
    
    print("Model info:", get_model_info())
    
    preprocessor = VideoPreprocessor()