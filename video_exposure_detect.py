import numpy as np
from collections import deque

class VideoOverexposureProcessor:
    """视频过曝处理器，带有时序平滑功能"""
    
    def __init__(self, detector, consecutive_frames: int = 3):
        """
        初始化视频处理器
        
        Args:
            detector: HSVOverexposureDetector实例
            consecutive_frames: 判定为有效过曝的连续帧数，推荐值为3
        """
        self.detector = detector
        self.consecutive_frames = consecutive_frames  # 推荐值：3帧
        self.detection_history = deque(maxlen=consecutive_frames)
        self.is_overexposed = False
        self.frame_count = 0
    
    def process_frame(self, frame: np.ndarray, save_dir: str = None) -> dict:
        """
        处理单帧图像
        
        Args:
            frame: 视频帧
            save_dir: 保存目录
            
        Returns:
            检测结果
        """
        self.frame_count += 1
        
        # 保存当前帧为临时图像进行检测
        temp_path = f"temp_frame_{self.frame_count}.png"
        cv2.imwrite(temp_path, frame)
        
        # 执行检测
        result = self.detector.detect(
            image_path=temp_path,
            exclude_regions=None,
            save_dir=os.path.join(save_dir, f"frame_{self.frame_count}") if save_dir else None
        )
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # 记录检测历史
        self.detection_history.append(result['detected'])
        
        # 时序平滑判断
        if len(self.detection_history) == self.consecutive_frames:
            # 如果连续N帧都检测到过曝，则判定为有效过曝
            self.is_overexposed = all(self.detection_history)
        
        return {
            **result,
            'frame_number': self.frame_count,
            'smoothed_result': self.is_overexposed,
            'consecutive_frames': self.consecutive_frames
        }
    
    def process_video(self, video_path: str, save_dir: str = None) -> dict:
        """
        处理整个视频
        
        Args:
            video_path: 视频路径
            save_dir: 保存目录
            
        Returns:
            视频检测汇总结果
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 创建保存目录
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_save_dir = os.path.join(save_dir, video_name) if save_dir else None
        if video_save_dir:
            os.makedirs(video_save_dir, exist_ok=True)
        
        # 视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        results = []
        overexposed_frames = 0
        overexposed_intervals = []
        in_overexposure = False
        start_frame = 0
        
        print(f"开始处理视频: {video_path}")
        print(f"视频信息: {width}x{height}, {fps:.1f} FPS, 共{total_frames}帧")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理当前帧
            result = self.process_frame(frame, video_save_dir)
            results.append(result)
            
            # 统计过曝帧
            if result['smoothed_result']:
                overexposed_frames += 1
                if not in_overexposure:
                    in_overexposure = True
                    start_frame = self.frame_count
            else:
                if in_overexposure:
                    in_overexposure = False
                    overexposed_intervals.append({
                        'start': start_frame,
                        'end': self.frame_count - 1,
                        'duration': (self.frame_count - 1 - start_frame) / fps
                    })
        
        # 处理视频结束时仍处于过曝状态的情况
        if in_overexposure:
            overexposed_intervals.append({
                'start': start_frame,
                'end': total_frames,
                'duration': (total_frames - start_frame) / fps
            })
        
        cap.release()
        
        # 汇总结果
        total_duration = total_frames / fps if fps > 0 else 0
        overexposed_duration = sum(interval['duration'] for interval in overexposed_intervals)
        overexposed_ratio = overexposed_duration / total_duration if total_duration > 0 else 0
        
        return {
            'video_path': video_path,
            'total_frames': total_frames,
            'total_duration': total_duration,
            'overexposed_frames': overexposed_frames,
            'overexposed_duration': overexposed_duration,
            'overexposed_ratio': overexposed_ratio,
            'overexposed_intervals': overexposed_intervals,
            'consecutive_frames_used': self.consecutive_frames
        }
