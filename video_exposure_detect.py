import numpy as np
import tempfile
import traceback
from collections import deque

class VideoOverexposureProcessor:
    """视频过曝处理器，带有时序平滑功能"""
    
    def __init__(self, detector, consecutive_frames: int = 3, save_frames: bool = False):
        """
        初始化视频处理器
        
        Args:
            detector: HSVOverexposureDetector实例
            consecutive_frames: 判定为有效过曝的连续帧数，推荐值为3
            save_frames: 是否保存单帧检测结果（图像/中间文件）
        """
        self.detector = detector
        self.save_frames = save_frames # 控制是否保存单帧
        self.consecutive_frames = consecutive_frames  # 推荐值：3帧
        self.detection_history = deque(maxlen=consecutive_frames)
        self.is_overexposed = False
        self.frame_count = 0
    
def process_frame(self, frame: np.ndarray, save_dir: str = None) -> dict:
    """处理单帧图像"""
    self.frame_count += 1
    try:
        # 使用系统临时目录创建唯一文件（避免冲突）
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, frame)
        
        # 仅当save_frames为True时才保存单帧结果
        save_frame_dir = os.path.join(save_dir, f"frame_{self.frame_count}") if (self.save_frames and save_dir) else None
        
        # 执行检测
        result = self.detector.detect(
            image_path=temp_path,
            exclude_regions=None,
            save_dir=save_frame_dir
        )
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # 记录检测历史
        self.detection_history.append(result['detected'])
        
        # 时序平滑判断
        if len(self.detection_history) == self.consecutive_frames:
            self.is_overexposed = all(self.detection_history)
        
        # 提取过曝区域边界框
        bboxes = [region['bbox'] for region in result['regions']] if result['detected'] else []
        
        return {
            **result,
            'frame_number': self.frame_count,
            'smoothed_result': self.is_overexposed,
            'consecutive_frames': self.consecutive_frames,
            'bboxes': bboxes
        }
    except Exception as e:
        # 打印详细错误信息（帧处理错误）
        print(f"\n===== 处理第 {self.frame_count} 帧时出错 =====")
        print(f"错误描述: {str(e)}")
        print("详细错误堆栈:")
        traceback.print_exc()  # 打印完整traceback
        # 出错时返回空结果，避免中断整个视频处理
        return {
            'detected': False,
            'frame_number': self.frame_count,
            'smoothed_result': False,
            'bboxes': []
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
        try:
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
    
            # 初始化视频写入器
            output_video_path = os.path.join(video_save_dir, f"{video_name}_overexposure.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式（根据需要调整）
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
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
                # 绘制过曝区域边界框（红色，线宽2）
                for bbox in result['bboxes']:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # 写入视频
                out.write(frame)
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
            out.release()
            
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
        except Exception as e:
                print(f"\n===== 视频 {video_path} 整体处理出错 =====")
                print(f"错误描述: {str(e)}")
                print("详细错误堆栈:")
                traceback.print_exc()  # 打印完整traceback
                return None
