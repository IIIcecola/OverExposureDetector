from glob import glob
import os
import cv2
from typing import List, Tuple

from overexposure_detection import HSVOverexposureDetector
from video_exposure_detect import VideoOverexposureProcessor
from utils import get_supported_extensions, find_media_files


def process_image(detector, image_path: str, save_dir: str):
    """处理单张图像"""
    try:
        # 执行检测
        result = detector.detect(
            image_path=image_path,
            # 不使用豁免区域
            exclude_regions=None,
            save_dir=os.path.join(save_dir, 'images', os.path.basename(os.path.dirname(image_path)))
        )
        
        # 输出结果
        print(f"处理完成: {os.path.basename(image_path)}")
        print(f"  是否检测到过曝: {result['detected']}")
        print(f"  过曝区域数量: {result['num_regions']}")
        print(f"  处理时间: {result['processing_time']:.3f}秒")
        
        if result['detected']:
            for i, region in enumerate(result['regions']):
                bbox = region['bbox']
                print(f"  区域{i+1}: 坐标=({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}), "
                      f"网格数={region['grid_count']}")
        return result
    except Exception as e:
        print(f"处理图像 {image_path} 失败: {str(e)}")
        return None

def process_video(detector, video_path: str, save_dir: str, consecutive_frames: int = 3, save_frames: bool = False):
    """处理视频文件"""
    try:
        # 创建视频处理器
        video_processor = VideoOverexposureProcessor(detector, consecutive_frames, save_frames)
        
        # 处理视频
        result = video_processor.process_video(
            video_path=video_path,
            save_dir=os.path.join(save_dir, 'videos')
        )
        # 保存视频整体检测结果为JSON
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        result_save_path = os.path.join(save_dir, 'videos', video_name, 'video_result.json')
        os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
        with open(result_save_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)  # 格式化保存
        
        # 输出结果
        print("\n视频处理完成:")
        print(f"  视频路径: {result['video_path']}")
        print(f"  总时长: {result['total_duration']:.2f}秒")
        print(f"  过曝时长: {result['overexposed_duration']:.2f}秒")
        print(f"  过曝比例: {result['overexposed_ratio']:.2%}")
        print(f"  过曝区间数量: {len(result['overexposed_intervals'])}")
        
        for i, interval in enumerate(result['overexposed_intervals']):
            print(f"  区间{i+1}: 帧 {interval['start']}-{interval['end']}, "
                  f"时长 {interval['duration']:.2f}秒")
        
        return result
    except Exception as e:
        print(f"处理视频 {video_path} 失败: {str(e)}")
        return None

# 使用示例
def demo():
    """演示如何使用HSV过曝检测器处理图像和视频"""
    # 创建检测器
    detector = HSVOverexposureDetector(
        grid_size=20,
        v_threshold=225,
        s_threshold=0.3,
        overexpose_threshold=0.2,
        min_region_size=2,
        verbose=True  # 开启详细模式
    )
    
    # 设置样本目录
    samples_dir = "/home/project/OverExposureDetector/source material/samples"
    
    # 查找所有图像和视频文件（包括嵌套目录）
    image_files, video_files = find_media_files(samples_dir)
    
    print(f"发现 {len(image_files)} 张图像，{len(video_files)} 个视频")
    
    # 设置保存目录
    base_save_dir = '/home/project/OverExposureDetector/data/grid_20_threshold_0.3_s_0.3_v_225_min_region_size_2_overexpose_threshold_0.2'
    
    # 处理所有图像
    if image_files:
        print("\n===== 开始处理图像 =====")
        for img_path in image_files:
            process_image(detector, img_path, base_save_dir)
    
    # 处理所有视频（使用3帧连续检测作为推荐值）
    if video_files:
        print("\n===== 开始处理视频 =====")
        for video_path in video_files:
            process_video(detector, video_path, base_save_dir, consecutive_frames=3, save_frames: bool = False)

if __name__ == "__main__":
    demo()

