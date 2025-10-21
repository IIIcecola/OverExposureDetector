import os
import cv2
from typing import List, Tuple

def get_supported_extensions() -> Tuple[List[str], List[str]]:
    """获取支持的图像和视频扩展名"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    return image_extensions, video_extensions

def find_media_files(root_dir: str) -> Tuple[List[str], List[str]]:
    """
    查找目录下所有支持的图像和视频文件（包括嵌套目录）
    
    Args:
        root_dir: 根目录路径
        
    Returns:
        图像文件列表和视频文件列表
    """
    image_extensions, video_extensions = get_supported_extensions()
    image_files = []
    video_files = []
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            file_path = os.path.join(dirpath, filename)
            
            if ext in image_extensions:
                image_files.append(file_path)
            elif ext in video_extensions:
                video_files.append(file_path)
    
    return image_files, video_files

def is_image_file(file_path: str) -> bool:
    """判断文件是否为支持的图像文件"""
    image_extensions, _ = get_supported_extensions()
    ext = os.path.splitext(file_path)[1].lower()
    return ext in image_extensions

def is_video_file(file_path: str) -> bool:
    """判断文件是否为支持的视频文件"""
    _, video_extensions = get_supported_extensions()
    ext = os.path.splitext(file_path)[1].lower()
    return ext in video_extensions
