from overexposure_detection import HSVOverexposureDetector
from glob import glob
import os


# 使用示例
def demo():
    """演示如何使用HSV过曝检测器"""
    # 创建检测器
    detector = HSVOverexposureDetector(
        grid_size=20,
        v_threshold=225,
        s_threshold=0.3,
        overexpose_threshold=0.2,
        min_region_size=2,
        verbose=True  # 开启详细模式
    )
    
    # 设置豁免区域（支持负数坐标）
    exclude_regions = [[0, 0, 1000, 100], [1000, -100, -1, -1]]
    
    samples_dir = "/home/project/OverExposureDetector/source material/samples"
    im_paths = glob(os.path.join(samples_dir, "*.png"))
    
    for im_path in im_paths:
        # 执行检测
        result = detector.detect(
            image_path=im_path,
            exclude_regions=exclude_regions,
            save_dir='/home/project/OverExposureDetector/data/grid_20_threshold_0.3_s_0.3_v_225_min_region_size_2_overexpose_threshold_0.2'
        )
        
        #输出结果
        print(f"检测完成!")
        print(f"是否检测到过曝: {result['detected']}")
        print(f"过曝区域数量: {result['num_regions']}")
        print(f"处理时间: {result['processing_time']:.3f}秒")
        
        if result['detected']:
            for i, region in enumerate(result['regions']):
                bbox = region['bbox']
                print(f"区域{i+1}: 坐标=({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}), "
                    f"网格数={region['grid_count']}")


if __name__ == "__main__":
    demo()

