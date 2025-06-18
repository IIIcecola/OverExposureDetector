# HSV过曝检测器 - 使用指南

## 概述

HSV过曝检测器V2是一个基于HSV色彩空间的图像过曝区域检测工具，通过分析图像的亮度(V)和饱和度(S)特征来精确识别过曝区域。

## 主要特性

- ✅ **清晰的处理流程**：9个明确定义的处理步骤
- ✅ **灵活的豁免区域**：支持多个豁免区域，支持负数坐标
- ✅ **智能的区域合并**：自动识别相邻网格并合并为过曝区域
- ✅ **详细的日志记录**：可选的verbose模式提供完整的处理过程记录
- ✅ **丰富的可视化**：生成多种分析图像辅助调试

## 快速开始

### 基础使用

```python
from hsv_overexposure_detector_v2 import HSVOverexposureDetectorV2

# 创建检测器
detector = HSVOverexposureDetectorV2(
    grid_size=30,           # 网格大小（像素）
    v_threshold=230,        # V通道阈值 (0-255)
    s_threshold=0.3,        # S通道阈值 (0-1)
    overexpose_threshold=0.5,  # 过曝像素比例阈值
    min_region_size=9,      # 最小区域大小（网格数）
    verbose=False           # 是否输出详细日志
)

# 执行检测
result = detector.detect(
    image_path='your_image.jpg',
    exclude_regions=None,   # 不设置豁免区域
    save_dir='./output'
)

# 查看结果
print(f"检测到过曝: {result['detected']}")
print(f"过曝区域数: {result['num_regions']}")
```

### 使用豁免区域

```python
# 设置豁免区域（如监控画面的时间戳、设备信息等）
exclude_regions = [
    [0, 0, 150, 80],        # 左上角时间戳
    [200, 0, 400, 50],      # 顶部标题
    [-200, -100, -1, -1]    # 右下角设备信息（负数坐标）
]

result = detector.detect(
    image_path='surveillance_image.jpg',
    exclude_regions=exclude_regions,
    save_dir='./output'
)
```

### 启用详细模式

```python
# 创建带详细日志的检测器
detector = HSVOverexposureDetectorV2(
    grid_size=30,
    v_threshold=230,
    s_threshold=0.3,
    overexpose_threshold=0.5,
    min_region_size=9,
    verbose=True  # 启用详细模式
)

# 执行检测会生成：
# 1. 详细的控制台输出
# 2. 日志文件保存到 analysis 文件夹
# 3. HSV分析图和过曝网格图
result = detector.detect(
    image_path='test_image.jpg',
    exclude_regions=exclude_regions,
    save_dir='./output'
)
```

## 参数说明

### 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `grid_size` | int | 30 | 网格大小（像素），影响检测精度和速度 |
| `v_threshold` | int | 230 | V通道（亮度）阈值，范围0-255 |
| `s_threshold` | float | 0.3 | S通道（饱和度）阈值，范围0-1 |
| `overexpose_threshold` | float | 0.5 | 网格内过曝像素比例阈值 |
| `min_region_size` | int | 9 | 最小过曝区域大小（相邻网格数） |
| `verbose` | bool | False | 是否输出详细日志和保存中间结果 |

### detect方法参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `image_path` | str | 输入图像路径 |
| `exclude_regions` | List[List[int]] | 豁免检测区域列表，格式：[[x1,y1,x2,y2], ...] |
| `save_dir` | str | 输出目录路径 |

### 豁免区域坐标说明

豁免区域支持负数坐标，方便从图像边缘定位：
- 正数：从左上角开始计算
- 负数：从右下角开始计算

示例：
```python
# 图像尺寸：640x480
exclude_regions = [
    [0, 0, 100, 50],       # 左上角 100x50 区域
    [-100, -50, -1, -1]    # 右下角 100x50 区域
    # 实际坐标会转换为 [540, 430, 639, 479]
]
```

## 输出说明

### 返回值结构

```python
{
    'detected': bool,           # 是否检测到过曝区域
    'num_regions': int,         # 过曝区域数量
    'regions': [                # 过曝区域列表
        {
            'bbox': (x1, y1, x2, y2),  # 区域边界框
            'grid_count': int,         # 包含的网格数
            'grids': [(row, col), ...] # 网格坐标列表
        },
        ...
    ],
    'processing_time': float,   # 处理时间（秒）
    'result_path': str         # 结果图像路径
}
```

### 生成的文件

1. **结果图像** (`results/` 文件夹)
   - `{filename}_result.jpg`: 在原图上用红色框标记过曝区域

2. **分析图像** (`analysis/` 文件夹，仅verbose=True时生成)
   - `{filename}_hsv_analysis.jpg`: HSV曲线分析图
     - 灰色网格线
     - 白色填充的豁免区域
     - 蓝色V值曲线
     - 粉色S值曲线
   - `{filename}_overexposure_grids.jpg`: 过曝网格标记图
     - 黄色框标记的过曝网格
   - `detection_log_{timestamp}.txt`: 详细处理日志

## 处理流程详解

1. **图像预处理**：BGR → HSV色彩空间转换
2. **网格划分**：按指定大小划分网格，处理边缘不完整网格
3. **豁免区域处理**：规范化坐标，支持负数
4. **有效网格过滤**：排除豁免区域内的网格
5. **HSV统计计算**：计算每个网格的HSV直方图和统计值
6. **过曝判定**：V≥v_threshold AND S≤s_threshold的像素比例
7. **区域合并**：使用DFS找出相邻的过曝网格组
8. **结果绘制**：在原图上标记过曝区域
9. **分析图生成**：可选的详细分析图像

## 最佳实践

### 参数调优建议

1. **检测严格度**
   - 严格：`v_threshold=240, s_threshold=0.2`
   - 平衡：`v_threshold=230, s_threshold=0.3`（推荐）
   - 宽松：`v_threshold=220, s_threshold=0.4`

2. **性能与精度平衡**
   - 高精度：`grid_size=20, min_region_size=4`
   - 平衡：`grid_size=30, min_region_size=9`（推荐）
   - 高性能：`grid_size=40, min_region_size=16`

3. **调试技巧**
   - 先用`verbose=True`模式分析几张典型图像
   - 观察HSV分析图调整阈值参数
   - 根据过曝网格图调整`min_region_size`

### 批量处理示例

```python
import os
from glob import glob

detector = HSVOverexposureDetectorV2(
    grid_size=30,
    v_threshold=230,
    s_threshold=0.3,
    verbose=False  # 批量处理时关闭详细模式
)

# 监控画面的典型豁免区域
exclude_regions = [
    [0, 0, 200, 60],      # 时间戳
    [-200, -60, -1, -1]   # 设备信息
]

# 批量处理
image_files = glob('input/*.jpg')
results = []

for img_path in image_files:
    try:
        result = detector.detect(
            image_path=img_path,
            exclude_regions=exclude_regions,
            save_dir='batch_output'
        )
        results.append({
            'file': img_path,
            'detected': result['detected'],
            'regions': result['num_regions']
        })
        print(f"处理完成: {os.path.basename(img_path)} - "
              f"过曝区域: {result['num_regions']}")
    except Exception as e:
        print(f"处理失败: {img_path} - {str(e)}")

# 统计结果
detected_count = sum(1 for r in results if r['detected'])
print(f"\n总计: {len(results)} 张图像")
print(f"检测到过曝: {detected_count} 张")
```

## 常见问题

### Q: 如何确定合适的网格大小？
A: 网格大小需要根据图像分辨率和过曝区域的典型大小来确定。一般建议：
- 640×480图像：20-30像素
- 1920×1080图像：30-50像素

### Q: 为什么需要设置豁免区域？
A: 监控画面通常包含时间戳、设备信息等高亮文字，这些区域容易被误判为过曝。设置豁免区域可以避免这些误检。

### Q: 如何理解相邻网格？
A: 本算法中，如果两个网格共享顶点（包括对角顶点），就认为它们相邻。这意味着每个网格最多有8个相邻网格。

### Q: verbose模式会影响性能吗？
A: verbose模式会增加日志输出和图像生成的开销，建议仅在调试时使用。批量处理时应设置`verbose=False`。