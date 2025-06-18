#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HSV过曝检测器 - 重构版
基于HSV色彩空间的图像过曝区域检测算法
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Set
import time
import logging
import csv


class HSVOverexposureDetector:
    """基于HSV色彩空间的过曝检测器（重构版）"""
    
    def __init__(self, 
                 grid_size: int = 30,
                 v_threshold: int = 230,
                 s_threshold: float = 0.3,
                 overexpose_threshold: float = 0.5,
                 min_region_size: int = 9,
                 verbose: bool = False):
        """
        初始化检测器
        
        Args:
            grid_size: 网格大小（像素）
            v_threshold: V通道阈值 (0-255)
            s_threshold: S通道阈值 (0-1)
            overexpose_threshold: 网格过曝像素比例阈值 (0-1)
            min_region_size: 最小过曝区域大小（相邻网格数）
            verbose: 是否输出详细日志和保存中间结果
        """
        self.grid_size = grid_size
        self.v_threshold = v_threshold
        self.s_threshold = s_threshold
        self.overexpose_threshold = overexpose_threshold
        self.min_region_size = min_region_size
        self.verbose = verbose
        self.step_name = ['preprocess', 'divide_into_grids', 'normalize_exclude_regions', 'filter_valid_grids', 'calculate_grid_hsv_stats', 'identify_overexposed_grids', 'find_overexposed_regions', 'draw_results', 'save_analysis_images']
        
        # 时间统计
        self.step_times = {}
        
        # 设置日志
        self._setup_logger()
        
    def _setup_logger(self):
        """设置日志记录器"""
        # 创建新的logger实例，避免重复
        logger_name = f'HSVOverexposureDetector_{id(self)}'
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        # 清除所有现有的处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 控制台输出
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG if self.verbose else logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def _setup_output_dirs(self, save_dir: str, image_name: str):
        """设置输出目录"""
        self.results_dir = os.path.join(save_dir, 'results')
        self.analysis_dir = os.path.join(save_dir, 'analysis')
        
        os.makedirs(self.results_dir, exist_ok=True)
        if self.verbose:
            os.makedirs(self.analysis_dir, exist_ok=True)
            
            # 设置日志文件 - 使用图像名称
            log_file = os.path.join(self.analysis_dir, f'{image_name}_detection.log')
            # 清理已存在的文件处理器
            for handler in self.logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    self.logger.removeHandler(handler)
                    
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        self.logger.info(f"输出目录设置完成: results={self.results_dir}, analysis={self.analysis_dir}")
    
    def _save_results_to_csv(self, save_dir: str, image_name: str, result: Dict, 
                           step_times: Dict, grid_stats: Dict):
        """保存检测结果到CSV文件"""
        csv_file = os.path.join(save_dir, 'detection_results.csv')
        
        # 检查文件是否存在，如果不存在则写入表头
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头
            if not file_exists:
                headers = [
                    'image_name', 'detected', 'num_regions', 'processing_time',
                    'step1_preprocess_time', 'step1_preprocess_percent',
                    'step2_grid_time', 'step2_grid_percent',
                    'step3_normalize_time', 'step3_normalize_percent',
                    'step4_filter_time', 'step4_filter_percent',
                    'step5_hsv_stats_time', 'step5_hsv_stats_percent',
                    'step6_identify_time', 'step6_identify_percent',
                    'step7_regions_time', 'step7_regions_percent',
                    'step8_draw_time', 'step8_draw_percent',
                    'step9_analysis_time', 'step9_analysis_percent',
                    'total_grids', 'valid_grids', 'overexposed_grids',
                    'regions_detail'
                ]
                writer.writerow(headers)
            
            # 计算时间占比
            total_time = result['processing_time']
            time_percents = {}
            for step, step_time in step_times.items():
                time_percents[step] = (step_time / total_time * 100) if total_time > 0 else 0
            
            # 统计网格信息
            total_grids = len(grid_stats) if hasattr(self, '_total_grids') else 0
            valid_grids = len(grid_stats)
            overexposed_grids = sum(1 for stats in grid_stats.values() 
                                  if stats['overexpose_ratio'] >= self.overexpose_threshold)
            
            # 格式化区域详情
            regions_detail = '; '.join([
                f"Region{i+1}:bbox{region['bbox']},grids{region['grid_count']}"
                for i, region in enumerate(result['regions'])
            ]) if result['regions'] else 'None'
            
            # 写入数据行
            row = [
                image_name,
                result['detected'],
                result['num_regions'],
                f"{result['processing_time']:.3f}",
                f"{step_times.get('step1', 0):.3f}",
                f"{time_percents.get('step1', 0):.1f}%",
                f"{step_times.get('step2', 0):.3f}",
                f"{time_percents.get('step2', 0):.1f}%",
                f"{step_times.get('step3', 0):.3f}",
                f"{time_percents.get('step3', 0):.1f}%",
                f"{step_times.get('step4', 0):.3f}",
                f"{time_percents.get('step4', 0):.1f}%",
                f"{step_times.get('step5', 0):.3f}",
                f"{time_percents.get('step5', 0):.1f}%",
                f"{step_times.get('step6', 0):.3f}",
                f"{time_percents.get('step6', 0):.1f}%",
                f"{step_times.get('step7', 0):.3f}",
                f"{time_percents.get('step7', 0):.1f}%",
                f"{step_times.get('step8', 0):.3f}",
                f"{time_percents.get('step8', 0):.1f}%",
                f"{step_times.get('step9', 0):.3f}",
                f"{time_percents.get('step9', 0):.1f}%",
                self._total_grids if hasattr(self, '_total_grids') else 0,
                valid_grids,
                overexposed_grids,
                regions_detail
            ]
            writer.writerow(row)
    
    def detect(self, 
               image_path: str,
               exclude_regions: List[List[int]] = None,
               save_dir: str = './output') -> Dict:
        """
        主检测方法
        
        Args:
            image_path: 输入图像路径
            exclude_regions: 豁免检测区域列表 [[x1,y1,x2,y2], ...]，支持负数坐标
            save_dir: 输出目录
            
        Returns:
            检测结果字典
        """
        # 获取图像名称
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        self.logger.info(f"开始检测图像: {image_path}")
        start_time = time.time()
        
        # 设置输出目录
        self._setup_output_dirs(save_dir, base_name)
        
        # 1. 读取并预处理图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        h, w = image.shape[:2]
        self.logger.info(f"图像尺寸: {w}x{h}")
        
        step_start = time.time()
        hsv_image = self._preprocess_image(image)
        self.step_times['step1'] = time.time() - step_start
        
        # 2. 划分网格
        step_start = time.time()
        grid_info = self._divide_into_grids(w, h)
        self._total_grids = len(grid_info)  # 保存总网格数
        self.step_times['step2'] = time.time() - step_start
        
        # 3. 处理豁免检测区域
        step_start = time.time()
        exclude_regions_normalized = self._normalize_exclude_regions(exclude_regions, w, h)
        self.step_times['step3'] = time.time() - step_start
        
        # 4. 过滤出有效网格
        step_start = time.time()
        valid_grids = self._filter_valid_grids(grid_info, exclude_regions_normalized)
        self.step_times['step4'] = time.time() - step_start
        
        # 5. 计算每个有效网格的HSV统计
        step_start = time.time()
        grid_hsv_stats = self._calculate_grid_hsv_stats(hsv_image, valid_grids)
        self.step_times['step5'] = time.time() - step_start
        
        # 6. 判定过曝网格
        step_start = time.time()
        overexposed_grids = self._identify_overexposed_grids(grid_hsv_stats)
        self.step_times['step6'] = time.time() - step_start
        
        # 7. 找出过曝区域
        step_start = time.time()
        overexposed_regions = self._find_overexposed_regions(overexposed_grids, grid_info)
        self.step_times['step7'] = time.time() - step_start
        
        # 8. 绘制结果
        step_start = time.time()
        result_image = self._draw_results(image.copy(), overexposed_regions)
        self.step_times['step8'] = time.time() - step_start
        
        # 保存结果图像
        result_path = os.path.join(self.results_dir, f"{base_name}_result.jpg")
        cv2.imwrite(result_path, result_image)
        self.logger.info(f"结果图像已保存: {result_path}")
        
        # 9. 如果verbose=True，保存中间分析图像
        step_start = time.time()
        if self.verbose:
            self._save_analysis_images(image, hsv_image, grid_info, valid_grids, 
                                     grid_hsv_stats, overexposed_grids, 
                                     exclude_regions_normalized, base_name)
        self.step_times['step9'] = time.time() - step_start
        
        # 构建返回结果
        end_time = time.time()
        result = {
            'detected': len(overexposed_regions) > 0,
            'num_regions': len(overexposed_regions),
            'regions': overexposed_regions,
            'processing_time': end_time - start_time,
            'result_path': result_path,
            'step_times': self.step_times.copy()
        }
        
        # 保存到CSV文件
        self._save_results_to_csv(save_dir, base_name, result, self.step_times, grid_hsv_stats)
        
        # 输出时间统计
        self.logger.info(f"检测完成，总耗时: {result['processing_time']:.3f}秒")
        self.logger.info(f"检测到 {result['num_regions']} 个过曝区域")
        
        if self.verbose:
            total_time = result['processing_time']
            self.logger.info("各步骤耗时统计:")
            for i, (step, step_time) in enumerate(self.step_times.items(), 1):
                percent = (step_time / total_time * 100) if total_time > 0 else 0
                self.logger.info(f"  步骤{self.step_name[i-1]}: {step_time*1000:.3f}ms ({percent:.1f}%)")
        
        return result
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """步骤1: BGR转HSV"""
        self.logger.debug("步骤1: 图像预处理 BGR->HSV")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return hsv
    
    def _divide_into_grids(self, width: int, height: int) -> Dict[Tuple[int, int], Tuple[int, int, int, int]]:
        """
        步骤2: 划分网格
        
        Returns:
            网格字典 {(row, col): (x1, y1, x2, y2)}
        """
        self.logger.debug(f"步骤2: 划分网格 (网格大小: {self.grid_size})")
        
        grid_info = {}
        rows = (height + self.grid_size - 1) // self.grid_size
        cols = (width + self.grid_size - 1) // self.grid_size
        
        for row in range(rows):
            for col in range(cols):
                x1 = col * self.grid_size
                y1 = row * self.grid_size
                x2 = min((col + 1) * self.grid_size, width)
                y2 = min((row + 1) * self.grid_size, height)
                
                grid_info[(row, col)] = (x1, y1, x2, y2)
        
        self.logger.debug(f"网格数量: {rows}x{cols} = {len(grid_info)}")
        return grid_info
    
    def _normalize_exclude_regions(self, exclude_regions: List[List[int]], 
                                  width: int, height: int) -> List[List[int]]:
        """
        步骤3: 规范化豁免检测区域（处理负数坐标）
        """
        if not exclude_regions:
            return []
            
        self.logger.debug(f"步骤3: 处理豁免检测区域 (数量: {len(exclude_regions)})")
        
        normalized_regions = []
        for region in exclude_regions:
            x1, y1, x2, y2 = region
            
            # 处理负数坐标
            if x1 < 0:
                x1 = width + x1
            if y1 < 0:
                y1 = height + y1
            if x2 < 0:
                x2 = width + x2
            if y2 < 0:
                y2 = height + y2
                
            # 确保坐标有效
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            # 确保x1<x2, y1<y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
                
            normalized_regions.append([x1, y1, x2, y2])
            self.logger.debug(f"  豁免区域: [{x1}, {y1}, {x2}, {y2}]")
            
        return normalized_regions
    
    def _is_grid_in_exclude_region(self, grid_bbox: Tuple[int, int, int, int], 
                                  exclude_regions: List[List[int]]) -> bool:
        """判断网格是否在豁免区域内"""
        if not exclude_regions:
            return False
            
        gx1, gy1, gx2, gy2 = grid_bbox
        grid_center_x = (gx1 + gx2) // 2
        grid_center_y = (gy1 + gy2) // 2
        
        for ex1, ey1, ex2, ey2 in exclude_regions:
            # 检查网格中心是否在豁免区域内
            if ex1 <= grid_center_x <= ex2 and ey1 <= grid_center_y <= ey2:
                return True
                
        return False
    
    def _filter_valid_grids(self, grid_info: Dict, exclude_regions: List[List[int]]) -> Dict:
        """
        步骤4: 过滤出有效网格（不在豁免区域内的网格）
        """
        self.logger.debug("步骤4: 过滤有效网格")
        
        valid_grids = {}
        excluded_count = 0
        
        for grid_pos, grid_bbox in grid_info.items():
            if not self._is_grid_in_exclude_region(grid_bbox, exclude_regions):
                valid_grids[grid_pos] = grid_bbox
            else:
                excluded_count += 1
                
        self.logger.debug(f"有效网格数量: {len(valid_grids)}, 排除网格数量: {excluded_count}")
        return valid_grids
    
    def _calculate_grid_hsv_stats(self, hsv_image: np.ndarray, 
                                 valid_grids: Dict) -> Dict:
        """
        步骤5: 计算每个有效网格的HSV统计信息
        """
        self.logger.debug("步骤5: 计算网格HSV统计")
        
        grid_stats = {}
        
        for grid_pos, (x1, y1, x2, y2) in valid_grids.items():
            # 提取网格区域
            grid_region = hsv_image[y1:y2, x1:x2]
            
            # 分离HSV通道
            h_channel = grid_region[:, :, 0]
            s_channel = grid_region[:, :, 1] / 255.0  # 归一化到0-1
            v_channel = grid_region[:, :, 2]
            
            # 计算HSV直方图
            h_hist, _ = np.histogram(h_channel.flatten(), bins=18, range=(0, 180))
            s_hist, _ = np.histogram(s_channel.flatten(), bins=20, range=(0, 1))
            v_hist, _ = np.histogram(v_channel.flatten(), bins=26, range=(0, 255))
            
            # 计算过曝像素
            total_pixels = grid_region.shape[0] * grid_region.shape[1]
            overexposed_mask = (v_channel >= self.v_threshold) & (s_channel <= self.s_threshold)
            overexposed_pixels = np.sum(overexposed_mask)
            overexpose_ratio = overexposed_pixels / total_pixels if total_pixels > 0 else 0
            
            grid_stats[grid_pos] = {
                'bbox': (x1, y1, x2, y2),
                'h_hist': h_hist,
                's_hist': s_hist,
                'v_hist': v_hist,
                'overexpose_ratio': overexpose_ratio,
                'overexposed_pixels': overexposed_pixels,
                'total_pixels': total_pixels,
                'avg_h': np.mean(h_channel),
                'avg_s': np.mean(s_channel),
                'avg_v': np.mean(v_channel)
            }
            
        return grid_stats
    
    def _identify_overexposed_grids(self, grid_stats: Dict) -> Set[Tuple[int, int]]:
        """
        步骤6: 识别过曝网格
        """
        self.logger.debug(f"步骤6: 识别过曝网格 (阈值: {self.overexpose_threshold})")
        
        overexposed_grids = set()
        
        for grid_pos, stats in grid_stats.items():
            if stats['overexpose_ratio'] >= self.overexpose_threshold:
                overexposed_grids.add(grid_pos)
                if self.verbose:
                    self.logger.debug(f"  过曝网格 {grid_pos}: 比例={stats['overexpose_ratio']:.3f}")
                    
        self.logger.debug(f"过曝网格数量: {len(overexposed_grids)}")
        return overexposed_grids
    
    def _are_grids_adjacent(self, grid1: Tuple[int, int], grid2: Tuple[int, int]) -> bool:
        """判断两个网格是否相邻（共享顶点）"""
        r1, c1 = grid1
        r2, c2 = grid2
        
        # 检查是否共享顶点（包括对角相邻）
        return abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1
    
    def _find_connected_components(self, grids: Set[Tuple[int, int]]) -> List[Set[Tuple[int, int]]]:
        """使用深度优先搜索找出连通的网格组"""
        visited = set()
        components = []
        
        def dfs(grid, component):
            if grid in visited:
                return
            visited.add(grid)
            component.add(grid)
            
            # 检查所有可能的相邻网格
            r, c = grid
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    neighbor = (r + dr, c + dc)
                    if neighbor in grids and neighbor not in visited:
                        dfs(neighbor, component)
        
        for grid in grids:
            if grid not in visited:
                component = set()
                dfs(grid, component)
                components.append(component)
                
        return components
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], 
                      box2: Tuple[int, int, int, int]) -> float:
        """
        计算两个边界框的IoU（交并比）
        
        Args:
            box1: 边界框1 (x1, y1, x2, y2)
            box2: 边界框2 (x1, y1, x2, y2)
            
        Returns:
            IoU值 (0-1)
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集区域
        intersect_x1 = max(x1_1, x1_2)
        intersect_y1 = max(y1_1, y1_2)
        intersect_x2 = min(x2_1, x2_2)
        intersect_y2 = min(y2_1, y2_2)
        
        # 如果没有交集
        if intersect_x1 >= intersect_x2 or intersect_y1 >= intersect_y2:
            return 0.0
        
        # 计算交集面积
        intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
        
        # 计算并集面积
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersect_area
        
        # 计算IoU
        iou = intersect_area / union_area if union_area > 0 else 0.0
        return iou
    
    def _merge_overlapping_regions(self, regions: List[Dict]) -> List[Dict]:
        """
        合并有重叠的过曝区域
        
        Args:
            regions: 原始过曝区域列表
            
        Returns:
            合并后的过曝区域列表
        """
        if not regions:
            return regions
            
        self.logger.debug(f"合并前区域数量: {len(regions)}")
        
        merged_regions = []
        used_indices = set()
        
        for i, region1 in enumerate(regions):
            if i in used_indices:
                continue
                
            # 初始化合并区域
            merged_bbox = list(region1['bbox'])
            merged_grids = set(region1['grids'])
            merged_indices = {i}
            
            # 检查与其他区域的重叠
            for j, region2 in enumerate(regions):
                if j <= i or j in used_indices:
                    continue
                    
                iou = self._calculate_iou(region1['bbox'], region2['bbox'])
                
                if iou > 0:  # 有重叠就合并
                    self.logger.debug(f"合并区域 {i} 和 {j}, IoU: {iou:.3f}")
                    
                    # 扩展边界框
                    merged_bbox[0] = min(merged_bbox[0], region2['bbox'][0])  # min_x
                    merged_bbox[1] = min(merged_bbox[1], region2['bbox'][1])  # min_y
                    merged_bbox[2] = max(merged_bbox[2], region2['bbox'][2])  # max_x
                    merged_bbox[3] = max(merged_bbox[3], region2['bbox'][3])  # max_y
                    
                    # 合并网格
                    merged_grids.update(region2['grids'])
                    merged_indices.add(j)
            
            # 标记已使用的索引
            used_indices.update(merged_indices)
            
            # 创建合并后的区域
            merged_region = {
                'bbox': tuple(merged_bbox),
                'grid_count': len(merged_grids),
                'grids': list(merged_grids)
            }
            merged_regions.append(merged_region)
            
            if self.verbose and len(merged_indices) > 1:
                self.logger.debug(f"  合并区域: bbox={merged_region['bbox']}, 网格数={merged_region['grid_count']}")
        
        self.logger.debug(f"合并后区域数量: {len(merged_regions)}")
        return merged_regions
    
    def _find_overexposed_regions(self, overexposed_grids: Set[Tuple[int, int]], 
                                 grid_info: Dict) -> List[Dict]:
        """
        步骤7: 找出过曝区域（相邻网格组成的区域）并合并重叠区域
        """
        self.logger.debug(f"步骤7: 查找过曝区域 (最小区域大小: {self.min_region_size})")
        
        # 找出连通的网格组
        connected_components = self._find_connected_components(overexposed_grids)
        
        regions = []
        for component in connected_components:
            if len(component) >= self.min_region_size:
                # 计算区域边界
                min_x = float('inf')
                min_y = float('inf')
                max_x = 0
                max_y = 0
                
                for grid_pos in component:
                    x1, y1, x2, y2 = grid_info[grid_pos]
                    min_x = min(min_x, x1)
                    min_y = min(min_y, y1)
                    max_x = max(max_x, x2)
                    max_y = max(max_y, y2)
                
                region = {
                    'bbox': (min_x, min_y, max_x, max_y),
                    'grid_count': len(component),
                    'grids': list(component)
                }
                regions.append(region)
                
                if self.verbose:
                    self.logger.debug(f"  初始过曝区域: bbox={region['bbox']}, 网格数={region['grid_count']}")
        
        # 合并有重叠的区域
        merged_regions = self._merge_overlapping_regions(regions)
        merged_regions = self._merge_overlapping_regions(merged_regions)
        
        self.logger.debug(f"最终找到 {len(merged_regions)} 个过曝区域")
        return merged_regions
    
    def _draw_results(self, image: np.ndarray, regions: List[Dict]) -> np.ndarray:
        """
        步骤8: 在原图上绘制检测结果
        """
        self.logger.debug("步骤8: 绘制检测结果")
        
        for i, region in enumerate(regions):
            x1, y1, x2, y2 = region['bbox']
            
            # 绘制红色矩形框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # 添加标签
            # label = f"Overexposed-{i+1}"
            # cv2.putText(image, label, (x1, y1-10), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return image
    
    def _save_analysis_images(self, original_image: np.ndarray, hsv_image: np.ndarray,
                             grid_info: Dict, valid_grids: Dict, grid_stats: Dict,
                             overexposed_grids: Set, exclude_regions: List,
                             base_name: str):
        """
        步骤9: 保存分析图像（仅在verbose=True时）
        """
        self.logger.debug("步骤9: 生成并保存分析图像")
        
        # 9a. HSV分析图
        hsv_analysis = self._create_hsv_analysis_image(
            original_image, grid_info, valid_grids, grid_stats, exclude_regions)
        hsv_analysis_path = os.path.join(self.analysis_dir, f"{base_name}_hsv_analysis.jpg")
        cv2.imwrite(hsv_analysis_path, hsv_analysis)
        self.logger.debug(f"  HSV分析图已保存: {hsv_analysis_path}")
        
        # 9b. 过曝网格图
        overexposed_grid_image = self._create_overexposed_grid_image(
            original_image, grid_info, overexposed_grids, grid_stats, exclude_regions)
        grid_image_path = os.path.join(self.analysis_dir, f"{base_name}_overexposure_grids.jpg")
        cv2.imwrite(grid_image_path, overexposed_grid_image)
        self.logger.debug(f"  过曝网格图已保存: {grid_image_path}")
    
    def _create_hsv_analysis_image(self, image: np.ndarray, grid_info: Dict,
                                  valid_grids: Dict, grid_stats: Dict,
                                  exclude_regions: List) -> np.ndarray:
        """创建HSV分析图像"""
        result = image.copy()
        h, w = result.shape[:2]
        
        # 绘制网格线（灰色）
        for (row, col), (x1, y1, x2, y2) in grid_info.items():
            cv2.rectangle(result, (x1, y1), (x2, y2), (128, 128, 128), 1)
        
        # 填充豁免区域（白色）
        for (row, col), bbox in grid_info.items():
            if (row, col) not in valid_grids:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(result, (x1+1, y1+1), (x2-1, y2-1), (255, 255, 255), -1)
        
        # 在每个有效网格中绘制HSV曲线
        for grid_pos, stats in grid_stats.items():
            x1, y1, x2, y2 = stats['bbox']
            grid_w = x2 - x1
            grid_h = y2 - y1
            
            # 根据grid_size调整绘制条件，确保有足够空间
            min_size = max(self.grid_size // 2, 10)  # 至少10像素
            if grid_w > min_size and grid_h > min_size:
                # 绘制V曲线（蓝色）
                v_hist = stats['v_hist']
                if len(v_hist) > 0 and np.max(v_hist) > 0:
                    v_normalized = (v_hist / np.max(v_hist) * (grid_h * 0.8)).astype(int)
                    for i in range(len(v_hist) - 1):
                        pt1_x = x1 + int(i * grid_w / len(v_hist))
                        pt1_y = y2 - 5 - v_normalized[i]
                        pt2_x = x1 + int((i + 1) * grid_w / len(v_hist))
                        pt2_y = y2 - 5 - v_normalized[i + 1]
                        cv2.line(result, (pt1_x, pt1_y), (pt2_x, pt2_y), (255, 0, 0), 1)
                
                # 绘制S曲线（绿色）
                s_hist = stats['s_hist']
                if len(s_hist) > 0 and np.max(s_hist) > 0:
                    s_normalized = (s_hist / np.max(s_hist) * (grid_h * 0.8)).astype(int)
                    for i in range(len(s_hist) - 1):
                        pt1_x = x1 + int(i * grid_w / len(s_hist))
                        pt1_y = y2 - 5 - s_normalized[i]
                        pt2_x = x1 + int((i + 1) * grid_w / len(s_hist))
                        pt2_y = y2 - 5 - s_normalized[i + 1]
                        cv2.line(result, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 0), 1)
        
        # 添加图例
        legend_x = 20
        legend_y = 30
        cv2.rectangle(result, (legend_x-5, legend_y-20), (legend_x+150, legend_y+40), 
                     (255, 255, 255), -1)
        cv2.rectangle(result, (legend_x-5, legend_y-20), (legend_x+150, legend_y+40), 
                     (0, 0, 0), 1)
        
        cv2.line(result, (legend_x, legend_y), (legend_x+30, legend_y), (255, 0, 0), 2)
        cv2.putText(result, "V (Value)", (legend_x+35, legend_y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.line(result, (legend_x, legend_y+20), (legend_x+30, legend_y+20), (0, 255, 0), 2)
        cv2.putText(result, "S (Saturation)", (legend_x+35, legend_y+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return result
    
    def _create_overexposed_grid_image(self, image: np.ndarray, grid_info: Dict,
                                      overexposed_grids: Set, grid_stats: Dict,
                                      exclude_regions: List) -> np.ndarray:
        """创建过曝网格图像"""
        result = image.copy()
        
        # 绘制所有网格线（灰色）
        for (row, col), (x1, y1, x2, y2) in grid_info.items():
            cv2.rectangle(result, (x1, y1), (x2, y2), (128, 128, 128), 1)
        
        # 填充豁免区域（白色）
        for (row, col), bbox in grid_info.items():
            if self._is_grid_in_exclude_region(bbox, exclude_regions):
                x1, y1, x2, y2 = bbox
                cv2.rectangle(result, (x1+1, y1+1), (x2-1, y2-1), (255, 255, 255), -1)
        
        # 显示所有有效网格的过曝比例
        for grid_pos, stats in grid_stats.items():
            x1, y1, x2, y2 = stats['bbox']
            ratio = stats['overexpose_ratio']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 计算合适的字体大小
            font_scale = min(self.grid_size / 50.0, 0.3)  # 根据网格大小调整字体
            
            text = f"{ratio:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2
            
            # 如果是过曝网格，绘制红色边框和红色文字
            if grid_pos in overexposed_grids:
                # 绘制红色边框
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 绘制红色文字
                cv2.putText(result, text, (text_x, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
            # else:
            #     # 非过曝网格，绘制紫色文字
            #     cv2.putText(result, text, (text_x, text_y), 
            #               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (128, 0, 128), 1)
        
        return result

