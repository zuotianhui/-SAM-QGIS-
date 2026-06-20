# -*- coding: utf-8 -*-
"""
/***************************************************************************
 SAM2 Segmentation Plugin - 工具类和辅助函数
                                 A QGIS plugin
 SAM2-based interactive segmentation plugin for QGIS
                             -------------------
        begin                : 2025-11-04
        copyright            : (C) 2025 by Your Name
        email                : your@email.com
 ***************************************************************************/
"""

import numpy as np
# 修复：添加QTextEdit导入
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QColor, QPainter, QPen, QFont, QBrush, QCursor, QPixmap
from qgis.gui import QgsMapToolEmitPoint, QgsMapCanvasItem
from qgis.core import QgsWkbTypes, QgsMessageLog, Qgis
from datetime import datetime
import rasterio
from affine import Affine
from skimage.transform import resize
from rasterio.features import shapes
from shapely.geometry import shape


class MapClickTool(QgsMapToolEmitPoint):
    """自定义地图点击工具（支持动态设置光标）"""
    def __init__(self, canvas):
        QgsMapToolEmitPoint.__init__(self, canvas)
        self.canvas = canvas
        self.current_cursor = Qt.CrossCursor
        self.setCursor(self.current_cursor)

    def set_custom_cursor(self, cursor):
        """设置自定义光标"""
        self.current_cursor = cursor
        self.setCursor(self.current_cursor)

    def canvasReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            point = self.toMapCoordinates(event.pos())
            self.canvasClicked.emit(point, event.button())


class LegendItem(QgsMapCanvasItem):
    """右上角图例（支持中文+彩色光标）"""
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.setZValue(1000)  # 最上层显示
        self.font = QFont("微软雅黑", 10, QFont.Bold)
        self.normal_font = QFont("微软雅黑", 9)

    def boundingRect(self):
        """图例边界（150x80px）"""
        return QRectF(0, 0, 150, 80)

    def paint(self, painter, option, widget):
        """绘制图例"""
        # 白色半透明背景+黑色边框
        painter.setBrush(QBrush(QColor(255, 255, 255, 240)))
        painter.setPen(QPen(QColor(0, 0, 0), 1.5))
        painter.drawRect(self.boundingRect())
        
        # 标题
        painter.setFont(self.font)
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(15, 25, "标注样式说明")
        
        # 前景标注（绿色圆形）
        painter.setPen(QPen(QColor(76, 175, 80), 2))
        painter.setBrush(QBrush(QColor(76, 175, 80, 200)))
        painter.drawEllipse(15, 35, 8, 8)
        painter.setFont(self.normal_font)
        painter.drawText(35, 42, "前景标注")
        
        # 背景标注（红色叉形）
        painter.setPen(QPen(QColor(244, 67, 54), 2))
        painter.drawLine(15, 55, 23, 63)
        painter.drawLine(23, 55, 15, 63)
        painter.setFont(self.normal_font)
        painter.drawText(35, 62, "背景标注")

    @staticmethod
    def create_custom_cursor(shape_type):
        """创建彩色自定义光标（兼容所有QGIS版本）"""
        size = 12  # 光标大小
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)  # 透明背景
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        
        if shape_type == "foreground":
            # 绿色圆形
            painter.setPen(QPen(QColor(76, 175, 80), 2))
            painter.setBrush(QBrush(QColor(76, 175, 80, 180)))
            painter.drawEllipse(1, 1, size-2, size-2)
        elif shape_type == "background":
            # 红色叉形
            painter.setPen(QPen(QColor(244, 67, 54), 2))
            painter.drawLine(1, 1, size-2, size-2)
            painter.drawLine(size-2, 1, 1, size-2)
        
        painter.end()
        return QCursor(pixmap, size//2, size//2)


def log_message(log_text_widget, message):
    """日志输出（插件日志+QGIS日志）"""
    # 输出到插件日志控件
    if log_text_widget and isinstance(log_text_widget, QTextEdit):
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_text_widget.append(f"[{time_str}] {message}")
        log_text_widget.verticalScrollBar().setValue(
            log_text_widget.verticalScrollBar().maximum()
        )
    # 输出到QGIS日志面板
    QgsMessageLog.logMessage(f"SAM2 Segmentation: {message}", "SAM2 Plugin", Qgis.Info)


def map_to_pixel_coords(map_point, image_extent, pixel_width, pixel_height, image_crs, canvas_crs, project):
    """地图坐标转像素坐标"""
    try:
        from qgis.core import QgsCoordinateTransform

        if (image_extent is None or pixel_width is None or pixel_height is None or
            abs(pixel_width) < 1e-12 or abs(pixel_height) < 1e-12):
            return None
        
        # CRS转换
        transform = QgsCoordinateTransform(canvas_crs, image_crs, project)
        try:
            transformed_point = transform.transform(map_point)
        except:
            return None
        
        # 检查是否在图像范围内
        if not image_extent.contains(transformed_point):
            return None
        
        # 计算像素坐标
        x = int((transformed_point.x() - image_extent.xMinimum()) / pixel_width)
        y = int((image_extent.yMaximum() - transformed_point.y()) / abs(pixel_height))
        
        return (x, y)
    except Exception:
        return None


def stretch_percentile(img, lower=2, upper=98):
    """对比度拉伸"""
    img = img.astype(np.float32)
    p_low, p_high = np.percentile(img, (lower, upper))
    
    if p_high - p_low < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
        
    stretched = np.clip((img - p_low) / (p_high - p_low) * 255, 0, 255)
    return stretched.astype(np.uint8)


def process_mask(mask, mask_params):
    """Mask后处理（平滑、填充、形态学操作）"""
    try:
        # 平滑
        if mask_params["mask_smoothing"] and mask_params["smoothing_kernel_size"] > 1:
            from scipy.ndimage import gaussian_filter
            mask = gaussian_filter(mask.astype(float), sigma=mask_params["smoothing_kernel_size"] / 6)
            mask = (mask > 0.5).astype(np.uint8)
    except ImportError:
        log_message(None, "警告：scipy未安装，无法进行掩码平滑")
    
    try:
        # 形态学操作
        if mask_params["dilate_iterations"] > 0 or mask_params["erode_iterations"] > 0:
            from scipy.ndimage import binary_dilate, binary_erosion
            kernel = np.ones((3, 3), dtype=np.uint8)
            if mask_params["dilate_iterations"] > 0:
                mask = binary_dilate(mask, kernel, iterations=mask_params["dilate_iterations"])
            if mask_params["erode_iterations"] > 0:
                mask = binary_erosion(mask, kernel, iterations=mask_params["erode_iterations"])
    except ImportError:
        log_message(None, "警告：scipy未安装，无法进行形态学操作")
    
    try:
        # 孔洞填充
        if mask_params["hole_filling"]:
            from scipy.ndimage import binary_fill_holes
            mask = binary_fill_holes(mask, structure=np.ones((3, 3))).astype(np.uint8)
    except ImportError:
        log_message(None, "警告：scipy未安装，无法进行孔洞填充")
    
    return mask


def mask_to_geometries(mask, original_shape, image_extent, current_layer, vector_params, map_crs):
    """掩码转QGIS几何对象（修复：新增map_crs参数，避免错误获取画布）"""
    from qgis.core import QgsGeometry, QgsCoordinateTransform, QgsProject

    # 上采样到原始尺寸
    upsampled_mask = resize(
        mask, original_shape, order=0, preserve_range=True
    ).astype(np.uint8)
    
    # 原始分辨率
    original_pixel_width = current_layer.rasterUnitsPerPixelX()
    original_pixel_height = current_layer.rasterUnitsPerPixelY()
    
    # 地理变换
    transform = Affine(
        original_pixel_width, 0, image_extent.xMinimum(),
        0, -abs(original_pixel_height), image_extent.yMaximum()
    )
    
    # 生成矢量要素
    results = (
        {'geometry': shape(s), 'properties': {}}
        for s, v in shapes(upsampled_mask, mask=upsampled_mask, transform=transform)
    )
    
    qgis_geoms = []
    image_crs = current_layer.crs()
    # 直接使用传入的map_crs（从主对话框传递的正确画布CRS）
    crs_transform = QgsCoordinateTransform(image_crs, map_crs, QgsProject.instance())
    
    for feature in results:
        geom = feature['geometry']
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            # 几何简化
            if vector_params["simplify_geometry"]:
                geom = geom.simplify(vector_params["simplify_tolerance"], preserve_topology=True)
            
            # 缓冲
            if vector_params["buffer_distance"] > 0:
                geom = geom.buffer(vector_params["buffer_distance"])
            
            # 面积过滤
            if geom.area < vector_params["min_polygon_area"]:
                continue
            
            # 转换为QGIS几何并变换CRS
            qgis_geom = QgsGeometry.fromWkt(geom.wkt)
            qgis_geom.transform(crs_transform)
            qgis_geoms.append(qgis_geom)
    
    return qgis_geoms