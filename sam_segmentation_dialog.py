# -*- coding: utf-8 -*-
"""
/***************************************************************************
 SamSegmentationDialog
                                 A QGIS plugin
 SAM2-based interactive segmentation plugin for QGIS
                             -------------------
        begin                : 2025-11-04
        copyright            : (C) 2025 by Your Name
        email                : your@email.com
 ***************************************************************************/
"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import torch
from PyQt5.QtWidgets import (QDialog, QFileDialog, QMessageBox, QSpinBox,
                            QDoubleSpinBox, QCheckBox, QTextEdit)
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QColor, QCursor, QPixmap, QPainter
import geopandas as gpd

# SAM2导入（兼容不同安装方式）
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    from sam2 import build_sam2, SAM2ImagePredictor

# QGIS核心导入
from qgis.core import (QgsProject, QgsRasterLayer, QgsVectorLayer, 
                      QgsPointXY, QgsGeometry, QgsCoordinateTransform,
                      QgsCoordinateReferenceSystem, QgsWkbTypes, QgsRectangle,
                      QgsSettings)
from qgis.gui import QgsRubberBand, QgsMapCanvasItem

# 导入UI和工具类
from .ui_sam_segmentation_dialog_base import Ui_SamSegmentationDialogBase
from .sam_segmentation_utils import (
    MapClickTool, LegendItem, log_message, map_to_pixel_coords,
    stretch_percentile, process_mask, mask_to_geometries
)

# 中文显示支持
try:
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
except:
    pass


class SamSegmentationDialog(QDialog, Ui_SamSegmentationDialogBase):
    def __init__(self, iface, parent=None):
        super(SamSegmentationDialog, self).__init__(parent)
        self.iface = iface
        self.setupUi(self)  # 初始化UI
        
        # 关键修复：设置插件窗口标题
        self.setWindowTitle("SAM2影像自动分割标注工具")
        
        # 插件路径和配置
        self.plugin_dir = os.path.dirname(__file__)
        self.config_path = os.path.join(self.plugin_dir, "config.xml")
        
        # 模型和数据变量
        self.model_path = ""
        self.sam2_config_path = ""
        self.model_type = "vit_l"
        self.sam_predictor = None
        self.current_layer = None
        self.image_data = None
        self.image_extent = None
        self.pixel_width = None
        self.pixel_height = None
        self.image_crs = None
        
        # 采样参数
        self.downsample_ratio = 1.0
        self.original_shape = None
        self.max_size = 2000
        
        # 交互数据
        self.click_history = []
        self.masks = None
        self.click_mode = 1  # 1=前景，0=背景
        
        # 工具对象
        self.point_tool = None
        self.rubber_band_mask = None
        self.foreground_points = None
        self.background_points = None
        self.legend_item = None
        
        # 预创建光标
        self.foreground_cursor = None
        self.background_cursor = None
        
        # 核心参数配置
        self.segment_params = {
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
            "stability_score_offset": 1.0,
            "box_nms_thresh": 0.7,
            "crop_n_layers": 0,
            "crop_nms_thresh": 0.7,
            "crop_overlap_ratio": 512 / 1500,
            "crop_bbox_temp": 0.1,
            "crop_mask_temp": 0.01,
            "point_grids": None,
            "min_mask_region_area": 0,
            "output_mode": "binary_mask",
            "multiscale_mode": False,
            "max_num_masks": 1000
        }
        
        self.mask_params = {
            "mask_smoothing": True,
            "smoothing_kernel_size": 3,
            "hole_filling": True,
            "min_hole_area": 10,
            "dilate_iterations": 1,
            "erode_iterations": 0
        }
        
        self.vector_params = {
            "simplify_geometry": True,
            "simplify_tolerance": 0.001,
            "keep_multipart": False,
            "min_polygon_area": 10.0,
            "buffer_distance": 0.0
        }
        
        # 初始化组件
        self.init_param_widgets()
        self.init_drawing_tools()
        self.init_cursors()
        self.create_legend()
        self.connect_signals()
        
        # 初始化图层选择和配置
        self.update_layer_combo()
        self.load_config()
        
        # 初始化日志
        log_message(self.logText, "SAM2插件已初始化。请选择栅格图层、加载模型开始操作（参数可通过「参数设置」按钮配置）")

    def init_param_widgets(self):
        """初始化参数设置页控件"""
        # SAM2分割参数
        self.pred_iou_spin = QDoubleSpinBox()
        self.pred_iou_spin.setRange(0.0, 1.0)
        self.pred_iou_spin.setSingleStep(0.01)
        self.pred_iou_spin.setValue(self.segment_params["pred_iou_thresh"])
        self.segment_layout.addRow("预测IOU阈值:", self.pred_iou_spin)
        
        self.stability_score_spin = QDoubleSpinBox()
        self.stability_score_spin.setRange(0.0, 1.0)
        self.stability_score_spin.setSingleStep(0.01)
        self.stability_score_spin.setValue(self.segment_params["stability_score_thresh"])
        self.segment_layout.addRow("稳定性分数阈值:", self.stability_score_spin)
        
        self.stability_offset_spin = QDoubleSpinBox()
        self.stability_offset_spin.setRange(0.0, 5.0)
        self.stability_offset_spin.setSingleStep(0.1)
        self.stability_offset_spin.setValue(self.segment_params["stability_score_offset"])
        self.segment_layout.addRow("稳定性分数偏移:", self.stability_offset_spin)
        
        self.box_nms_spin = QDoubleSpinBox()
        self.box_nms_spin.setRange(0.0, 1.0)
        self.box_nms_spin.setSingleStep(0.01)
        self.box_nms_spin.setValue(self.segment_params["box_nms_thresh"])
        self.segment_layout.addRow("Box NMS阈值:", self.box_nms_spin)
        
        self.min_mask_area_spin = QSpinBox()
        self.min_mask_area_spin.setRange(0, 1000)
        self.min_mask_area_spin.setSingleStep(10)
        self.min_mask_area_spin.setValue(self.segment_params["min_mask_region_area"])
        self.segment_layout.addRow("最小掩码面积:", self.min_mask_area_spin)
        
        self.multiscale_check = QCheckBox()
        self.multiscale_check.setChecked(self.segment_params["multiscale_mode"])
        self.segment_layout.addRow("多尺度模式:", self.multiscale_check)
        
        self.max_masks_spin = QSpinBox()
        self.max_masks_spin.setRange(1, 10000)
        self.max_masks_spin.setSingleStep(100)
        self.max_masks_spin.setValue(self.segment_params["max_num_masks"])
        self.segment_layout.addRow("最大掩码数量:", self.max_masks_spin)
        
        # Mask处理参数
        self.smoothing_check = QCheckBox()
        self.smoothing_check.setChecked(self.mask_params["mask_smoothing"])
        self.mask_layout.addRow("掩码平滑:", self.smoothing_check)
        
        self.smoothing_kernel_spin = QSpinBox()
        self.smoothing_kernel_spin.setRange(1, 15)
        self.smoothing_kernel_spin.setSingleStep(2)
        self.smoothing_kernel_spin.setValue(self.mask_params["smoothing_kernel_size"])
        self.mask_layout.addRow("平滑核大小:", self.smoothing_kernel_spin)
        
        self.hole_filling_check = QCheckBox()
        self.hole_filling_check.setChecked(self.mask_params["hole_filling"])
        self.mask_layout.addRow("孔洞填充:", self.hole_filling_check)
        
        self.min_hole_spin = QSpinBox()
        self.min_hole_spin.setRange(1, 1000)
        self.min_hole_spin.setSingleStep(10)
        self.min_hole_spin.setValue(self.mask_params["min_hole_area"])
        self.mask_layout.addRow("最小孔洞面积:", self.min_hole_spin)
        
        self.dilate_spin = QSpinBox()
        self.dilate_spin.setRange(0, 10)
        self.dilate_spin.setSingleStep(1)
        self.dilate_spin.setValue(self.mask_params["dilate_iterations"])
        self.mask_layout.addRow("膨胀迭代次数:", self.dilate_spin)
        
        self.erode_spin = QSpinBox()
        self.erode_spin.setRange(0, 10)
        self.erode_spin.setSingleStep(1)
        self.erode_spin.setValue(self.mask_params["erode_iterations"])
        self.mask_layout.addRow("腐蚀迭代次数:", self.erode_spin)
        
        # 矢量转换参数
        self.simplify_check = QCheckBox()
        self.simplify_check.setChecked(self.vector_params["simplify_geometry"])
        self.vector_layout.addRow("几何简化:", self.simplify_check)
        
        self.simplify_tol_spin = QDoubleSpinBox()
        self.simplify_tol_spin.setRange(0.0, 0.1)
        self.simplify_tol_spin.setSingleStep(0.001)
        self.simplify_tol_spin.setValue(self.vector_params["simplify_tolerance"])
        self.vector_layout.addRow("简化容差:", self.simplify_tol_spin)
        
        self.multipart_check = QCheckBox()
        self.multipart_check.setChecked(self.vector_params["keep_multipart"])
        self.vector_layout.addRow("保留多部件要素:", self.multipart_check)
        
        self.min_polygon_spin = QDoubleSpinBox()
        self.min_polygon_spin.setRange(0.1, 1000.0)
        self.min_polygon_spin.setSingleStep(1.0)
        self.min_polygon_spin.setValue(self.vector_params["min_polygon_area"])
        self.vector_layout.addRow("最小多边形面积:", self.min_polygon_spin)
        
        self.buffer_spin = QDoubleSpinBox()
        self.buffer_spin.setRange(0.0, 10.0)
        self.buffer_spin.setSingleStep(0.1)
        self.buffer_spin.setValue(self.vector_params["buffer_distance"])
        self.vector_layout.addRow("缓冲距离:", self.buffer_spin)

    def init_drawing_tools(self):
        """初始化绘图工具"""
        # 前景点（绿色圆形）
        self.foreground_points = QgsRubberBand(self.iface.mapCanvas(), QgsWkbTypes.PointGeometry)
        self.foreground_points.setColor(QColor(76, 175, 80, 255))
        self.foreground_points.setIconSize(10)
        self.foreground_points.setIcon(QgsRubberBand.ICON_CIRCLE)
        
        # 背景点（红色叉形）
        self.background_points = QgsRubberBand(self.iface.mapCanvas(), QgsWkbTypes.PointGeometry)
        self.background_points.setColor(QColor(244, 67, 54, 255))
        self.background_points.setIconSize(10)
        self.background_points.setIcon(QgsRubberBand.ICON_X)
        
        log_message(self.logText, "点绘制工具初始化完成")

    def init_cursors(self):
        """预初始化光标"""
        try:
            self.foreground_cursor = LegendItem.create_custom_cursor("foreground")
            self.background_cursor = LegendItem.create_custom_cursor("background")
            log_message(self.logText, "标注光标初始化成功")
        except Exception as e:
            self.foreground_cursor = Qt.PointingHandCursor
            self.background_cursor = Qt.ForbiddenCursor
            log_message(self.logText, f"自定义光标创建失败，使用系统默认光标: {str(e)}")
            QMessageBox.warning(self, "警告", f"自定义标注光标创建失败，将使用系统默认光标：{str(e)}")

    def create_legend(self):
        """创建右上角图例"""
        try:
            self.legend_item = LegendItem(self.iface.mapCanvas())
            self.iface.mapCanvas().scene().addItem(self.legend_item)
            
            # 监听地图变化更新图例位置
            self.iface.mapCanvas().extentsChanged.connect(self.update_legend_position)
            original_resize = self.iface.mapCanvas().resizeEvent
            def new_resize(event):
                original_resize(event)
                self.update_legend_position()
            self.iface.mapCanvas().resizeEvent = new_resize
            
            self.update_legend_position()
            log_message(self.logText, "右上角图例创建成功")
        except Exception as e:
            log_message(self.logText, f"创建图例失败: {str(e)}")
            QMessageBox.warning(self, "警告", f"无法创建右上角图例：{str(e)}")

    def update_legend_position(self):
        """更新图例位置到右上角"""
        if not self.legend_item:
            return
        canvas_rect = self.iface.mapCanvas().rect()
        x = canvas_rect.width() - self.legend_item.boundingRect().width() - 20
        y = 20
        self.legend_item.setPos(x, y)

    def connect_signals(self):
        """连接所有信号槽（确保save_shapefile绑定正确）"""
        # 页面切换
        self.btn_switch_to_params.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.page_params))
        self.btn_switch_to_function.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.page_function))
        
        # 模型配置
        self.browseModelBtn.clicked.connect(self.browse_model)
        self.browseConfigBtn.clicked.connect(self.browse_sam2_config)
        self.modelTypeCombo.currentTextChanged.connect(self.on_model_type_changed)
        self.loadModelBtn.clicked.connect(self.confirm_load_model)
        
        # 图层和采样
        self.layerCombo.currentIndexChanged.connect(self.on_layer_changed)
        self.sampleSizeSlider.valueChanged.connect(self.on_sample_size_changed)
        
        # 标注操作
        self.foregroundBtn.clicked.connect(lambda: self.set_click_mode(1))
        self.backgroundBtn.clicked.connect(lambda: self.set_click_mode(0))
        self.undoBtn.clicked.connect(self.undo_last)
        self.clearBtn.clicked.connect(self.clear_all)
        # 关键绑定：确保方法名与类内定义一致
        self.saveShpBtn.clicked.connect(self.save_shapefile)
        
        # 图层变化监听
        QgsProject.instance().layersAdded.connect(self.update_layer_combo)
        QgsProject.instance().layersRemoved.connect(self.update_layer_combo)

    def browse_model(self):
        """浏览选择SAM2模型文件"""
        initial_dir = os.path.dirname(self.model_path) if self.model_path else "."
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择SAM2模型文件", initial_dir, "模型文件 (*.pth *.pt)"
        )
        if model_path:
            self.model_path = model_path
            self.modelPathValue.setText(os.path.basename(model_path))
            log_message(self.logText, f"已选择SAM2模型文件: {os.path.basename(model_path)}")
            self.save_config()

    def browse_sam2_config(self):
        """浏览选择SAM2配置文件"""
        config_path, _ = QFileDialog.getOpenFileName(
            self, "选择SAM2配置文件", "", "YAML配置文件 (*.yaml *.yml)"
        )
        if config_path:
            self.sam2_config_path = config_path
            self.configPathValue.setText(os.path.basename(config_path))
            log_message(self.logText, f"已选择SAM2配置文件: {os.path.basename(config_path)}")
            self.save_config()

    def on_sample_size_changed(self, value):
        """采样尺寸变化"""
        self.max_size = value
        self.sampleSizeLabel.setText(f"最大处理尺寸：{self.max_size}px")
        if self.current_layer:
            self.load_raster_data()

    def update_layer_combo(self):
        """更新图层选择下拉框"""
        current_text = self.layerCombo.currentText()
        self.layerCombo.clear()
        
        # 加载所有有效栅格图层
        for layer in QgsProject.instance().mapLayers().values():
            if isinstance(layer, QgsRasterLayer) and layer.isValid():
                self.layerCombo.addItem(layer.name(), layer)
        
        # 恢复之前选择
        idx = self.layerCombo.findText(current_text)
        if idx >= 0:
            self.layerCombo.setCurrentIndex(idx)
            try:
                self.on_layer_changed(idx)
            except Exception as e:
                log_message(self.logText, f"恢复图层选择失败: {str(e)}")

    def on_layer_changed(self, index):
        """图层选择变化"""
        try:
            if index >= 0:
                self.current_layer = self.layerCombo.itemData(index)
                if self.current_layer:
                    log_message(self.logText, f"已选择栅格图层: {self.current_layer.name()}")
                    self.load_raster_data()
            else:
                self.current_layer = None
                self.image_data = None
                self.image_extent = None
                self.pixel_width = None
                self.pixel_height = None
                self.image_crs = None
                self.original_shape = None
                self.downsample_ratio = 1.0
                log_message(self.logText, "未选择栅格图层")
        except Exception as e:
            log_message(self.logText, f"图层切换错误: {str(e)}")

    def load_raster_data(self):
        """加载栅格数据并降采样"""
        if not self.current_layer:
            return
            
        try:
            import rasterio
            
            raster_path = self.current_layer.source()
            self.image_crs = self.current_layer.crs()
            self.image_extent = self.current_layer.extent()
            
            with rasterio.open(raster_path) as src:
                self.original_shape = (src.height, src.width)
                log_message(self.logText, f"原始栅格尺寸: {self.original_shape[0]}x{self.original_shape[1]}")
                
                # 计算降采样比例
                max_dim = max(self.original_shape)
                if max_dim > self.max_size:
                    self.downsample_ratio = self.max_size / max_dim
                    log_message(self.logText, f"图像过大，自动降采样 {self.downsample_ratio:.2f} 倍")
                else:
                    self.downsample_ratio = 1.0
                    log_message(self.logText, "图像尺寸适中，无需降采样")
                
                # 降采样后尺寸
                out_height = int(self.original_shape[0] * self.downsample_ratio)
                out_width = int(self.original_shape[1] * self.downsample_ratio)
                
                # 读取数据
                if src.count >= 3:
                    data = src.read(
                        [1, 2, 3],
                        out_shape=(3, out_height, out_width),
                        resampling=rasterio.enums.Resampling.bilinear
                    )
                    self.image_data = np.transpose(data, (1, 2, 0))
                else:
                    data = src.read(
                        1,
                        out_shape=(out_height, out_width),
                        resampling=rasterio.enums.Resampling.bilinear
                    )
                    self.image_data = np.stack([data, data, data], axis=-1)
                
                # 对比度拉伸
                self.image_data = stretch_percentile(self.image_data)
                
                # 更新分辨率
                self.pixel_width = self.current_layer.rasterUnitsPerPixelX() / self.downsample_ratio
                self.pixel_height = self.current_layer.rasterUnitsPerPixelY() / self.downsample_ratio
                
                log_message(self.logText, f"处理后栅格尺寸: {self.image_data.shape[0]}x{self.image_data.shape[1]}")
                log_message(self.logText, f"处理后分辨率: {abs(self.pixel_width):.4f}x{abs(self.pixel_height):.4f}")
            
            # 如果模型已加载，设置图像
            if self.sam_predictor:
                if hasattr(self.sam_predictor.model, 'dtype') and self.sam_predictor.model.dtype == torch.float16:
                    self.sam_predictor.set_image(self.image_data.astype(np.float16))
                else:
                    self.sam_predictor.set_image(self.image_data)
                log_message(self.logText, "已为分割更新图像数据")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载栅格数据失败: {str(e)}")
            log_message(self.logText, f"栅格加载错误: {str(e)}")
            self.image_data = None

    def set_click_mode(self, mode):
        """设置标注模式（前景/背景）"""
        self.click_mode = mode
        
        # 按钮样式切换
        if mode == 1:
            self.foregroundBtn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            self.backgroundBtn.setStyleSheet("")
            current_cursor = self.foreground_cursor
            log_message(self.logText, "切换到前景标注模式 - 光标已变为绿色圆形编辑点")
        else:
            self.backgroundBtn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
            self.foregroundBtn.setStyleSheet("")
            current_cursor = self.background_cursor
            log_message(self.logText, "切换到背景标注模式 - 光标已变为红色叉形编辑点")
        
        # 激活工具并设置光标
        self.activate_map_tool(current_cursor)

    def activate_map_tool(self, custom_cursor=None):
        """激活地图标注工具"""
        if not self.current_layer or self.image_data is None or self.image_data.size == 0:
            QMessageBox.warning(self, "警告", "请先选择并加载有效的栅格图层")
            return
            
        if (self.pixel_width is None or self.pixel_height is None or 
            abs(self.pixel_width) < 1e-12 or abs(self.pixel_height) < 1e-12 or
            self.image_extent is None):
            QMessageBox.warning(self, "警告", "栅格图层参数无效，无法进行坐标转换")
            return
            
        if not self.sam_predictor:
            QMessageBox.warning(self, "警告", "请先加载SAM2模型和配置文件")
            return
            
        # 创建工具实例
        if not self.point_tool:
            self.point_tool = MapClickTool(self.iface.mapCanvas())
            self.point_tool.canvasClicked.connect(self.on_map_click)
        
        # 设置光标
        if custom_cursor:
            self.point_tool.set_custom_cursor(custom_cursor)
            self.iface.mapCanvas().setCursor(custom_cursor)
        else:
            self.point_tool.set_custom_cursor(Qt.CrossCursor)
            self.iface.mapCanvas().setCursor(Qt.CrossCursor)
        
        # 激活工具
        self.iface.mapCanvas().setMapTool(self.point_tool)

    def on_map_click(self, point, button):
        """地图点击事件（添加标注点）"""
        if button != Qt.LeftButton:
            return
            
        if not self.sam_predictor:
            QMessageBox.warning(self, "警告", "请先加载SAM2模型")
            return
            
        # 转换为像素坐标
        pixel_coords = map_to_pixel_coords(
            point, self.image_extent, self.pixel_width, self.pixel_height,
            self.image_crs, self.iface.mapCanvas().mapSettings().destinationCrs(),
            QgsProject.instance()
        )
        if not pixel_coords:
            QMessageBox.warning(self, "警告", "点击位置超出栅格图层范围")
            return
            
        x, y = pixel_coords
        point_type = "前景" if self.click_mode == 1 else "背景"
        log_message(self.logText, f"添加{point_type}点: ({x}, {y})")
        
        # 绘制标注点
        if self.click_mode == 1:
            self.foreground_points.addPoint(point)
        else:
            self.background_points.addPoint(point)
        
        # 记录历史
        self.click_history.append( (point, (x, y), self.click_mode) )
        
        # 执行分割
        self._perform_segmentation()

    def _perform_segmentation(self):
        """执行SAM2分割（兼容旧版本，移除不支持参数）"""
        try:
            log_message(self.logText, "正在进行SAM2分割...")
            
            # 获取当前参数
            segment_params, mask_params, _ = self._get_current_params()
            
            # 设置图像（如果未设置）
            if not hasattr(self.sam_predictor, 'features'):
                if hasattr(self.sam_predictor.model, 'dtype') and self.sam_predictor.model.dtype == torch.float16:
                    self.sam_predictor.set_image(self.image_data.astype(np.float16))
                else:
                    self.sam_predictor.set_image(self.image_data)
            
            # 提取标注点
            input_points = np.array([item[1] for item in self.click_history])
            input_labels = np.array([item[2] for item in self.click_history])
            
            # 构建兼容参数（仅保留基础参数）
            sam_kwargs = {
                "point_coords": input_points,
                "point_labels": input_labels,
                "multimask_output": True
            }
            
            # 可选：添加最小掩码面积过滤（部分版本支持）
            if segment_params["min_mask_region_area"] > 0:
                sam_kwargs["min_mask_region_area"] = segment_params["min_mask_region_area"]
            
            # 执行预测
            masks, scores, logits = self.sam_predictor.predict(**sam_kwargs)
            
            # 选择置信度最高的掩码
            min_confidence = 0.7
            valid_mask_indices = np.where(scores >= min_confidence)[0]
            if len(valid_mask_indices) > 0:
                best_idx = valid_mask_indices[np.argmax(scores[valid_mask_indices])]
            else:
                best_idx = np.argmax(scores)
                log_message(self.logText, f"警告：所有掩码置信度低于{min_confidence}，将使用置信度最高的掩码 ({scores[best_idx]:.2f})")
            
            self.masks = masks[best_idx].astype(np.uint8)
            
            # 处理掩码
            self.masks = process_mask(self.masks, mask_params)
            
            log_message(self.logText, f"SAM2分割完成 (置信度: {scores[best_idx]:.2f})")
            self.display_mask_on_map()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"SAM2分割失败: {str(e)}")
            log_message(self.logText, f"SAM2分割错误: {str(e)}")

    def _get_current_params(self):
        """获取当前参数配置"""
        # 更新分割参数
        self.segment_params.update({
            "pred_iou_thresh": self.pred_iou_spin.value(),
            "stability_score_thresh": self.stability_score_spin.value(),
            "stability_score_offset": self.stability_offset_spin.value(),
            "box_nms_thresh": self.box_nms_spin.value(),
            "min_mask_region_area": self.min_mask_area_spin.value(),
            "multiscale_mode": self.multiscale_check.isChecked(),
            "max_num_masks": self.max_masks_spin.value()
        })
        
        # 更新Mask参数
        self.mask_params.update({
            "mask_smoothing": self.smoothing_check.isChecked(),
            "smoothing_kernel_size": self.smoothing_kernel_spin.value(),
            "hole_filling": self.hole_filling_check.isChecked(),
            "min_hole_area": self.min_hole_spin.value(),
            "dilate_iterations": self.dilate_spin.value(),
            "erode_iterations": self.erode_spin.value()
        })
        
        # 更新矢量参数
        self.vector_params.update({
            "simplify_geometry": self.simplify_check.isChecked(),
            "simplify_tolerance": self.simplify_tol_spin.value(),
            "keep_multipart": self.multipart_check.isChecked(),
            "min_polygon_area": self.min_polygon_spin.value(),
            "buffer_distance": self.buffer_spin.value()
        })
        
        return self.segment_params, self.mask_params, self.vector_params

    def display_mask_on_map(self):
        """在地图上显示分割结果（修复：传递正确的map_crs）"""
        if self.rubber_band_mask:
            self.iface.mapCanvas().scene().removeItem(self.rubber_band_mask)
        
        self.rubber_band_mask = QgsRubberBand(self.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry)
        self.rubber_band_mask.setColor(QColor(255, 0, 0, 100))  # 红色边框
        self.rubber_band_mask.setFillColor(QColor(255, 0, 0, 50))  # 红色半透明填充
        
        # 获取矢量参数
        _, _, vector_params = self._get_current_params()
        
        # 关键修复：从iface获取正确的地图画布CRS，传递给工具函数
        map_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        
        # 转换掩码为几何对象（传入map_crs）
        qgis_geoms = mask_to_geometries(
            self.masks, self.original_shape, self.image_extent,
            self.current_layer, vector_params, map_crs  # 新增传递map_crs
        )
        
        # 绘制几何对象
        for geom in qgis_geoms:
            self.rubber_band_mask.addGeometry(geom, None)
        
        self.iface.mapCanvas().refresh()

    def undo_last(self):
        """撤销上一步标注"""
        if not self.click_history:
            QMessageBox.information(self, "提示", "没有可撤销的操作")
            return
            
        # 移除最后一个点
        last_point, _, last_mode = self.click_history.pop()
        
        # 重新绘制标注点
        self.foreground_points.reset(QgsWkbTypes.PointGeometry)
        self.background_points.reset(QgsWkbTypes.PointGeometry)
        
        for point, _, mode in self.click_history:
            if mode == 1:
                self.foreground_points.addPoint(point)
            else:
                self.background_points.addPoint(point)
        
        # 更新分割结果
        if self.click_history:
            self._perform_segmentation()
            log_message(self.logText, "撤销上一步操作")
        else:
            # 无标注点时清除掩码
            if self.rubber_band_mask:
                self.iface.mapCanvas().scene().removeItem(self.rubber_band_mask)
                self.rubber_band_mask = None
            self.masks = None
            log_message(self.logText, "已清除所有操作，光标恢复默认")
            self.iface.mapCanvas().setCursor(Qt.ArrowCursor)
            if self.point_tool:
                self.point_tool.set_custom_cursor(Qt.ArrowCursor)
        
        self.iface.mapCanvas().refresh()

    def clear_all(self):
        """清除所有标注和结果"""
        # 清除标注点
        self.foreground_points.reset(QgsWkbTypes.PointGeometry)
        self.background_points.reset(QgsWkbTypes.PointGeometry)
        
        # 清除分割结果
        if self.rubber_band_mask:
            self.iface.mapCanvas().scene().removeItem(self.rubber_band_mask)
            self.rubber_band_mask = None
        
        # 清空历史和掩码
        self.click_history = []
        self.masks = None
        
        # 恢复默认光标
        self.iface.mapCanvas().setCursor(Qt.ArrowCursor)
        if self.point_tool:
            self.point_tool.set_custom_cursor(Qt.ArrowCursor)
        
        log_message(self.logText, "已清除所有点和结果，光标恢复默认")
        self.iface.mapCanvas().refresh()

    def confirm_load_model(self):
        """加载SAM2模型"""
        if not self.sam2_config_path:
            QMessageBox.warning(self, "警告", "请先选择SAM2配置文件（yaml/yml）")
            return
            
        if not self.model_path or not os.path.exists(self.model_path):
            QMessageBox.warning(self, "警告", "模型路径无效或未设置，请先选择有效的模型文件（pth/pt）")
            return

        try:
            log_message(self.logText, "正在加载SAM2模型...")
            self.model_type = self.modelTypeCombo.currentText()
            
            # 检查设备
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log_message(self.logText, f"使用设备: {device}")

            # 加载模型
            sam2_model = build_sam2(
                self.sam2_config_path,
                self.model_path,
                device=device,
                apply_postprocessing=False
            )
            
            # 禁用半精度，避免类型不匹配
            self.sam_predictor = SAM2ImagePredictor(sam2_model.float())
            log_message(self.logText, "模型加载时禁用半精度，避免类型不匹配错误")
            
            # 清理GPU缓存
            if device == "cuda":
                torch.cuda.empty_cache()
                log_message(self.logText, "GPU缓存已清理")
            
            log_message(self.logText, f"SAM2模型加载成功: {self.model_type}")
            
            # 如果已加载图像，设置模型图像
            if self.image_data is not None and self.image_data.size > 0:
                if hasattr(self.sam_predictor.model, 'dtype') and self.sam_predictor.model.dtype == torch.float16:
                    self.sam_predictor.set_image(self.image_data.astype(np.float16))
                else:
                    self.sam_predictor.set_image(self.image_data)
                log_message(self.logText, "已为分割设置图像")
            
            self.save_config()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载SAM2模型失败: {str(e)}")
            log_message(self.logText, f"SAM2模型加载错误: {str(e)}")

    def on_model_type_changed(self, text):
        """模型类型变化"""
        self.model_type = text
        self.save_config()

    # 关键修复：确保save_shapefile方法在类内，缩进正确
    def save_shapefile(self):
        """导出分割结果为Shapefile（修复几何对象格式+信号绑定）"""
        if self.masks is None or not self.masks.any():
            QMessageBox.information(self, "提示", "没有可保存的分割结果")
            return

        if not self.image_crs or not self.image_crs.isValid():
            QMessageBox.information(self, "提示", "没有可用的有效坐标系统信息")
            return

        # 选择保存路径
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存Shapefile", "", "Shapefile (*.shp)"
        )
        if not save_path:
            return

        try:
            # 关键导入：确保Shapely能解析WKT
            from shapely import wkt
            
            # 获取矢量参数
            _, _, vector_params = self._get_current_params()
            
            # 从iface获取正确的地图画布CRS，传递给工具函数
            map_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            
            # 转换掩码为几何对象（传入map_crs）
            qgis_geoms = mask_to_geometries(
                self.masks, self.original_shape, self.image_extent,
                self.current_layer, vector_params, map_crs
            )
            
            if not qgis_geoms:
                QMessageBox.warning(self, "警告", "没有符合条件的矢量要素可保存")
                return
            
            # 核心修复：将QgsGeometry转换为Shapely几何对象（而非WKT字符串）
            features = []
            for geom in qgis_geoms:
                try:
                    # 转换QgsGeometry到Shapely几何对象（兼容性更好）
                    shapely_geom = wkt.loads(geom.asWkt())
                    features.append({
                        'properties': {'class': 1, 'area': geom.area()},
                        'geometry': shapely_geom  # 传入Shapely对象，而非字符串
                    })
                except Exception as e:
                    log_message(self.logText, f"几何对象转换失败: {str(e)}")
                    continue
            
            if not features:
                QMessageBox.warning(self, "警告", "没有可转换的有效几何对象")
                return
            
            # 获取CRS（确保格式正确）
            try:
                crs_wkt = self.image_crs.toWkt()
            except AttributeError:
                epsg_code = self.image_crs.postgisSrid()
                crs_wkt = f"EPSG:{epsg_code}"

            # 保存为Shapefile（此时geometry字段是Shapely对象，兼容所有GeoPandas版本）
            gdf = gpd.GeoDataFrame.from_features(features, crs=crs_wkt)
            gdf.to_file(save_path)

            # 添加到QGIS项目
            layer_name = os.path.splitext(os.path.basename(save_path))[0]
            vlayer = QgsVectorLayer(save_path, layer_name, "ogr")
            if vlayer.isValid():
                QgsProject.instance().addMapLayer(vlayer)
                log_message(self.logText, f"Shapefile保存成功并添加到项目: {save_path}")
                QMessageBox.information(self, "成功", f"分割结果已保存为: {os.path.basename(save_path)}")
            else:
                log_message(self.logText, f"Shapefile保存成功，但添加到项目失败: {save_path}")
                QMessageBox.information(self, "成功", f"分割结果已保存为: {os.path.basename(save_path)}，但添加到QGIS项目失败")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存Shapefile失败: {str(e)}")
            log_message(self.logText, f"Shapefile保存错误: {str(e)}")

    def load_config(self):
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            log_message(self.logText, "配置文件不存在，使用默认配置")
            return
            
        try:
            tree = ET.parse(self.config_path)
            root = tree.getroot()
            
            # 读取模型路径
            if root.find("model_path") is not None:
                self.model_path = root.find("model_path").text or ""
                if os.path.exists(self.model_path):
                    self.modelPathValue.setText(os.path.basename(self.model_path))
            
            # 读取配置文件路径
            if root.find("sam2_config_path") is not None:
                self.sam2_config_path = root.find("sam2_config_path").text or ""
                if os.path.exists(self.sam2_config_path):
                    self.configPathValue.setText(os.path.basename(self.sam2_config_path))
            
            # 读取模型类型
            if root.find("model_type") is not None:
                model_type = root.find("model_type").text or "vit_l"
                idx = self.modelTypeCombo.findText(model_type)
                if idx >= 0:
                    self.modelTypeCombo.setCurrentIndex(idx)
                    self.model_type = model_type
            
            log_message(self.logText, "配置文件加载成功")
        except Exception as e:
            log_message(self.logText, f"加载配置文件失败: {str(e)}")

    def save_config(self):
        """保存配置文件"""
        try:
            root = ET.Element("config")
            
            # 保存模型路径
            model_path_elem = ET.SubElement(root, "model_path")
            model_path_elem.text = self.model_path
            
            # 保存配置文件路径
            config_path_elem = ET.SubElement(root, "sam2_config_path")
            config_path_elem.text = self.sam2_config_path
            
            # 保存模型类型
            model_type_elem = ET.SubElement(root, "model_type")
            model_type_elem.text = self.model_type
            
            # 美化XML并保存
            xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
            with open(self.config_path, "w", encoding="utf-8") as f:
                f.write(xml_str)
            
            log_message(self.logText, "配置已保存")
        except Exception as e:
            log_message(self.logText, f"保存配置文件失败: {str(e)}")

    def closeEvent(self, event):
        """关闭对话框时清理资源"""
        # 恢复默认光标
        self.iface.mapCanvas().setCursor(Qt.ArrowCursor)
        if self.point_tool:
            self.point_tool.set_custom_cursor(Qt.ArrowCursor)
        
        # 清理绘图对象
        if self.foreground_points:
            self.foreground_points.reset(QgsWkbTypes.PointGeometry)
        if self.background_points:
            self.background_points.reset(QgsWkbTypes.PointGeometry)
        if self.rubber_band_mask:
            self.iface.mapCanvas().scene().removeItem(self.rubber_band_mask)
        if self.legend_item:
            self.iface.mapCanvas().scene().removeItem(self.legend_item)
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        log_message(self.logText, "SAM2插件已关闭，资源已清理")
        event.accept()