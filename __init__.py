# -*- coding: utf-8 -*-
"""
/***************************************************************************
 sam2_segmentation
                                 A QGIS plugin
 SAM2-based interactive segmentation plugin for QGIS
                              -------------------
        begin                : 2025-11-04
        copyright            : (C) 2025 by Your Name
        email                : your@email.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from __future__ import absolute_import
from qgis.PyQt.QtCore import QSettings, QTranslator, qVersion, QCoreApplication
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsProject
import os.path

# 导入插件主对话框
from .sam_segmentation_dialog import SamSegmentationDialog


class Sam2SegmentationPlugin:
    """QGIS插件主类"""
    def __init__(self, iface):
        """构造函数"""
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.pluginIsActive = False
        self.dlg = None
        
        # 初始化翻译（可选）
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'sam2_segmentation_{}.qm'.format(locale)
        )
        
        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

        # 创建工具栏
        self.toolbar = self.iface.addToolBar(u'SAM2 Segmentation')
        self.toolbar.setObjectName(u'SAM2SegmentationToolbar')

    def initGui(self):
        """创建菜单和工具栏图标（修复图标路径：根目录icon.png）"""
        # 关键修复：图标路径改为插件根目录下的 icon.png
        icon_path = os.path.join(self.plugin_dir, 'icon.png')
        
        # 图标不存在时使用QGIS默认图标（容错处理）
        if not os.path.exists(icon_path):
            icon_path = ':/images/themes/default/mActionAddRasterLayer.svg'
            self.iface.messageBar().pushInfo("提示", "未找到自定义图标，使用默认图标")
        
        # 使用QIcon直接创建图标（兼容所有QGIS版本）
        try:
            plugin_icon = QIcon(icon_path)
        except Exception as e:
            plugin_icon = QIcon(':/images/themes/default/mActionAddRasterLayer.svg')
            self.iface.messageBar().pushWarning("警告", f"图标加载失败：{str(e)}，使用默认图标")
        
        # 创建工具栏按钮
        self.action = QAction(
            plugin_icon,
            u'SAM2 Interactive Segmentation',
            self.iface.mainWindow()
        )
        self.action.triggered.connect(self.run)
        self.action.setStatusTip(u'基于SAM2的交互式分割工具')
        self.action.setWhatsThis(u'支持前景/背景标注和矢量结果导出')
        
        # 添加到工具栏和菜单
        self.toolbar.addAction(self.action)
        self.iface.addPluginToMenu(u'&SAM2分割工具', self.action)

    def unload(self):
        """卸载插件清理资源"""
        self.iface.removePluginMenu(u'&SAM2分割工具', self.action)
        self.iface.removeToolBarIcon(self.action)
        del self.toolbar
        
        if self.dlg:
            self.dlg.close()
            self.dlg = None
        
        self.pluginIsActive = False

    def run(self):
        """启动插件（显示对话框）"""
        if self.dlg is None:
            self.dlg = SamSegmentationDialog(self.iface)
        
        self.dlg.show()
        result = self.dlg.exec_()
        
        if result:
            pass

# QGIS必需的工厂函数
def classFactory(iface):
    return Sam2SegmentationPlugin(iface)