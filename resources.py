# -*- coding: utf-8 -*-
"""
Resource file for SAM2 Segmentation plugin
"""

from PyQt5.QtCore import QObject, Qt
from PyQt5.QtGui import QIcon, QPixmap

def icon(name):
    """Load icon from resources"""
    try:
        return QIcon(f":/plugins/sam2_segmentation/icons/icon.png")
    except:
        return QIcon()

# 示例：如果有图标资源，可在此定义
class Resources(QObject):
    @staticmethod
    def get_plugin_icon():
        return icon("icon")