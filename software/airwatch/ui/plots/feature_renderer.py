"""Render model features without presenting them as physical time/frequency."""
from .pyqtgraph_renderer import render_waveform, render_spectrogram
from .pyqtgraph_renderer import _reset_axes
import pyqtgraph as pg
from pathlib import Path


def render_tsne(plot, view):
    item = getattr(plot, 'plotItem', plot)
    _reset_axes(item)
    item.clear()
    item.getAxis('left').setTicks(None)
    item.getAxis('bottom').setTicks(None)
    item.setLabel('bottom', 't-SNE 维度 1', units='')
    item.setLabel('left', 't-SNE 维度 2', units='')
    item.setTitle('每点=一个窗口；颜色=来源文件（悬停查看）；非类别预测或准确率')
    points = pg.ScatterPlotItem(size=7, pen=None, hoverable=True,
        tip=lambda x, y, data: str(data))
    points.addPoints([dict(pos=xy, brush=pg.intColor(int(file_id)),
        data=f'窗口 {index+1} / 文件 {int(file_id)+1}: {Path(view.file_names[int(file_id)]).name}')
        for index, (xy, file_id) in enumerate(zip(view.coordinates, view.file_ids))])
    item.addItem(points)
    item.getViewBox().autoRange()
    return points


def render_feature_vector(plot, view):
    render_waveform(plot, view.values)
    item = getattr(plot, 'plotItem', plot)
    item.setLabel('bottom', '特征维度索引', units='')
    item.setLabel('left', '模型特征值', units='')
    item.setTitle(f'当前文件第 1 个窗口（共 {view.rows_in_file} 个）；非原始波形')


def render_feature_spectrogram(plot, view, rows_in_file):
    image = render_spectrogram(plot, view)
    item = getattr(plot, 'plotItem', plot)
    item.setLabel('bottom', '特征维度索引', units='')
    item.setLabel('left', '索引域频率（周期/特征）', units='')
    item.setTitle(f'特征索引域谱：第 1/{rows_in_file} 个窗口；非物理时频；颜色为峰值以下 80 dB')
    return image
