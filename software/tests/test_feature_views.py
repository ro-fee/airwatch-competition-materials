"""Real Qt acceptance for index-domain model feature views."""
import os
import unittest
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['PYQTGRAPH_QT_LIB']='PyQt5'
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
import main
from airwatch.analysis.feature_views import select_feature_vector, compute_feature_spectrogram


class FeatureViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.w=main.MyWindow()
        self.addCleanup(self.w.close)
        self.p=self.w.label_featuremap
        self.data=np.arange(5*32,dtype=float).reshape(5,32)
        self.w.filename=['a']

    def test_single_file_curve_uses_first_window_not_first_scalar(self):
        self.w.plotsig_featuremap(self.data,self.p)
        curves=self.p.plotItem.listDataItems()
        self.assertEqual(len(curves),1)
        np.testing.assert_array_equal(curves[0].getData()[1],self.data[0])
        self.assertIn('特征维度',self.p.getAxis('bottom').labelText)
        self.assertIn('共 5',self.p.plotItem.titleLabel.text)

    def test_multifile_uses_window_counts_not_file_number_as_row(self):
        self.w.filename=['a','b']; self.w.feature_row_counts=[3,2]
        self.w.comboBox.blockSignals(True)
        self.w.comboBox.clear(); self.w.comboBox.addItems(['a','b'])
        self.w.comboBox.setCurrentIndex(1)
        self.w.plotsig_featuremap(self.data,self.p)
        np.testing.assert_array_equal(self.p.plotItem.listDataItems()[0].getData()[1],self.data[3])
        self.w.comboBox.setCurrentIndex(-1)
        self.w.plotsig_featuremap(self.data,self.p)
        np.testing.assert_array_equal(self.p.plotItem.listDataItems()[0].getData()[1],self.data[0])

    def test_missing_or_incorrect_file_mapping_is_rejected(self):
        for counts in (None,[1,1],[3,0],[True,4]):
            with self.assertRaises(ValueError):
                select_feature_vector(self.data,file_count=2,row_counts=counts)

    def test_feature_spectrum_does_not_use_raw_sample_rate(self):
        arrays=[]
        for fs in (9600,40000000,'invalid'):
            self.w.plotspec_featuremap(self.data,self.p,fs,16)
            images=[i for i in self.p.plotItem.items if isinstance(i,pg.ImageItem)]
            self.assertEqual(len(images),1)
            arrays.append(images[0].image.copy())
            self.assertEqual(self.p.getAxis('left').labelUnits,'')
            self.assertIn('非物理时频',self.p.plotItem.titleLabel.text)
        np.testing.assert_array_equal(arrays[0],arrays[1])
        np.testing.assert_array_equal(arrays[1],arrays[2])

    def test_bad_data_clears_stale_graph(self):
        for data in ([],np.ones((2,2,2)),np.full((2,32),np.nan),np.ones((2,32),complex)):
            self.w.plotsig_featuremap(self.data,self.p)
            self.w.plotsig_featuremap(data,self.p)
            self.assertFalse(self.p.plotItem.items)
            self.assertIn('特征暂不可用',self.p.plotItem.titleLabel.text)
        self.w.plotspec_featuremap(self.data,self.p,1,33)
        self.assertFalse(self.p.plotItem.items)

    def test_input_is_immutable_and_window_is_strict(self):
        before=self.data.copy()
        view=select_feature_vector(self.data)
        self.assertFalse(view.values.flags.writeable)
        compute_feature_spectrogram(view,16)
        np.testing.assert_array_equal(self.data,before)
        with self.assertRaises(ValueError):
            compute_feature_spectrogram(view,16.5)

    def test_switch_task_clears_feature_cache(self):
        self.w.feature=self.data; self.w.feature_map=self.data
        self.w.feature_row_counts=[5]
        self.w.plotsig_featuremap(self.data,self.p)
        self.w._on_task_changed()
        self.assertEqual(self.w.feature,[])
        self.assertEqual(self.w.feature_map,[])
        self.assertIsNone(self.w.feature_row_counts)
        self.assertFalse(self.p.plotItem.items)

    def test_middle_layer_choice_uses_maintained_in_memory_renderer(self):
        self.w.feature_map=self.data
        self.w.comboBox_choose_feature.setCurrentText('中间层特征')
        self.w.radioButton_time_3.setChecked(True)
        self.w.choose_feature_signal()
        curves=self.p.plotItem.listDataItems()
        self.assertEqual(len(curves),1)
        np.testing.assert_array_equal(curves[0].getData()[1],self.data[0])


if __name__=='__main__':
    unittest.main()
