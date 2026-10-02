"""Bearing plots: numerical, metadata and real Qt background acceptance."""
import os
import csv
import tempfile
import time
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM']='offscreen'
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
import numpy as np
from scipy.io import savemat
from airwatch.workflows.bearing_preview import load_bearing_preview, MAX_PREVIEW_SAMPLES
from airwatch.data.cwru import AmbiguousSignalError
from airwatch.ui.bearing_background import BearingInferenceWorker
from airwatch.analysis.bearing_fault_frequencies import BearingGeometry, compute_characteristic_frequencies
import main

ROOT=Path(__file__).resolve().parents[1]


class BearingPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.path=self.root/'test.mat'
        self.signal=2*np.cos(2*np.pi*64*np.arange(2048)/1024)
        savemat(self.path,{'X_DE_time':self.signal})
        self.manifest=self.root/'manifest.csv'
        with self.manifest.open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=['local_path','sensor_key','sample_rate_hz'])
            wr.writeheader(); wr.writerow(dict(local_path='test.mat',sensor_key='X_DE_time',sample_rate_hz=1024))

    def load(self,**kwargs):
        return load_bearing_preview(self.path,manifest_path=self.manifest,project_root=self.root,**kwargs)

    def test_frequency_time_amplitude_readonly_and_same_channel_as_inference(self):
        before=self.path.read_bytes()
        v=self.load()
        self.assertEqual(v.sensor_key,BearingInferenceWorker._sensor_key_from_manifest(self.manifest,self.root,self.path))
        np.testing.assert_allclose(v.waveform.values[0],self.signal,atol=1e-7)
        self.assertAlmostEqual(v.waveform.x[-1],2047/1024)
        i=np.argmax(v.spectrum.magnitude)
        self.assertEqual(v.spectrum.frequency_hz[i],64)
        self.assertAlmostEqual(v.spectrum.magnitude[i],1,places=6)
        self.assertFalse(v.waveform.values.flags.writeable)
        self.assertEqual(self.path.read_bytes(),before)

    def test_unknown_rate_requires_explicit_input_and_conflict_rejected(self):
        with self.assertRaisesRegex(ValueError,'不一致'): self.load(sample_rate_hz=48000)
        self.manifest.unlink()
        v=self.load(); self.assertIsNone(v.spectrum)
        self.assertEqual(v.waveform.x[-1],2047)
        self.assertIsNotNone(self.load(sample_rate_hz=1024).spectrum)
        for rate in (0,-1,float('nan'),'bad'):
            with self.assertRaises(ValueError): self.load(sample_rate_hz=rate)

    def test_ambiguity_nan_and_explicit_preview_bound(self):
        savemat(self.path,{'A_DE_time':self.signal,'B_DE_time':self.signal})
        self.manifest.unlink()
        with self.assertRaises(AmbiguousSignalError): self.load()
        savemat(self.path,{'X_DE_time':[np.nan,1]})
        with self.assertRaises(ValueError): self.load()
        savemat(self.path,{'X_DE_time':np.arange(MAX_PREVIEW_SAMPLES+100)})
        v=self.load()
        self.assertEqual(v.shown_samples,MAX_PREVIEW_SAMPLES)
        self.assertEqual(v.total_samples,MAX_PREVIEW_SAMPLES+100)

    def test_real_audited_multichannel_file_selects_frozen_channel(self):
        path=ROOT/'datasets/bearing/cwru/raw/normal/Normal_2.mat'
        v=load_bearing_preview(path)
        self.assertEqual(v.sensor_key,'X099_DE_time')
        self.assertEqual(v.waveform.sample_rate_hz,12000)
        self.assertEqual(v.rate_source,'已审计清单')

    def test_metadata_path_resolution_ignores_cwd_and_supports_bundle_root(self):
        import sys
        before=Path.cwd()
        try:
            os.chdir(self.root)
            # Explicit temporary resource root models a read-only extracted bundle.
            with patch.object(sys,'frozen',True,create=True),patch.object(sys,'_MEIPASS',str(self.root),create=True):
                v=self.load()
                self.assertEqual(v.sensor_key,'X_DE_time')
        finally: os.chdir(before)
        with self.assertRaises(InterruptedError): self.load(cancelled=lambda: True)

    def window(self):
        w=main.MyWindow(); self.addCleanup(w.close)
        w.tabWidget.setCurrentWidget(w.tab_3)
        w.recognition_workbench.select_module(2)
        w.show()
        return w,w.recognition_workbench.bearing_plots

    def wait(self,panel):
        deadline=time.monotonic()+10
        while panel.thread is not None and time.monotonic()<deadline: QTest.qWait(10)
        self.assertIsNone(panel.thread)

    def test_characteristic_frequency_math_and_invalid_geometry(self):
        frequencies = compute_characteristic_frequencies(BearingGeometry(8, 10, 50, 1200))
        self.assertEqual((frequencies.shaft_hz, frequencies.ftf_hz,
                          frequencies.bpfo_hz, frequencies.bpfi_hz, frequencies.bsf_hz),
                         (20.0, 8.0, 64.0, 96.0, 48.0))
        for geometry in (BearingGeometry(8.5, 10, 50, 1200),
                         BearingGeometry(8, 50, 50, 1200),
                         BearingGeometry(8, 10, 50, 1200, float('nan'))):
            with self.assertRaises(ValueError):
                compute_characteristic_frequencies(geometry)

    def test_frequency_lines_have_source_and_are_cleared_on_edit_and_view_switch(self):
        w, panel = self.window()
        panel.select_file(self.path)
        self.wait(panel)
        panel.rate.setText('1024')
        panel.refresh.click()
        self.wait(panel)
        panel.spectrum_button.click()
        panel.frequency_toggle.click()
        values = {'roller_count': '8', 'roller_diameter_mm': '10',
                  'pitch_diameter_mm': '50', 'shaft_rpm': '1200',
                  'contact_angle_deg': '0'}
        for key, value in values.items():
            panel.geometry_edits[key].setText(value)
        panel.frequency_button.click()
        self.assertEqual(len(panel._frequency_lines), 4)
        self.assertIn('来源：用户填写', panel.frequency_note.text())
        self.assertGreaterEqual(panel.plot.height(), 220)
        self.assertTrue(panel.frequency_group.isVisible())
        panel.frequency_button.click()
        self.assertEqual(len(panel._frequency_lines), 4)
        panel.geometry_edits['shaft_rpm'].setText('1300')
        self.assertEqual(len(panel._frequency_lines), 0)
        panel.wave_button.click()
        self.assertEqual(len(panel._frequency_lines), 0)
        self.assertFalse(panel.frequency_group.isVisible())

    def test_frequency_lines_omit_out_of_band_and_unknown_rate(self):
        w, panel = self.window()
        panel.select_file(self.path)
        self.wait(panel)
        panel.rate.setText('1024')
        panel.refresh.click()
        self.wait(panel)
        panel.spectrum_button.click()
        for key, value in {'roller_count':'8', 'roller_diameter_mm':'10',
                           'pitch_diameter_mm':'50', 'shaft_rpm':'120000',
                           'contact_angle_deg':'0'}.items():
            panel.geometry_edits[key].setText(value)
        panel.frequency_button.click()
        self.assertEqual(len(panel._frequency_lines), 0)
        self.assertIn('超出当前频带', panel.frequency_note.text())
        panel.select_file(self.root/'missing.mat')
        self.wait(panel)
        self.assertEqual(len(panel._frequency_lines), 0)

    def test_frequency_parameters_do_not_carry_to_next_file(self):
        w, panel = self.window()
        panel.geometry_edits['shaft_rpm'].setText('1200')
        panel.select_file(self.path)
        self.wait(panel)
        self.assertTrue(all(not edit.text() for edit in panel.geometry_edits.values()))

    def test_same_spectrum_render_clears_marker_state(self):
        w, panel = self.window()
        panel.path = str(self.path.resolve())
        panel.view = self.load()
        panel.render('spectrum')
        for key, value in {'roller_count':'8', 'roller_diameter_mm':'10',
                           'pitch_diameter_mm':'50', 'shaft_rpm':'1200',
                           'contact_angle_deg':'0'}.items():
            panel.geometry_edits[key].setText(value)
        panel.update_frequency_lines()
        self.assertEqual(len(panel._frequency_lines), 4)
        panel.render('spectrum')
        self.assertEqual(panel._frequency_lines, [])

    def test_actual_file_dialog_entry_background_and_error_clear(self):
        w,p=self.window()
        with patch.object(main.QFileDialog,'getOpenFileName',return_value=(str(self.path),'')):
            w.select_bearing_file()
        self.wait(p)
        self.assertIsNotNone(p.view,p.caption.text())
        self.assertFalse(p.spectrum_button.isEnabled())
        p.rate.setText('1024'); p.refresh.click(); self.wait(p)
        p.spectrum_button.click()
        x,y=p.plot.listDataItems()[0].getData()
        self.assertEqual(x[np.argmax(y)],64)
        self.assertEqual(w.bearing_diagnosis_label.text(),'—')
        p.select_file(self.root/'missing.mat'); self.wait(p)
        self.assertIsNone(p.view)
        self.assertFalse(p.plot.listDataItems())
        self.assertIn('无法显示',p.caption.text())

    def test_rate_edit_and_spectrum_return_refit_waveform(self):
        w, panel = self.window()
        panel.select_file(self.path); self.wait(panel)
        self.assertGreater(panel.plot.viewRange()[0][1], 2000)
        panel.rate.setText('48000'); panel.refresh.click(); self.wait(panel)
        QTest.qWait(50)
        low, high = panel.plot.viewRange()[0]
        self.assertLess(high - low, 0.1)
        panel.spectrum_button.click(); QTest.qWait(10)
        panel.wave_button.click(); QTest.qWait(50)
        low, high = panel.plot.viewRange()[0]
        self.assertLess(high - low, 0.1)
        self.assertGreater(len(panel.plot.listDataItems()[0].getData()[0]), 2)

    def test_latest_request_wins_and_runs_off_ui_thread(self):
        w,p=self.window()
        actual=load_bearing_preview
        started=threading.Event(); release=threading.Event(); identities=[]
        def delayed(path,**kwargs):
            identities.append(threading.get_ident()); started.set(); release.wait(3)
            return actual(path,**kwargs)
        second=self.root/'second.mat'; savemat(second,{'Y_DE_time':self.signal})
        with patch('airwatch.ui.bearing_plot_panel.load_bearing_preview',side_effect=delayed):
            p.select_file(self.path)
            self.assertTrue(started.wait(3))
            old=p.thread
            p.select_file(second)
            self.assertIs(p.thread,old)
            self.assertIsNone(p.view)
            release.set(); self.wait(p)
        self.assertEqual(p.view.path,str(second.resolve()))
        self.assertTrue(all(i!=threading.get_ident() for i in identities))
        self.assertEqual(len(identities),2)

    def test_cancel_and_close_wait_do_not_destroy_running_thread(self):
        w,p=self.window()
        release=threading.Event(); started=threading.Event()
        def delayed(*args,**kwargs):
            started.set(); release.wait(3); raise InterruptedError()
        with patch('airwatch.ui.bearing_plot_panel.load_bearing_preview',side_effect=delayed):
            p.select_file(self.path); self.assertTrue(started.wait(3))
            p.cancel_button.click()
            self.assertIsNone(p.view)
            self.assertFalse(w.close())
            self.assertIsNotNone(p.thread)
            release.set(); self.wait(p)
        self.assertTrue(w.close())


if __name__=='__main__': unittest.main()

