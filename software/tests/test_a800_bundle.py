import json
from pathlib import Path
import tempfile
import unittest
from training.build_a800_bundle import write_integrity, verify_integrity

class BundleTests(unittest.TestCase):
    def test_mutable_outputs_excluded_but_payload_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'code.txt').write_text('source',encoding='utf-8')
            (root/'outputs').mkdir()
            (root/'outputs/log.txt').write_text('mutable')
            write_integrity(root,{'ready':False})
            self.assertTrue(verify_integrity(root)['ok'])
            (root/'outputs/log.txt').write_text('changed')
            self.assertTrue(verify_integrity(root)['ok'])
            (root/'code.txt').write_text('changed')
            with self.assertRaises(ValueError):
                verify_integrity(root)

    def test_index_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_integrity(root,{'ready':False})
            (root/'bundle-files.json').write_text('{}')
            with self.assertRaises(ValueError):
                verify_integrity(root)

if __name__=='__main__':
    unittest.main()
