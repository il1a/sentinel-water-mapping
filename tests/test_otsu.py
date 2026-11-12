"""Synthetic safety checks for released-map harmonization."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from water_mapping.otsu import (CoverageError, assert_reference_match, otsu_name,
                                run_comparison, summarize, validate_prediction, prediction_valid, sensitivity_nonwater, paired_defined, METHODS)


class OtsuTests(unittest.TestCase):
    def setUp(self):
        self.geo = dict(GeographicTypeGeoKey=4326, GTModelTypeGeoKey=2,
                        GTRasterTypeGeoKey=1, ModelPixelScale=(.1,.1,0),
                        ModelTiepoint=(0,0,0,1,2,0))
        self.image = 'Bolivia_123_S1Hand.tif'
        self.mask = np.ones((2,2), dtype=bool)
        self.values = np.array([[0,1],[1,0]],dtype=np.int16)

    def check(self, values=None, mask=None, geo=None, nodata=-1, name=None):
        return validate_prediction(self.values if values is None else values,
            self.mask if mask is None else mask, self.geo if geo is None else geo,
            self.geo,nodata,name=name or otsu_name(self.image),image=self.image)

    def test_valid(self):
        self.assertEqual(self.check()['uncovered_scored_pixels'],0)

    def test_identity(self):
        for image in ('../Bolivia_123_S1Hand.tif','Bolivia_x_S1Hand.tif','Bolivia_123_LabelHand.tif'):
            with self.assertRaises(ValueError): otsu_name(image)
        with self.assertRaisesRegex(ValueError,'identity'):
            self.check(name='Bolivia_124_S1OtsuLabelHand.tif')

    def test_shape_and_class(self):
        with self.assertRaisesRegex(ValueError,'shape'): self.check(values=np.zeros((1,2)))
        with self.assertRaisesRegex(ValueError,'class'): self.check(values=np.full((2,2),2))
        with self.assertRaisesRegex(ValueError,'class'): self.check(values=np.full((2,2),-1),nodata=None)

    def test_grid_keys(self):
        for key in self.geo:
            geo=dict(self.geo);del geo[key]
            with self.subTest(key=key),self.assertRaises(ValueError): self.check(geo=geo)
        for key,value in [('GeographicTypeGeoKey',3857),('GTRasterTypeGeoKey',2),
                          ('ModelTiepoint',(0,0,0,2,2,0)),('ModelPixelScale',(.2,.1,0))]:
            geo=dict(self.geo);geo[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): self.check(geo=geo)

    def test_nodata_on_mask_is_error(self):
        a=self.values.copy();a[0,0]=-1
        with self.assertRaises(CoverageError) as context: self.check(values=a)
        self.assertEqual(context.exception.check['uncovered_scored_pixels'],1)
        self.assertIn(self.image,str(context.exception))
        mask=self.mask.copy();mask[0,0]=False
        self.assertEqual(self.check(values=a,mask=mask)['missing_pixels'],1)

    def test_zero_nodata_not_silently_nonwater(self):
        with self.assertRaises(CoverageError): self.check(nodata=0)

    def test_nonfinite_missing(self):
        a=self.values.astype(float);a[0,0]=np.nan
        with self.assertRaises(CoverageError): self.check(values=a,nodata=np.nan)

    def test_counts_and_fingerprint(self):
        row=dict(tp=1,fp=2,fn=3,tn=4,scored_pixels=10,prediction_sha256='abc')
        assert_reference_match(row,{k:str(v) for k,v in row.items()})
        for key in row:
            changed=dict(row);changed[key]=0 if key!='prediction_sha256' else 'def'
            with self.subTest(key=key),self.assertRaises(ValueError): assert_reference_match(row,changed)

    def test_undefined_excluded_and_accuracy(self):
        result=summarize([dict(tp=0,fp=0,fn=0,tn=10),dict(tp=1,fp=1,fn=0,tn=8)])
        self.assertEqual(result['macro_iou'],.5)
        self.assertEqual(result['undefined_iou_chips'],1)
        self.assertEqual(result['overall_accuracy'],.95)
        self.assertIsNone(summarize([])['overall_accuracy'])

    def test_identical_intersection_and_sensitivity(self):
        from water_mapping.core import confusion
        labels = np.array([[1,1],[0,-1]])
        released = np.array([[-1,1],[0,1]])
        sar = np.ones((2,2), dtype=bool)
        common = sar & prediction_valid(released, -1)
        predictions = [np.ones((2,2),dtype=bool), np.zeros((2,2),dtype=bool), released==1]
        scores = [confusion(p, labels, common) for p in predictions]
        self.assertEqual([r['scored_pixels'] for r in scores], [2,2,2])
        self.assertEqual(scores[2]['iou'],1)
        sensitivity = sensitivity_nonwater(scores[2],1,0)
        self.assertEqual(sensitivity['fn'],1)
        self.assertEqual(sensitivity['iou'],.5)
        self.assertEqual(sensitivity['scored_pixels'],3)
        self.assertEqual(int(prediction_valid(np.array([0,1]),0).sum()),1)
        self.assertIsNone(sensitivity_nonwater(dict(tp=0,fp=0,fn=0,tn=0),0,0)['iou'])
        self.assertEqual(sensitivity_nonwater(dict(tp=0,fp=0,fn=0,tn=0),1,0)['iou'],0)

    def test_failed_preflight_persists_failure_without_csv(self):
        import json
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('water_mapping.otsu.sha256', return_value='synthetic'), \
                 patch('water_mapping.otsu.environment_record', return_value={}), \
                 patch('water_mapping.otsu.load_splits', side_effect=ValueError('synthetic failure')):
                with self.assertRaisesRegex(ValueError, 'synthetic failure'):
                    run_comparison(root, root/'output')
            evidence = json.loads((root/'output/study.json').read_text())
            self.assertEqual(evidence['status'], 'failed')
            self.assertNotIn('result', evidence)
            self.assertFalse((root/'output/evaluation.csv').exists())

    def test_paired_defined_uses_same_chip_set(self):
        rows=[]
        for name in ('a','b'):
            for method in METHODS:
                undefined = name=='a' and method=='released_otsu'
                rows.append(dict(split='test',image=name,method=method,
                    tp=0 if undefined else 1,fp=0,fn=0,tn=1,iou=None if undefined else 1))
        result=paired_defined(rows,set())['test_all_90']
        self.assertEqual(result['included_images'],['b'])
        self.assertEqual(result['excluded'][0]['undefined_methods'],['released_otsu'])
        self.assertEqual(result['macro_iou'],dict.fromkeys(METHODS,1.0))
        self.assertIsNone(paired_defined([],set())['test_all_90']['macro_iou']['selected'])

    def test_fresh_output_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'existing').mkdir();(root/'existing/sentinel').write_text('keep')
            with self.assertRaisesRegex(ValueError,'fresh'): run_comparison(root,root/'existing')
            for protected in ('reference',):
                with self.assertRaisesRegex(ValueError,'immutable'):
                    run_comparison(root,root/protected/'new')


if __name__=='__main__': unittest.main()
