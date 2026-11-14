import numpy as np
from src.metrics.detection import _mask,pii_miss_rate,over_redaction_rate
from tools.t07_precision_scoring import score_rectangles
import pandas as pd


def test_compressed_score_equals_repo_pixel_metrics_for_overlap_and_clipping():
    rng=np.random.default_rng(42)
    for _ in range(20):
        gt=np.c_[rng.uniform(-3,50,(5,2)),rng.uniform(1,20,(5,2))]
        red=np.c_[rng.uniform(-3,50,(7,2)),rng.uniform(1,20,(7,2))]
        columns=['x','y','w','h'];g=pd.DataFrame(gt,columns=columns).assign(image_id='a',region_id=range(5))
        p=pd.DataFrame(red,columns=columns).assign(image_id='a')
        expected=over_redaction_rate(g,p,{'a':(64,64)})
        missed=pii_miss_rate(g,p,{'a':(64,64)})
        actual=score_rectangles({'a':(gt,_mask(gt,(64,64)))},{'a':p.to_dict('records')})
        assert actual['outside']==expected['numerator']
        assert actual['total']==expected['denominator']
        assert actual['missed']==missed['numerator']
