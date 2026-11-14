"""Upload validation and intermediate exports, without writing real review history."""
import io
import json
import zipfile

import cv2
import numpy as np
import pytest
from PIL import Image

from tools import t07_walkthrough as walk


def encoded(image, fmt='PNG'):
    output=io.BytesIO();image.save(output,format=fmt);return output.getvalue()


def test_upload_preserves_aspect_flattens_transparency_and_strips_metadata():
    image=Image.new('RGBA',(2000,1000),(255,0,0,0))
    decoded,size=walk.decode_upload(encoded(image))
    assert size==(2000,1000) and decoded.size==(1280,640)
    assert decoded.mode=='RGB' and decoded.getpixel((0,0))==(255,255,255)
    exif=Image.Exif();exif[274]=6
    output=io.BytesIO();Image.new('RGB',(30,10)).save(output,format='JPEG',exif=exif)
    decoded,size=walk.decode_upload(output.getvalue())
    assert size==(10,30) and not decoded.getexif()


@pytest.mark.parametrize('raw',[b'',b'not an image',encoded(Image.new('RGB',(10,10)),'GIF')])
def test_invalid_upload_rejected_before_start(raw):
    with pytest.raises(ValueError):walk.decode_upload(raw)


def test_job_identifiers_cannot_escape_output(tmp_path,monkeypatch):
    monkeypatch.setattr(walk,'JOBS',tmp_path)
    for job in ['../README','A'*32,'x'*32,'a'*31]:
        with pytest.raises(ValueError):walk.folder_for(job)


def test_status_retries_windows_reader_lock_without_exposing_partial_json(tmp_path,monkeypatch):
    walk.write_status(tmp_path,{'stage':1})
    replace=walk.os.replace
    calls=[]
    def locked_once(source,target):
        calls.append(1)
        if len(calls)==1:
            assert json.loads(target.read_text())=={'stage':1}
            raise PermissionError('File open by reader')
        replace(source,target)
    monkeypatch.setattr(walk.os,'replace',locked_once)
    monkeypatch.setattr(walk.time,'sleep',lambda seconds:None)
    walk.write_status(tmp_path,{'stage':6})
    assert len(calls)==2 and json.loads((tmp_path/'status.json').read_text())=={'stage':6}


def test_worker_exports_actual_stages_and_pixels(tmp_path,monkeypatch):
    from src.common import offline
    from src import t07_multiscale
    monkeypatch.setattr(offline,'enable_offline_guard',lambda:None)
    monkeypatch.setattr(walk,'JOBS',tmp_path)
    box=dict(x=20,y=20,w=30,h=20,type='face',confidence=.9,source='test')
    class Detector:
        def __init__(self,config):pass
        def detect_trace(self,image,progress):
            progress(1,1)
            return {'raw':[box,box],'merged':[box],'views':[(0,0,80,60)]}
    monkeypatch.setattr(t07_multiscale,'MultiScaleDetectors',Detector)
    job='a'*32;folder=tmp_path/job;folder.mkdir()
    image=np.random.default_rng(42).integers(0,255,(60,80,3),dtype=np.uint8)
    cv2.imwrite(str(folder/'source.png'),image)
    (folder/'config.json').write_text(json.dumps(dict(mode='general',padding=0,original_size=[80,60],started=0)))
    walk.run(job)
    data=json.loads((folder/'status.json').read_text(encoding='utf-8'))
    assert data['state']=='complete' and data['raw_count']==2 and data['final_count']==1
    assert data['human_review'] is False and data['quality_metrics'] is None
    result=cv2.imread(str(folder/'result.png'))
    mask=np.ones(image.shape[:2],bool);mask[20:40,20:50]=False
    assert np.array_equal(result[mask],image[mask])
    assert not np.array_equal(result[~mask],image[~mask])
    for name in ['letterbox.png','tiles.png','detections.png','merged.png','padding.png']:
        assert cv2.imread(str(folder/name)) is not None
    with zipfile.ZipFile(folder/'result.zip') as archive:
        assert set(archive.namelist())=={'result.png','redactions.csv','report.json'}
