"""Package recorded top-15 reviews and synthetic evidence; never create reviews or GT."""
from __future__ import annotations

import collections
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile

from src.common.io import sha256_file
from tools.t07_review_web import ROOT, ReviewStore, csv_bytes, read_csv


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def git(*args):
    result=subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}',*args],cwd=ROOT,
                          text=True,capture_output=True,check=True,encoding='utf-8')
    return result.stdout.strip()


def prepare():
    store=ReviewStore()
    queue=sorted(store.queue,key=lambda key:int(store.queue[key]['rank']))
    if len(queue)!=15 or any(key not in store.reviews for key in queue):
        raise ValueError('Need the recorded 15-item human queue; do not fabricate missing reviews')
    if any(store.reviews[key]['decision'] not in {'approve','edit'} for key in queue):
        raise ValueError('Queue contains unresolved decisions; inspect before packaging')
    evidence=ROOT/'reports/evidence';evidence.mkdir(exist_ok=True)
    output=ROOT/'out/submission_t07_review15';output.mkdir(exist_ok=True)
    (output/'redacted').mkdir(exist_ok=True)
    machine={key:[] for key in store.images}
    for row in read_csv(ROOT/'out/poc/redactions.csv'):
        machine[row['image_id']].append({**{k:int(float(row[k])) for k in ['x','y','w','h']},'type':row['type']})
    final={key:(store.boxes(key) if key in store.queue else machine[key]) for key in store.images}
    rows=[{'image_id':key,**box} for key,boxes in final.items() for box in boxes]
    predictions=csv_bytes(rows,['image_id','x','y','w','h','type'])
    (output/'redactions.csv').write_bytes(predictions)
    (ROOT/'reports/redactions_review15.csv').write_bytes(predictions)
    unchanged=[]
    for key in store.images:
        image=store.render(key,final[key])
        (output/'redacted'/f'{key}.png').write_bytes(image)
        if key not in store.queue:
            if image!=(ROOT/'out/poc/redacted'/f'{key}.png').read_bytes():
                raise ValueError(f'Outside-budget output changed: {key}')
            unchanged.append(key)
    reviewed=[]
    for key in queue:
        event=store.reviews[key]
        reviewed.append({'item_id':key,'rank':store.queue[key]['rank'],'score':store.queue[key]['score'],
            'split':store.images[key]['split'],'reason':store.queue[key]['reason'],
            'reviewer':event['actor'],'reviewer_decision':event['decision'],
            'override_value':json.dumps(event['after'],ensure_ascii=False), 'review_note':event['reason'],
            'reviewed_at':event['ts'],'web_snapshot':event['snapshot'],'review_source':'local_web'})
    columns=list(reviewed[0])
    (ROOT/'reports/review_queue.csv').write_bytes(csv_bytes(reviewed,columns))
    (output/'review_queue_reviewed.csv').write_bytes(csv_bytes(reviewed,columns))
    extras=[{'item_id':key,'split':store.images[key]['split'],'decision':event['decision'],
             'reviewer':event['actor'],'reviewed_at':event['ts'],'note':event['reason']}
            for key,event in store.reviews.items() if key not in store.queue]
    (evidence/'additional_web_reviews.csv').write_bytes(csv_bytes(extras,
        ['item_id','split','decision','reviewer','reviewed_at','note']))
    gate=store.gate()
    decision={**gate,'scope':'t07_review15_candidate','n_images':len(store.images),'review_budget':15,
              'known_rejected_ids':[r['item_id'] for r in extras if r['decision']=='reject'],
              'official_quality_metrics':None,'extra_reviews_applied':False,
              'note':'Export of recorded queue decisions only; extra reviews retained as known risks, not counted within B=15.'}
    write_json(output/'decision.json',decision)
    (ROOT/'reports/decision_log.csv').write_bytes(csv_bytes([decision],
        ['status','reason_code','owner','scope','completed','budget','known_rejected_ids','reason']))
    # Exact source journals retain their actors/timestamps. Packaging does not append an event.
    for source,name in [(ROOT/'out/web_review/audit_log.jsonl','audit_web.jsonl'),
                        (ROOT/'out/poc_ai_corrected/audit_log.jsonl','audit_corrections_and_confirmation.jsonl'),
                        (ROOT/'out/poc_ai_corrected/human_post_review.csv','human_post_review.csv')]:
        shutil.copyfile(source,evidence/name)
        shutil.copyfile(source,output/name)
    for name in ['baseline','poc','blur_all']:
        target=evidence/name;target.mkdir(exist_ok=True)
        for file in ['redactions.csv','run_info.json','decision.json']:
            shutil.copyfile(ROOT/'out'/name/file,target/file)
    target=evidence/'review15';target.mkdir(exist_ok=True)
    (target/'redactions.csv').write_bytes(predictions)
    inference=json.loads((evidence/'poc/run_info.json').read_text(encoding='utf-8'))
    write_json(target/'run_info.json',{'track':'t07','run_name':'poc_review15','status':0,'seed':42,
        'offline':True,'git_sha':inference['git_sha'],'runtime_seconds':None,
        'stats':{'n_items':len(store.images),'n_regions':len(rows),'review_budget':15},
        'source_inference':'reports/evidence/poc/run_info.json','export_only':True,
        'note':'Recorded human corrections replayed; inference runtime is reported separately; no end-to-end human time measured.'})
    for key in ['pii_10','pii_40','pii_08']:
        target=ROOT/'demo/t07_samples'/key;target.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(store.source_image(key),target/'source.jpg')
        for name in ['baseline','poc']:
            shutil.copyfile(ROOT/'out'/name/'redacted'/f'{key}.png',target/f'{name}.png')
        shutil.copyfile(output/'redacted'/f'{key}.png',target/'review15.png')
    manifest={'seed':42,'n_images':len(store.images),'review_budget':15,'n_regions':len(rows),
              'queue_ids':queue,'review_decisions':dict(collections.Counter(r['reviewer_decision'] for r in reviewed)),
              'review_splits':dict(collections.Counter(r['split'] for r in reviewed)),
              'reviewers':sorted({r['reviewer'] for r in reviewed}),
              'extra_reviews':extras,'unchanged_outside_queue':len(unchanged),'source_web_snapshot':store.fingerprint,
              'inference_source':'reports/evidence/poc/run_info.json','export_only':True,
              'predictions_sha256':sha256_file(output/'redactions.csv'),'quality_metrics':None,
              'provenance':'Machine inference then AI-assisted correction and recorded human web decisions; no new review action.'}
    write_json(output/'export_manifest.json',manifest)
    write_json(evidence/'review15_manifest.json',manifest)
    note=('T07 G07: recorded review15 candidate, not a validated release.\n'
          '60 synthetic images, seed42, dev36/eval24; only original top15 edits applied.\n'
          '45 outside-queue PNGs match original PoC byte-for-byte.\n'
          'Raw source journals are included for provenance; additional web decisions are not added to B=15.\n'
          'Known rejected cases remain; inspect decision.json. No GT metrics or new human decisions generated.\n'
          'Replay: python -m tools.prepare_t07_submission (requires original local run/review artifacts).\n'
          'Official evaluation: python -m src.evaluate --track t07 --pred reports/evidence/review15/redactions.csv --split eval\n')
    (output/'README.txt').write_text(note,encoding='utf-8')
    target=ROOT/'submissions/T07_G07_review15_candidate.zip';target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(output).as_posix())
    tests=ET.parse(ROOT/'out/submission_audit_tests.xml').getroot().find('testsuite')
    if tests is None:raise ValueError('Missing actual JUnit results')
    validation={'command':'python -m pytest -q -o addopts= --junitxml=out/submission_audit_tests.xml',
                'passed':int(tests.get('tests'))-int(tests.get('skipped'))-int(tests.get('failures'))-int(tests.get('errors')),
                'skipped':int(tests.get('skipped')),'failed':int(tests.get('failures'))+int(tests.get('errors')),
                'seconds':float(tests.get('time')),'source':'reports/evidence/pytest.xml'}
    shutil.copyfile(ROOT/'out/submission_audit_tests.xml',evidence/'pytest.xml')
    write_json(ROOT/'reports/t07_current_validation.json',validation)
    remote=json.loads((ROOT/'out/submission_audit_remote/commit.json').read_text(encoding='utf-8'))
    published=git('rev-parse','refs/remotes/origin/main')
    frozen=json.loads((ROOT/'reports/t07_protocol_freeze.json').read_text(encoding='utf-8'))
    audit={'observed_at':datetime.now().astimezone().isoformat(),'remote_main_commit':published,
           'audit_baseline_commit':remote['sha'],'remote_docs_snapshot':remote['sha'],'local_head':git('rev-parse','HEAD'),
           'local_commits_not_on_main':git('log','--format=%H %s',published+'..HEAD').splitlines(),
           'tags':git('tag','--list').splitlines(),'review15':manifest,'gate':gate,'tests':validation,
           'remote_report_state':'T07 code and evidence published' if published!=remote['sha'] else 'starter templates, config G00/T02',
           'frozen_file_current_match':{p:sha256_file(ROOT/p)==digest for p,digest in frozen['files'].items()},
           'missing_official_metrics':[p for p in ['metrics_baseline.json','metrics_poc.json','table1_baseline_vs_poc.md'] if not (ROOT/'reports'/p).exists()],
           'bundle':{'path':target.relative_to(ROOT).as_posix(),'bytes':target.stat().st_size,'sha256':sha256_file(target)},
           'ready_for_final_submission':False,'git_push_performed':False}
    write_json(ROOT/'reports/t07_submission_audit.json',audit)
    print(json.dumps({'candidate':str(target.relative_to(ROOT)),'regions':len(rows),'queue':len(reviewed),
                      'unchanged_outside_queue':len(unchanged),'tests':validation,'gate':gate['status']},ensure_ascii=True))
    return audit


if __name__=='__main__':prepare()
