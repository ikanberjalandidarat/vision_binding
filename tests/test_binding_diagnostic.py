"""Non-Minecraft test doubles; no model/render validation implied."""
import json
from PIL import Image
from mc_binding.binding_data import binding_contexts
from mc_binding.binding_diagnostic import diagnostic_queries, parsed, summarize


def test_queries_budget_pairing_and_missing_targets():
    records=binding_contexts(0,731)
    assert sum(len(diagnostic_queries(r)) for r in records)==172
    for r in records:
        qs=diagnostic_queries(r)
        short=[q for q in qs if q['variant']=='original_8']
        long=[q for q in qs if q['variant']=='original_64']
        assert [(q['prompt'],q['expected']) for q in short]==[(q['prompt'],q['expected']) for q in long]
        if r['kind']=='absent':
            assert sum(q['expected']=='no' for q in qs if q['task']=='existence')==1
    assert parsed('Yes.','existence')=='yes'
    assert parsed('There is an arch.','shape') is None
    assert parsed('yes and no','existence') is None


def test_runner_outputs_resume_and_no_gate_changes(tmp_path,monkeypatch):
    from mc_binding import binding_diagnostic as bd
    from mc_binding.models import qwen
    records=binding_contexts(0,731)
    for i,r in enumerate(records):
        r.update(record_id=f'{i:06d}',image=f'{i:06d}.png')
        Image.new('RGB',(20,20)).save(tmp_path/r['image'])
    data={'records':records,'is_minecraft':False,'test_fixture':True}
    monkeypatch.setattr(bd,'load_binding',lambda root:(data,{'f0000':records}))
    class Fake:
        calls=0
        budgets=set()
        def __init__(self,config): self.config=config
        def manifest(self): return {'test_double':True}
        def inputs(self,im,p): return p
        def answer(self,inp):
            Fake.calls+=1;Fake.budgets.add(self.config['max_new_tokens']);return 'unfinished response'
    monkeypatch.setattr(qwen,'Qwen',Fake)
    config=dict(dtype='bfloat16',load_in_4bit=False,check_finite_scores=True,max_new_tokens=8)
    out=tmp_path/'result'
    bd.run(tmp_path,out,config,True)
    assert Fake.calls==172 and Fake.budgets=={8,64}
    assert config['max_new_tokens']==8
    assert json.loads((out/'status.json').read_text())=={'state':'complete','rows':172}
    summary=json.loads((out/'summary.json').read_text())
    assert sum(r['n'] for r in summary['counts'])==172
    assert sum(r['correct'] for r in summary['counts'])==0
    assert not (out/'gates').exists()
    assert 'data:image/png;base64,' in (out/'report.html').read_text()
    bd.run(tmp_path,out,config,True)
    assert Fake.calls==172
