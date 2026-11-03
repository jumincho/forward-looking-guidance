"""Validate immutable shared plans before selecting a question shard."""
from pathlib import Path
from flg.io import read_jsonl, write_jsonl, digest
from flg.pipeline import seed_for

def supply_plans(source, questions, assigned, output, sampling):
    rows = read_jsonl(source)
    questions = {str(row['id']): row['question'] for row in questions}
    plans = {str(row['id']): row for row in rows}
    if len(rows) != len(plans) or set(plans) != set(questions):
        raise ValueError('Supplied plans must cover exactly the input questions once')
    for qid, row in plans.items():
        if row['question_sha256'] != digest(questions[qid]) or row['plan_sha256'] != digest(row['plan']):
            raise ValueError('Supplied question or plan hash differs')
        if row.get('seed') != seed_for(42, qid, 'planner'):
            raise ValueError('Supplied planning seed differs')
        if any(row['backend'].get(key) != sampling[key] for key in ['top_p', 'top_k']):
            raise ValueError('Supplied planning decoder differs')
        if row['backend'].get('temperature') != 1 or row['backend'].get('max_tokens') != 512:
            raise ValueError('Supplied planning generation budget differs')
        if row['backend'].get('enable_thinking') is not False:
            raise ValueError('Supplied plans require thinking disabled')
    selected = [plans[str(row['id'])] for row in assigned]
    path = Path(output)
    if path.exists():
        if read_jsonl(path) != selected:
            raise ValueError('Existing shared-plan checkpoint differs')
    else:
        write_jsonl(path, selected)
    return selected
