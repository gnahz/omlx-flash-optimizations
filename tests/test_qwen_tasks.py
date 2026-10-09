"""Frozen task schedules and rejection checks, independent of model inference."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark_qwen as benchmark

def contract_text(fixture):
    contract = copy.deepcopy(fixture['contract'])
    if contract['kind'] == 'python-rename':
        text = contract['original_source'].replace('calculate_total', 'calculate_order_total')
    else:
        text = json.dumps(contract['expected_json'])
    contract['historical_output_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    return text, contract

def test_frozen_historical_geometry_and_twenty_measured_task_requests():
    frozen = benchmark.load_fixtures()['tasks']
    assert frozen['archive_report_sha256'] == '12976e4aab5374cca575347122d3c52a30bc3a4a5283591e2f4fd0c8ba64ecb4'
    fixtures = frozen['fixtures']
    assert [(f['metadata']['id'], f['metadata']['local_prompt_tokens'], f['contract']['historical_completion_tokens'])
        for f in fixtures] == [('edit_file', 1344, 1278), ('json_change', 880, 822)]
    class Recorder:
        def __init__(self): self.calls = []
        def request(self, *args, **kwargs): self.calls.append((args, kwargs)); return {}, ''
    recorder = Recorder()
    benchmark.tasks(recorder, benchmark.load_fixtures())
    assert len(recorder.calls) == 12
    assert sum(k['measured'] for _, k in recorder.calls) == 10
    assert all(k['cold'] and k['contract'] for _, k in recorder.calls)
    assert [a[1] for a, k in recorder.calls if not k['measured']] == ['edit_file-warmup', 'json_change-warmup']

def test_python_only_requested_rename_preserves_entire_program_contract():
    fixture = benchmark.load_fixtures()['tasks']['fixtures'][0]
    text, contract = contract_text(fixture)
    assert all(benchmark.validate_task(text, contract, 1278, 'stop').values())
    changed = text.replace('Decimal("4.99")', 'Decimal("5.99")')
    contract['historical_output_sha256'] = hashlib.sha256(changed.encode()).hexdigest()
    checks = benchmark.validate_task(changed, contract, 1278, 'stop')
    assert checks['historical_whole_output']
    assert not checks['requested_ast_change_only']
    assert not all(benchmark.validate_task(text.replace('calculate_order_total', 'calculate_total'), contract, 1278, 'stop').values())

def test_truncated_or_early_stop_output_is_not_a_valid_throughput_sample():
    fixture = benchmark.load_fixtures()['tasks']['fixtures'][0]
    text, contract = contract_text(fixture)
    assert not benchmark.validate_task(text, contract, 1277, 'stop')['historical_completion_tokens']
    assert not benchmark.validate_task(text, contract, 1278, 'length')['normal_stop']
    assert not benchmark.validate_task(text[:-1], contract, 1278, 'stop')['historical_whole_output']

def test_json_change_retains_all_fields_and_exact_json_types():
    fixture = benchmark.load_fixtures()['tasks']['fixtures'][1]
    text, contract = contract_text(fixture)
    assert all(benchmark.validate_task(text, contract, 822, 'stop').values())
    changed = copy.deepcopy(contract['expected_json'])
    changed['service']['request_timeout_seconds'] = 45.0
    changed_text = json.dumps(changed)
    contract['historical_output_sha256'] = hashlib.sha256(changed_text.encode()).hexdigest()
    assert not benchmark.validate_task(changed_text, contract, 822, 'stop')['exact_json_change']
    duplicate = '{"service":{},"service":{}}'
    contract['historical_output_sha256'] = hashlib.sha256(duplicate.encode()).hexdigest()
    assert not benchmark.validate_task(duplicate, contract, 822, 'stop')['exact_json_change']
