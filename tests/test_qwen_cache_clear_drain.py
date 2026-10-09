"""Cold clearing waits for asynchronous renames before deleting cache files."""
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark_qwen as benchmark

def runtime(persisted=1, errors=0, pending=0):
    return {'models': [{'id': 'model', 'saves': 1, 'saves_persisted': persisted,
        'errors': errors, 'gdn_staging': {'pending_bytes': pending}}],
        'total_num_files': 0, 'total_size_bytes': 0, 'hot_cache_entries': 0, 'hot_cache_size_bytes': 0}

def test_pending_main_save_gdn_and_temporary_files_prevent_clear_readiness():
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        assert not benchmark.writer_quiet(benchmark.writer_snapshot(runtime(persisted=0), cache))
        assert not benchmark.writer_quiet(benchmark.writer_snapshot(runtime(pending=64), cache))
        (cache / 'block_tmp.safetensors').write_bytes(b'incomplete')
        assert not benchmark.writer_quiet(benchmark.writer_snapshot(runtime(), cache))
        (cache / 'block_tmp.safetensors').unlink()
        assert benchmark.writer_quiet(benchmark.writer_snapshot(runtime(), cache))
        with pytest.raises(RuntimeError, match='writer reported errors'):
            benchmark.writer_quiet(benchmark.writer_snapshot(runtime(errors=1), cache))

def test_clear_occurs_after_persisted_counts_are_stably_complete():
    with tempfile.TemporaryDirectory() as tmp:
        client = benchmark.Client(8013, 'test-key', Path(tmp) / 'out', Path(tmp) / 'cache')
        events = []; snapshots = iter([runtime(persisted=0), runtime(persisted=0), runtime(), runtime(), runtime(), runtime()])
        def stats():
            value = next(snapshots, runtime()); events.append(('stats', value['models'][0]['saves_persisted']))
            return {'runtime_cache': value}
        def command(path, data):
            events.append(('clear', path)); return {'status': 'ok'}
        client.stats, client.json = stats, command
        with patch.object(benchmark.time, 'sleep'):
            client.clear()
        first_clear = next(i for i, event in enumerate(events) if event[0] == 'clear')
        assert all(event == ('stats', 1) for event in events[first_clear - 4:first_clear])
        assert len([event for event in events if event[0] == 'clear']) == 2
