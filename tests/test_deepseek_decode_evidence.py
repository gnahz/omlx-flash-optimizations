"""Verify public evidence integrity and the arithmetic behind release claims."""
import hashlib
import json
from pathlib import Path
from statistics import mean
import unittest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'results/qualification/deepseek/decode-20261015'


def load(name):
    return json.loads((EVIDENCE / name).read_text())


def canonical_sha(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


class DecodeEvidenceTests(unittest.TestCase):
    def test_released_file_hashes_and_sanitization(self):
        manifest = load('manifest.json')
        self.assertTrue(manifest['source_paths_sanitized'])
        self.assertFalse(manifest['full_inputs_released'])
        self.assertFalse(manifest['full_server_logs_released'])
        sources = {entry['path']: entry['source_sha256'] for entry in manifest['sources']}
        for entry in manifest['files']:
            path = EVIDENCE / entry['path']
            self.assertTrue(path.resolve().is_relative_to(EVIDENCE))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry['released_sha256'])
            if entry['exact_original_bytes']:
                self.assertEqual(len(entry['source_artifacts']), 1)
                self.assertEqual(entry['released_sha256'], sources[entry['source_artifacts'][0]])
            text = path.read_text()
            self.assertNotIn('/Users/', text)
            self.assertNotIn('Authorization: Bearer', text)
        for source in manifest['sources']:
            self.assertFalse(Path(source['path']).is_absolute())
            self.assertRegex(source['source_sha256'], r'^[0-9a-f]{64}$')

    def test_growth_claim_and_complete_parity(self):
        data, summary = load('growth-rounds.json'), load('summary.json')['growth_1m']
        self.assertEqual([r['arm'] for r in data['rounds']], ['baseline', 'fixed', 'fixed', 'baseline'])
        self.assertTrue(all(data['parity'].values()))
        for row in data['rounds']:
            self.assertEqual(row['controls'], {'packed_cache_growth': row['arm'] == 'fixed',
                             'packed_cache_growth_pingpong': True,
                             'preserved_prefill_max_defer_seconds': 0, 'terminal_response_barrier': False})
        for field in ('payload_sha256', 'payload_file_sha256', 'output_sha256', 'output_token_ids_sha256',
                      'output_tokens', 'prompt_tokens', 'reused_tokens', 'prefill_chunks', 'non_timing_usage'):
            self.assertTrue(all(r[field] == data['rounds'][0][field] for r in data['rounds']))
        baseline, fixed = [mean(r['generation_tps'] for r in data['rounds'] if r['arm'] == arm)
                           for arm in ('baseline', 'fixed')]
        self.assertAlmostEqual((fixed / baseline - 1) * 100, summary['generation_tps_increase_percent'])
        self.assertEqual((summary['requested_max_tokens'], summary['actual_output_tokens']), (512, 259))

    def test_concurrent_claim_and_complete_parity(self):
        data, summary = load('concurrency-rounds.json'), load('summary.json')['concurrent_decode']
        baseline, fixed = [[r for r in data['rounds'] if r['arm'] == arm] for arm in ('baseline', 'fixed')]
        self.assertEqual([r['arm'] for r in data['rounds']], ['baseline', 'fixed', 'fixed', 'baseline'])
        self.assertAlmostEqual(mean(r['A']['metadata']['generation_tokens_per_second'] for r in fixed) /
                               mean(r['A']['metadata']['generation_tokens_per_second'] for r in baseline), summary['A_tps_ratio'])
        for arm, rows in [('baseline', baseline), ('fixed', fixed)]:
            self.assertAlmostEqual(mean(r['A']['client_full_response_seconds'] for r in rows), summary[arm + '_A_full_response_seconds'])
        for comparison in data['comparisons']:
            self.assertTrue(comparison['passed'])
            self.assertTrue(all(all(check.values()) for check in comparison['checks'].values()))
        for row in data['rounds']:
            self.assertEqual(row['controls'], {'packed_cache_growth': row['arm'] == 'fixed',
                             'packed_cache_growth_pingpong': True,
                             'preserved_prefill_max_defer_seconds': 8 if row['arm'] == 'fixed' else 0,
                             'terminal_response_barrier': row['arm'] == 'fixed'})
            self.assertTrue(row['overlap_observed'])
            for name in ('seed', 'reference', 'A', 'B'):
                self.assertTrue(row[name]['sse_done'])
                self.assertEqual(row[name]['token_match_count'], 1)
                self.assertFalse(row[name]['errors'])
                self.assertEqual(row[name]['identity'], data['rounds'][0][name]['identity'])
            self.assertTrue(row['B']['known_concurrent_validation'])

    def test_public_full_outputs_match_canonical_hashes(self):
        cases = load('input-output-summary.json')['cases']
        self.assertEqual(len(cases), 3)
        growth = load('growth-rounds.json')['rounds'][0]
        concurrent = load('concurrency-rounds.json')['rounds'][0]
        expected = {'growth-1m': {'payload_sha256': growth['payload_sha256'],
                                  'output_sha256': growth['output_sha256'],
                                  'token_ids_sha256': growth['output_token_ids_sha256']},
                    'concurrent-A': concurrent['A']['identity'], 'concurrent-B': concurrent['B']['identity']}
        for case in cases:
            self.assertFalse(case['input']['full_input_released'])
            self.assertEqual(canonical_sha(load(case['output']['path'])), case['output']['canonical_sha256'])
            self.assertRegex(case['output']['token_ids_sha256'], r'^[0-9a-f]{64}$')
            self.assertEqual(case['input']['payload_sha256'], expected[case['label']]['payload_sha256'])
            self.assertEqual(case['output']['canonical_sha256'], expected[case['label']]['output_sha256'])
            self.assertEqual(case['output']['token_ids_sha256'], expected[case['label']]['token_ids_sha256'])

    def test_original_qualification_counts_and_six_files(self):
        data = load('production-qualification.json')
        counts = data['cpu_tests']
        self.assertEqual(sum(counts[k] for k in ('scheduler', 'engine_terminal', 'growth', 'local_launcher')), counts['passed'])
        self.assertEqual(counts['passed'], 791)
        self.assertEqual(counts['existing_async_tests_not_executed'], 47)
        self.assertEqual(data['runtime_pin_count'], 601)
        self.assertEqual(len(data['qualified_patch_files_sha256']), 6)
        self.assertTrue(all(r['passed'] for r in data['concurrency_and_restart']))
        self.assertFalse(load('summary.json')['portable_release_full_model_rerun'])

    def test_fairness_is_scoped_and_observed(self):
        data = load('long-decode-fairness.json')
        self.assertTrue(data['A_generation_exceeded_8_seconds'])
        self.assertTrue(data['B_full_chunk_completed_before_A_generation_finished'])
        self.assertIn(8192, data['B_processed_tokens_while_A_generating'])
        self.assertEqual(data['A']['identity']['actual_completion_tokens'], 2048)
        self.assertTrue(any('indivisible' in s for s in load('summary.json')['limits']))


if __name__ == '__main__':
    unittest.main()
