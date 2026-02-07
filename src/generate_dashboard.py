#!/usr/bin/env python3
"""Generate the GitHub Pages coverage dashboard and data.js from pipeline artifacts."""
import json
import os
import sys
import xml.etree.ElementTree as ET


def extract_file_coverage(coverage_xml_path):
    """Extract per-file coverage data from coverage.xml (Cobertura format)."""
    files = []
    if not os.path.exists(coverage_xml_path):
        return files
    try:
        tree = ET.parse(coverage_xml_path)
        for cls in tree.findall('.//class'):
            fn = cls.attrib.get('filename', '')
            lr = float(cls.attrib.get('line-rate', 0))
            lines = cls.findall('.//line')
            total = len(lines)
            hit = sum(1 for l in lines if int(l.attrib.get('hits', 0)) > 0)
            files.append({
                'f': fn, 's': total, 'h': hit,
                'm': total - hit, 'c': round(lr * 100, 1)
            })
    except Exception as e:
        print(f"Warning: coverage.xml parse error: {e}", file=sys.stderr)
    return files


def _load_materialization_map(pipeline_dir):
    """Load materialization map if it exists (maps old filenames to new)."""
    map_path = os.path.join(pipeline_dir, 'tests', 'generated',
                            '_materialization_map.json')
    if not os.path.exists(map_path):
        return None
    try:
        with open(map_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: materialization map parse error: {e}", file=sys.stderr)
        return None


def _detect_test_kind(nodeid):
    """Detect test kind (unit/integ/e2e) from a pytest nodeid."""
    import re
    m = re.search(r'test_(unit|integ|e2e)_', nodeid)
    return m.group(1) if m else 'unit'


def _transform_nodeid(nodeid, mat_map):
    """Transform a pytest nodeid using the materialization file map.

    Input:  tests/generated/test_unit_20250207_143022_01.py::test_foo
    Output: test_app.py::TestAppUnit::test_foo
    """
    if not mat_map:
        return nodeid
    file_map = mat_map.get('file_map', {})
    # Extract the filename from the nodeid (last path component before ::)
    parts = nodeid.split('::')
    file_part = parts[0]
    basename = os.path.basename(file_part)
    if basename in file_map:
        new_file = file_map[basename]
        parts[0] = new_file
        return '::'.join(parts)
    return nodeid


def extract_test_results(pipeline_dir):
    """Extract per-test results from pytest JSON report.

    If a materialization map exists, transforms nodeids to reflect the
    source-file-mapped test structure and adds a kind field (unit/integ/e2e).
    """
    mat_map = _load_materialization_map(pipeline_dir)
    tests = []
    for jf in ['.pytest_combined.json', '.pytest_generated.json', '.pytest_manual.json']:
        fpath = os.path.join(pipeline_dir, jf)
        if os.path.exists(fpath):
            try:
                with open(fpath) as f:
                    jdata = json.load(f)
                for t in jdata.get('tests', []):
                    nodeid = t.get('nodeid', '')
                    kind = _detect_test_kind(nodeid)
                    tests.append({
                        'n': _transform_nodeid(nodeid, mat_map),
                        'o': t.get('outcome', 'unknown'),
                        'd': round(t.get('duration', 0), 3),
                        'k': kind,
                    })
            except Exception as e:
                print(f"Warning: {jf} parse error: {e}", file=sys.stderr)
            break
    return tests


def compute_replacements(env):
    """Compute all template placeholder replacements from environment variables."""
    cov_after = float(env.get('COV_AFTER', '0'))
    cov_before = env.get('COV_BEFORE', '0')
    cov_delta = env.get('COV_DELTA', '0')
    tests_pass = int(env.get('TESTS_PASS', '0'))
    tests_fail = int(env.get('TESTS_FAIL', '0'))
    tests_total = int(env.get('TESTS_TOTAL', '0'))
    threshold_ok = env.get('THRESHOLD_OK', 'false')
    delta_ok = env.get('DELTA_OK', 'false')
    cov_int = int(cov_after)

    # Coverage hue and label
    if cov_int >= 90:
        cov_hue, cov_label = '142', 'Excellent'
    elif cov_int >= 70:
        cov_hue, cov_label = '60', 'Good'
    elif cov_int >= 50:
        cov_hue, cov_label = '30', 'Fair'
    else:
        cov_hue, cov_label = '0', 'Low'

    # SVG ring offset
    ring_offset = str(int(314 - (314 * cov_after / 100)))

    # Pass rate
    pass_rate = str(int(tests_pass / tests_total * 100)) if tests_total > 0 else '0'

    # Delta class
    try:
        d = float(cov_delta.replace('+', ''))
        delta_class = 'delta-positive' if d > 0 else ('delta-negative' if d < 0 else 'delta-neutral')
    except ValueError:
        delta_class = 'delta-neutral'

    # Fail color
    fail_color = 'var(--danger)' if tests_fail > 0 else 'var(--muted)'

    # Status
    status_class = 'status-passed' if threshold_ok == 'true' else 'status-failed'
    status_text = 'Quality Gate Passed' if threshold_ok == 'true' else 'Quality Gate Not Met'

    # Threshold badges
    threshold_badge = 'badge-pass' if threshold_ok == 'true' else 'badge-fail'
    threshold_text = 'PASSED' if threshold_ok == 'true' else 'FAILED'
    delta_badge = 'badge-pass' if delta_ok == 'true' else 'badge-fail'
    delta_text = 'PASSED' if delta_ok == 'true' else 'FAILED'

    return {
        '{{REPO_NAME}}': env.get('REPO_NAME', ''),
        '{{BRANCH}}': env.get('BRANCH', ''),
        '{{RUN_NUM}}': env.get('RUN_NUM', ''),
        '{{RUN_URL}}': env.get('RUN_URL', ''),
        '{{TIMESTAMP}}': env.get('TIMESTAMP', ''),
        '{{COV_BEFORE}}': cov_before,
        '{{COV_AFTER}}': env.get('COV_AFTER', '0'),
        '{{COV_DELTA}}': cov_delta,
        '{{COV_HUE}}': cov_hue,
        '{{COV_LABEL}}': cov_label,
        '{{RING_OFFSET}}': ring_offset,
        '{{TESTS_GEN}}': env.get('TESTS_GEN', '0'),
        '{{TESTS_TOTAL}}': str(tests_total),
        '{{TESTS_PASS}}': str(tests_pass),
        '{{TESTS_FAIL}}': str(tests_fail),
        '{{PASS_RATE}}': pass_rate,
        '{{FAIL_COLOR}}': fail_color,
        '{{MIN_COV}}': env.get('MIN_COV', '90'),
        '{{DELTA_REQ}}': env.get('DELTA_REQ', '0'),
        '{{STATUS_CLASS}}': status_class,
        '{{STATUS_TEXT}}': status_text,
        '{{DELTA_CLASS}}': delta_class,
        '{{THRESHOLD_BADGE}}': threshold_badge,
        '{{THRESHOLD_TEXT}}': threshold_text,
        '{{DELTA_BADGE}}': delta_badge,
        '{{DELTA_TEXT}}': delta_text,
    }


def main():
    if len(sys.argv) < 4:
        print("Usage: generate_dashboard.py <template_path> <output_dir> <pipeline_dir>",
              file=sys.stderr)
        sys.exit(1)

    template_path = sys.argv[1]
    output_dir = sys.argv[2]
    pipeline_dir = sys.argv[3]

    # Extract detailed data for data.js
    cov_xml = os.path.join(pipeline_dir, 'coverage.xml')
    file_data = extract_file_coverage(cov_xml)
    test_data = extract_test_results(pipeline_dir)

    # Write data.js
    data_js_content = 'window.PIPELINE_DATA=' + json.dumps(
        {'files': file_data, 'tests': test_data}, separators=(',', ':')
    ) + ';\n'
    data_js_path = os.path.join(output_dir, 'data.js')
    with open(data_js_path, 'w') as f:
        f.write(data_js_content)
    print(f"Generated data.js: {len(file_data)} files, {len(test_data)} tests")

    # Read template
    with open(template_path) as f:
        html = f.read()

    # Compute and apply all replacements
    replacements = compute_replacements(os.environ)
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)

    # Write final dashboard
    index_path = os.path.join(output_dir, 'index.html')
    with open(index_path, 'w') as f:
        f.write(html)
    print(f"Generated dashboard: {index_path}")


if __name__ == '__main__':
    main()
