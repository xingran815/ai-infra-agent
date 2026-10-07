import subprocess
import os
import json
import sys
import time

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(THIS_FILE_DIR, "../")
EVAL_FILE_PATH = os.path.join(BASE_DIR, "checks/expected_results.json")

with open(EVAL_FILE_PATH, "r") as f:
    expected_results = json.load(f)

eval_results = {}
TIMEOUT = 300 # seconds
# TEST_CASES = ["01", "02", "03", "04"]
TEST_CASES = ["01"]
for case_id in TEST_CASES:
    print(f"Running test for case {case_id}...")
    eval_results[case_id] = {}
    eval_results[case_id]["passed"] = False
    eval_results[case_id]["completed"] = False
    eval_results[case_id]["tool_call_count"] = 0
    eval_results[case_id]["tools_used"] = []
    eval_results[case_id]["time_elapsed"] = 0

    time_start = time.perf_counter()
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy, sys; "
                "agent = runpy.run_path('test-agent-functions.py'); "
                "agent['diagnose_cluster_with_fixture'](sys.argv[1], prompt=None, eval=True)",
                case_id,
            ],
            cwd=BASE_DIR,
            timeout=TIMEOUT,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        eval_results[case_id]["error"] = f"Test case {case_id} timed out after {TIMEOUT} seconds."
        print(eval_results[case_id]["error"])
        continue

    print(f"result.stdout: {result.stdout}")
    time_elapsed = time.perf_counter() - time_start
    eval_results[case_id]["time_elapsed"] = time_elapsed

    if result.returncode != 0:
        eval_results[case_id]["return_code"] = result.returncode
        eval_results[case_id]["error"] = f"""Test failed with return code {result.returncode}, 
            stderr: {result.stderr.strip()}"""
        print(eval_results[case_id]["error"])
        continue
    eval_results[case_id]["return_code"] = result.returncode
    eval_results[case_id]["completed"] = True

    try:
        result_json = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        eval_results[case_id]["error"] = f"""Failed to parse JSON from stdout: {e}"""
        print(eval_results[case_id]["error"])
        continue

    eval_results[case_id]["tool_call_count"] = result_json.get("tools_call_count", 0)
    eval_results[case_id]["tools_used"] = result_json.get("tools_used", [])

    suspected_component = result_json.get('diagnosis', {}).get('suspected_component')
    if suspected_component is None:
        eval_results[case_id]["error"] = "No suspected component found in the result."
        print(eval_results[case_id]["error"])
        continue
    expected_component = expected_results[f'case_{case_id}']['expected component']

    if suspected_component.lower() != expected_component.lower():
        eval_results[case_id]["error"] = f"Test case {case_id} failed, expected suspected component: {expected_component}, got: {suspected_component}"
        print(eval_results[case_id]["error"])
        continue

    confidence = result_json.get('confidence')
    if confidence is None:
        eval_results[case_id]["error"] = "No confidence score found in the result."
        print(eval_results[case_id]["error"])
        continue
    elif confidence < 0.5:
        eval_results[case_id]["error"] = f"Test case {case_id} failed, confidence score too low: {confidence}"
        print(eval_results[case_id]["error"])
        continue

    eval_results[case_id]["passed"] = True
    print(f"case_{case_id} passed, expected={expected_component}, got={suspected_component}"
          f" in {time_elapsed:.2f} seconds, confidence: {result_json['confidence']}.")

failed_cases = [case_id for case_id, result in eval_results.items() if not result["passed"]]
summary = f"Summary: {len(eval_results) - len(failed_cases)}/{len(eval_results)} cases passed,"
f"accuracy: {(len(eval_results) - len(failed_cases)) / len(eval_results) * 100:.2f}%"
if failed_cases:
    summary += f"; failed: {', '.join(failed_cases)}"
print(summary)
