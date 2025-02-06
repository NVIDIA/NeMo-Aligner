import concurrent.futures
import json
import logging
import os
import resource
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm


class CodeVerifier:
    def __init__(self, memory_limit_mb=256, cpu_time=10, timeout=10):
        self.memory_limit = memory_limit_mb * 1024 * 1024
        self.cpu_time = cpu_time
        self.timeout = timeout
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _set_resource_limits(self):
        resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit, -1))
        resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_time, -1))

    def _create_temp_files(self, code_str, test_input=None, wrapper_code=None):
        tmp_dir = tempfile.mkdtemp()
        code_path = os.path.join(tmp_dir, "solution.py")

        with open(code_path, "w") as f:
            if wrapper_code:
                f.write(wrapper_code)
            else:
                f.write(code_str)

        input_path = None
        if test_input is not None:
            input_path = os.path.join(tmp_dir, "input.txt")
            with open(input_path, "w") as f:
                f.write(test_input)

        return tmp_dir, code_path, input_path

    def _run_process(self, code_path, input_path=None):
        try:
            process_args = {
                "args": ["python3", code_path],
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "preexec_fn": self._set_resource_limits,
            }

            if input_path:
                process_args["stdin"] = open(input_path, "r")  # Don't close this file immediately

            process = subprocess.Popen(**process_args)

            try:
                output, error = process.communicate(timeout=self.timeout)
                if process.returncode != 0:
                    return f"Error: {error.decode('utf-8')}"
                return output.decode("utf-8")
            finally:
                if input_path:
                    process_args["stdin"].close()  # Close the file after communication is done

        except subprocess.TimeoutExpired:
            process.kill()
            return f"Error: Timeout exceeded {self.timeout} seconds"
        except Exception as e:
            self.logger.error(f"Execution error: {str(e)}")
            return f"Error: {str(e)}"

    def _compare_outputs(self, actual, expected, atol=1e-5):
        try:
            # Try parsing as numbers for approximate comparison
            actual_val = float(actual.strip())
            expected_val = float(expected.strip())
            return np.isclose(actual_val, expected_val, atol=atol)
        except (ValueError, TypeError):
            # Fall back to string comparison
            return actual.strip() == expected.strip()

    def _create_function_wrapper(self, code_str, fn_name, input_args):
        # print(input_args)
        arg_list_str = ", ".join(repr(arg) for arg in input_args)
        return f"""
import json
{code_str}
result = {fn_name}({arg_list_str})
print(json.dumps(result))
"""

    def _run_single_test(self, test_data, code_str, fn_name=None):
        test_idx, test = test_data
        tmp_dir = None
        try:
            if fn_name:
                wrapper_code = self._create_function_wrapper(code_str, fn_name, test["inputs"])
                tmp_dir, code_path, _ = self._create_temp_files(code_str, wrapper_code=wrapper_code)
                actual_output = self._run_process(code_path)
                try:
                    actual_output = json.loads(actual_output)
                except json.JSONDecodeError:
                    return {
                        "test_number": test_idx,
                        "passed": False,
                        "actual_output": actual_output,
                        "expected_output": test["outputs"],
                    }
            else:
                tmp_dir, code_path, input_path = self._create_temp_files(code_str, test["inputs"])
                actual_output = self._run_process(code_path, input_path)
                actual_output = actual_output + "\n"

            passed = self._compare_outputs(str(actual_output), str(test["outputs"]))
            return {
                "test_number": test_idx,
                "passed": passed,
                "actual_output": actual_output,
                "expected_output": test["outputs"],
            }
        except Exception as e:
            self.logger.error(f"Test {test_idx} failed with error: {str(e)}")
            return {"test_number": test_idx, "passed": False, "error": str(e)}
        finally:
            if tmp_dir:
                import shutil

                shutil.rmtree(tmp_dir)

    def verify(self, code_str, tests, fn_name=None):
        max_workers = min(len(tests), (os.cpu_count() or 1) * 2)
        test_data = list(enumerate(tests, 1))
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_test = {executor.submit(self._run_single_test, td, code_str, fn_name): td for td in test_data}

            with tqdm(total=len(tests), desc="Running tests") as pbar:
                for future in concurrent.futures.as_completed(future_to_test):
                    results.append(future.result())
                    pbar.update(1)

        results.sort(key=lambda x: x["test_number"])
        return results

    def _create_assertion_wrapper(self, code_str, assertion):
        return f"""
{code_str}

try:
    {assertion}
    print("PASS")
except AssertionError:
    print("FAIL")
except Exception as e:
    print(f"ERROR: {{str(e)}}")
"""

    def _run_assertion_test(self, test_data, code_str):
        test_idx, assertion = test_data
        tmp_dir = None
        try:
            wrapper_code = self._create_assertion_wrapper(code_str, assertion)
            tmp_dir, code_path, _ = self._create_temp_files(wrapper_code)
            output = self._run_process(code_path)

            passed = output.strip() == "PASS"
            return {
                "test_number": test_idx,
                "passed": passed,
                "expected_output": assertion,
                "actual_output": output.strip(),
            }
        except Exception as e:
            self.logger.error(f"Assertion test {test_idx} failed with error: {str(e)}")
            return {"test_number": test_idx, "passed": False, "error": str(e)}
        finally:
            if tmp_dir:
                import shutil

                shutil.rmtree(tmp_dir)

    def verify_assertions(self, code_str, assertions):
        max_workers = min(len(assertions), (os.cpu_count() or 1) * 2)
        test_data = list(enumerate(assertions, 1))
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_test = {executor.submit(self._run_assertion_test, td, code_str): td for td in test_data}

            with tqdm(total=len(assertions), desc="Running assertion tests") as pbar:
                for future in concurrent.futures.as_completed(future_to_test):
                    results.append(future.result())
                    pbar.update(1)

        results.sort(key=lambda x: x["test_number"])
        return results


def main():
    # Example usage
    verifier = CodeVerifier()

    # Example code (the one you want to test)
    code = """
from collections import deque

def min_moves(x0, y0, x1, y1, segments):
    # Create a set of allowed cells
    allowed = set()
    for r, a, b in segments:
        for c in range(a, b + 1):
            allowed.add((r, c))

    # Check if the initial and final positions are allowed
    if (x0, y0) not in allowed or (x1, y1) not in allowed:
        return -1

    # Create a graph where each node represents an allowed cell
    graph = {cell: set() for cell in allowed}
    for r, c in allowed:
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in allowed:
                graph[(r, c)].add((nr, nc))

    # Perform BFS to find the shortest path
    queue = deque([(x0, y0, 0)])
    visited = set()
    while queue:
        r, c, dist = queue.popleft()
        if (r, c) == (x1, y1):
            return dist
        if (r, c) in visited:
            continue
        visited.add((r, c))
        for nr, nc in graph[(r, c)]:
            queue.append((nr, nc, dist + 1))

    return -1

# Read input
x0, y0, x1, y1 = map(int, input().split())
n = int(input())
segments = [tuple(map(int, input().split())) for _ in range(n)]

# Print the result
print(min_moves(x0, y0, x1, y1, segments))
    """
    prompt = "The black king is standing on a chess field consisting of 10^9 rows and 10^9 columns. We will consider the rows of the field numbered with integers from 1 to 10^9 from top to bottom. The columns are similarly numbered with integers from 1 to 10^9 from left to right. We will denote a cell of the field that is located in the i-th row and j-th column as (i, j).\n\nYou know that some squares of the given chess field are allowed. All allowed cells of the chess field are given as n segments. Each segment is described by three integers r_{i}, a_{i}, b_{i} (a_{i} ≤ b_{i}), denoting that cells in columns from number a_{i} to number b_{i} inclusive in the r_{i}-th row are allowed.\n\nYour task is to find the minimum number of moves the king needs to get from square (x_0, y_0) to square (x_1, y_1), provided that he only moves along the allowed cells. In other words, the king can be located only on allowed cells on his way.\n\nLet us remind you that a chess king can move to any of the neighboring cells in one move. Two cells of a chess field are considered neighboring if they share at least one point.\n\n\n-----Input-----\n\nThe first line contains four space-separated integers x_0, y_0, x_1, y_1 (1 ≤ x_0, y_0, x_1, y_1 ≤ 10^9), denoting the initial and the final positions of the king.\n\nThe second line contains a single integer n (1 ≤ n ≤ 10^5), denoting the number of segments of allowed cells. Next n lines contain the descriptions of these segments. The i-th line contains three space-separated integers r_{i}, a_{i}, b_{i} (1 ≤ r_{i}, a_{i}, b_{i} ≤ 10^9, a_{i} ≤ b_{i}), denoting that cells in columns from number a_{i} to number b_{i} inclusive in the r_{i}-th row are allowed. Note that the segments of the allowed cells can intersect and embed arbitrarily.\n\nIt is guaranteed that the king's initial and final position are allowed cells. It is guaranteed that the king's initial and the final positions do not coincide. It is guaranteed that the total length of all given segments doesn't exceed 10^5.\n\n\n-----Output-----\n\nIf there is no path between the initial and final position along allowed cells, print -1.\n\nOtherwise print a single integer — the minimum number of moves the king needs to get from the initial position to the final one.\n\n\n-----Examples-----\nInput\n5 7 6 11\n3\n5 3 8\n6 7 11\n5 2 5\n\nOutput\n4\n\nInput\n3 4 3 10\n3\n3 1 4\n4 5 9\n3 10 10\n\nOutput\n6\n\nInput\n1 1 2 10\n2\n1 1 3\n2 6 10\n\nOutput\n-1\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end."
    tests = {
        "ground_truth": '{\n "inputs": [\n "5 7 6 11\\n3\\n5 3 8\\n6 7 11\\n5 2 5\\n",\n "3 4 3 10\\n3\\n3 1 4\\n4 5 9\\n3 10 10\\n",\n "1 1 2 10\\n2\\n1 1 3\\n2 6 10\\n",\n "9 8 7 8\\n9\\n10 6 6\\n10 6 6\\n7 7 8\\n9 5 6\\n8 9 9\\n9 5 5\\n9 8 8\\n8 5 6\\n9 10 10\\n",\n "6 15 7 15\\n9\\n6 15 15\\n7 14 14\\n6 15 15\\n9 14 14\\n7 14 16\\n6 15 15\\n6 15 15\\n7 14 14\\n8 15 15\\n",\n "13 16 20 10\\n18\\n13 16 16\\n20 10 10\\n19 10 10\\n12 15 15\\n20 10 10\\n18 11 11\\n19 10 10\\n19 10 10\\n20 10 10\\n19 10 10\\n20 10 10\\n20 10 10\\n19 10 10\\n18 11 11\\n13 16 16\\n12 15 15\\n19 10 10\\n19 10 10\\n",\n "89 29 88 30\\n16\\n87 31 31\\n14 95 95\\n98 88 89\\n96 88 88\\n14 97 97\\n13 97 98\\n100 88 88\\n88 32 32\\n99 88 89\\n90 29 29\\n87 31 31\\n15 94 96\\n89 29 29\\n88 32 32\\n97 89 89\\n88 29 30\\n",\n "30 14 39 19\\n31\\n35 7 11\\n37 11 12\\n32 13 13\\n37 5 6\\n46 13 13\\n37 14 14\\n31 13 13\\n43 13 19\\n45 15 19\\n46 13 13\\n32 17 17\\n41 14 19\\n30 14 14\\n43 13 17\\n34 16 18\\n44 11 19\\n38 13 13\\n40 12 20\\n37 16 18\\n46 16 18\\n34 10 14\\n36 9 10\\n36 15 19\\n38 15 19\\n42 13 19\\n33 14 15\\n35 15 19\\n33 17 18\\n39 12 20\\n36 5 7\\n45 12 12\\n",\n "2 1 1 1\\n2\\n1 1 2\\n2 1 2\\n",\n "1 1 1 2\\n5\\n1000000000 1 10000\\n19920401 1188 5566\\n1000000000 1 10000\\n1 1 10000\\n5 100 200\\n",\n "1 1 1000000000 2\\n5\\n1000000000 1 10000\\n19920401 1188 5566\\n1000000000 1 10000\\n1 1 10000\\n5 100 200\\n"\n ],\n "outputs": [\n "4\\n",\n "6\\n",\n "-1\\n",\n "2\\n",\n "1\\n",\n "-1\\n",\n "1\\n",\n "9\\n",\n "1\\n",\n "1\\n",\n "-1\\n"\n ]\n}',
        "style": "rule",
    }
    test_list = []
    for i, o in zip(json.loads(tests["ground_truth"])["inputs"], json.loads(tests["ground_truth"])["outputs"]):
        test_list.append({"inputs": i, "outputs": o})

    code = """
def calculate_time(battery, charger):
    # Fast charge phase (0% to 85%)
    fast_charge_time = (battery * 0.85) / charger
    
    # Decreasing charge phase (85% to 95%)
    decreasing_charge_time = (battery * 0.10) / (charger * 0.5)
    
    # Trickle charge phase (95% to 100%)
    trickle_charge_time = (battery * 0.05) / (charger * 0.2)
    
    # Total charging time
    total_time = fast_charge_time + decreasing_charge_time + trickle_charge_time
    
    # Round to 2 decimal places
    total_time = round(total_time, 2)
    
    return total_time

"""
    test_list = []
    a = {
        "inputs": [[1000, 500], [1500, 500], [2000, 1000], [5000, 1000], [1000, 5000], [3050, 2600]],
        "outputs": [[2.6], [3.9], [2.6], [6.5], [0.26], [1.53]],
    }
    for i, o in zip(a["inputs"], a["outputs"]):
        test_list.append({"inputs": i, "outputs": ", ".join(repr(arg) for arg in o)})
    # Run verification
    results = verifier.verify(code, test_list, fn_name="calculate_time")

    # Print results
    num_pass = 0
    for result in results:
        test_num = result["test_number"]
        if result.get("passed"):
            print(f"Test #{test_num}: PASS")
            num_pass += 1
        else:
            print(f"Test #{test_num}: FAIL")
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print("Expected:")
                print(result["expected_output"])
                print("Got:")
                print(result["actual_output"])
    print("pass rate: ", num_pass / len(results))


if __name__ == "__main__":
    main()
