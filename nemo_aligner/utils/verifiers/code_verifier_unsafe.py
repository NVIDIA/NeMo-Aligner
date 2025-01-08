import concurrent.futures
import json
import logging
import os
import resource
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm


class CodeVerifier:
    def __init__(self, memory_limit_mb=256, cpu_time=10, timeout=10):
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
        self.cpu_time = cpu_time
        self.timeout = timeout

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _prepare_test_files(self, code_str, test_input):
        tmp_dir = tempfile.mkdtemp()
        code_path = os.path.join(tmp_dir, "solution.py")
        input_path = os.path.join(tmp_dir, "input.txt")

        with open(code_path, "w") as f:
            f.write(code_str)
        with open(input_path, "w") as f:
            f.write(test_input)

        return tmp_dir, code_path, input_path

    def _set_resource_limits(self):
        resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit, -1))
        resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_time, -1))

    def _run_code(self, code_path, input_path):
        try:
            with open(input_path, "r") as input_file:
                process = subprocess.Popen(
                    ["python3", code_path],
                    stdin=input_file,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=self._set_resource_limits,
                )

                try:
                    output, error = process.communicate(timeout=self.timeout)
                    if process.returncode != 0:
                        return f"Error: {error.decode('utf-8')}"
                    return output.decode("utf-8")
                except subprocess.TimeoutExpired:
                    process.kill()
                    return f"Error: Timeout exceeded {self.timeout} seconds"
        except Exception as e:
            self.logger.error(f"Execution error: {str(e)}")
            return f"Error: {str(e)}"

    def verify(self, code_str, tests):
        def run_single_test(test_data):
            test_idx, test = test_data
            try:
                input_str = test["inputs"]
                expected_output = test["outputs"]

                tmp_dir, code_path, input_path = self._prepare_test_files(code_str, input_str)
                try:
                    actual_output = self._run_code(code_path, input_path)
                    actual_output = actual_output + "\n"

                    passed = (actual_output == expected_output) or (actual_output.strip() == expected_output.strip())
                    return {
                        "test_number": test_idx,
                        "passed": passed,
                        "actual_output": actual_output,
                        "expected_output": expected_output,
                    }
                finally:
                    import shutil

                    shutil.rmtree(tmp_dir)

            except Exception as e:
                self.logger.error(f"Test {test_idx} failed with error: {str(e)}")
                return {"test_number": test_idx, "passed": False, "error": str(e)}

        max_workers = min(len(tests), (os.cpu_count() or 1) * 2)
        test_data = list(enumerate(tests, 1))
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_test = {executor.submit(run_single_test, td): td for td in test_data}
            with tqdm(total=len(tests), desc="Running tests") as pbar:
                for future in concurrent.futures.as_completed(future_to_test):
                    result = future.result()
                    results.append(result)
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

    # Run verification
    results = verifier.verify(code, test_list)

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
