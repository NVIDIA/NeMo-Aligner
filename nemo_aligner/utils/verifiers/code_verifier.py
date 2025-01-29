# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

import concurrent.futures
import json
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import docker
from tqdm import tqdm


class CodeVerifier:
    def __init__(self, memory_limit="256m", cpu_limit="1.0", timeout=10):
        """
        Initialize the verifier with resource constraints
        memory_limit: Docker memory limit (e.g., "256m")
        cpu_limit: Docker CPU limit (e.g., "1.0" = 1 core)
        timeout: Timeout in seconds
        """
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.timeout = timeout
        self.client = docker.from_env()

        # Ensure the Docker image exists or build it
        self.image_name = "python-code-verifier"
        self._ensure_docker_image()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _ensure_docker_image(self):
        """Create a minimal Docker image for running Python code"""
        dockerfile = """
        FROM python:3.9-slim
        WORKDIR /app
        RUN pip install --no-cache-dir numpy==1.21.0
        """

        try:
            self.client.images.get(self.image_name)
        except docker.errors.ImageNotFound:
            with tempfile.TemporaryDirectory() as tmpdir:
                docker_path = Path(tmpdir) / "Dockerfile"
                docker_path.write_text(dockerfile)
                self.client.images.build(path=tmpdir, tag=self.image_name)

    def _prepare_test_files(self, code_str, test_input):
        """Prepare temporary files for code and input"""
        tmp_dir = tempfile.mkdtemp()

        # Write the code file
        code_path = os.path.join(tmp_dir, "solution.py")
        with open(code_path, "w") as f:
            f.write(code_str)

        # Write the input file
        input_path = os.path.join(tmp_dir, "input.txt")
        with open(input_path, "w") as f:
            f.write(test_input)

        return tmp_dir, code_path, input_path

    def _run_in_docker(self, code_path, input_path):
        """Run code in Docker container with resource limits"""
        container = None
        try:
            # Read input file content
            with open(input_path, "r") as f:
                input_content = f.read()

            container = self.client.containers.run(
                self.image_name,
                command=["python", "/app/solution.py"],
                volumes={code_path: {"bind": "/app/solution.py", "mode": "ro"},},
                mem_limit=self.memory_limit,
                cpu_period=100000,
                cpu_quota=int(float(self.cpu_limit) * 100000),
                network_mode="none",
                stdin_open=True,
                tty=True,
                detach=True,
            )

            # Write input to container
            container.exec_run("sh -c 'echo \"{}\" > /tmp/input'".format(input_content.replace('"', '\\"')))

            # Run python script with input and timeout
            start_time = time.time()
            result = container.exec_run(
                "sh -c 'cat /tmp/input | timeout {} python /app/solution.py'".format(self.timeout)
            )

            # Check if execution exceeded timeout
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Execution exceeded {self.timeout} seconds")

            return result.output.decode("utf-8").strip()

        except TimeoutError as e:
            self.logger.error(f"Timeout error: {str(e)}")
            return f"Error: Timeout exceeded {self.timeout} seconds"
        except docker.errors.ContainerError as e:
            self.logger.error(f"Container error: {str(e)}")
            return f"Error: {str(e)}"
        except Exception as e:
            self.logger.error(f"Execution error: {str(e)}")
            return f"Error: {str(e)}"
        finally:
            # Ensure container is always cleaned up
            if container:
                # Attempt to stop the container
                try:
                    container.stop(timeout=1)
                except:
                    pass  # Stopping may fail if the container is already stopped

                # Attempt to remove regardless of stop success
                try:
                    container.remove(force=True)
                except:
                    pass

    def verify(self, code_str, tests):
        """
        Verify code against test cases using multiple threads
        
        code_str: String containing Python code
        tests: List of dictionaries with 'inputs' and 'outputs' keys
        """

        def run_single_test(test_data):
            test_idx, test = test_data
            try:
                # Prepare input
                input_str = test["inputs"]
                expected_output = test["outputs"]

                # Create temporary files
                tmp_dir, code_path, input_path = self._prepare_test_files(code_str, input_str)

                try:
                    # Run code in Docker
                    actual_output = self._run_in_docker(code_path, input_path)
                    actual_output = actual_output + "\n"

                    # Compare outputs
                    passed = (actual_output == expected_output) or (actual_output.strip() == expected_output.strip())
                    return {
                        "test_number": test_idx,
                        "passed": passed,
                        "actual_output": actual_output,
                        "expected_output": expected_output,
                    }
                finally:
                    # Cleanup
                    import shutil

                    shutil.rmtree(tmp_dir)

            except Exception as e:
                self.logger.error(f"Test {test_idx} failed with error: {str(e)}")
                return {"test_number": test_idx, "passed": False, "error": str(e)}

        # Calculate optimal number of workers based on CPU cores and test count
        max_workers = min(len(tests), (os.cpu_count() or 1) * 2)

        # Create a list of test data with indices
        test_data = list(enumerate(tests, 1))

        # Run tests in parallel using ThreadPoolExecutor
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_test = {executor.submit(run_single_test, td): td for td in test_data}

            # Use tqdm to show progress
            with tqdm(total=len(tests), desc="Running tests") as pbar:
                for future in concurrent.futures.as_completed(future_to_test):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)

        # Sort results by test number to maintain order
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
