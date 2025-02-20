from flask import Flask, request, jsonify
import multiprocessing as mp
from enum import Enum
import logging
from typing import Callable, Dict, List, Union
from nemo_aligner.utils.verifiers.code_verifier_unsafe import CodeVerifier
import numpy as np

app = Flask(__name__)

PROCESS_COUNT = 32

class WorkerSignal(Enum):
    RUN = 0
    QUIT = 1

class TestType(Enum):
    IO_TEST = "io_test"
    ASSERTION = "assertion"

def verify_code_worker(verifier: CodeVerifier,
                      input_queue: mp.Queue,
                      output_queue: mp.Queue):
    while True:
        signal, idx, args = input_queue.get()
        if signal == WorkerSignal.RUN:
            code_str, test_data = args
            test_type = test_data["test_type"]
            
            if test_type == TestType.IO_TEST.value:
                tests = test_data["unittests"]
                fn_name = test_data.get("fn_name", None)
                results = verifier.verify(code_str, tests, fn_name)
            else:  # ASSERTION test
                assertions = test_data["unittests"]
                results = verifier.verify_assertions(code_str, assertions)

            # Calculate reward based on test results
            num_passed = sum(1 for r in results if r.get('passed', False))
            reward = num_passed / len(results) if results else 0.0
            
            # Include detailed results for debugging
            output_queue.put((idx, {
                "reward": reward,
            }))
        else:
            return

# Initialize queues and workers
submit_queue = mp.Queue()
result_queue = mp.Queue()
verifier = CodeVerifier(memory_limit_mb=256, cpu_time=10, timeout=10)

workers = []
for _ in range(PROCESS_COUNT):
    p = mp.Process(target=verify_code_worker, args=(verifier, submit_queue, result_queue))
    p.start()
    workers.append(p)

@app.route('/code_verifier', methods=['POST'])
def verify_code():
    """
    Endpoint to evaluate code submissions against test cases.
    Expects a JSON payload with:
    {
        "pred_responses": [code_str1, code_str2, ...],  # List of code strings to evaluate
        "test_data": [  # List of test configurations for each code submission
            {
                "test_type": "io_test",  # or "assertion"
                "unittests": [{"inputs": input1, "outputs": output1}, ...],  # for io_test / ["assert func(1) == 2", ...],  # for assertion
                "fn_name": "optional_fn_name",  # for io_test
            },
            ...
        ],
    }
    """
    try:
        data = request.get_json()
        responses = data.get("pred_responses")
        test_data = data.get("test_data")
        format_rewards = data.get("format_rewards", None)

        # Validate inputs
        if not responses or not test_data:
            return jsonify({"error": "Both 'pred_responses' and 'test_data' must be provided."}), 400
        if len(responses) != len(test_data):
            return jsonify({"error": "Number of responses must match number of test configurations."}), 400

        # Validate test types
        for test_config in test_data:
            if test_config.get("test_type") not in [t.value for t in TestType]:
                return jsonify({"error": f"Invalid test type: {test_config.get('test_type')}"}), 400
            if "unittests" not in test_config:
                return jsonify({"error": "test_data must include 'unittests' field"}), 400

        # Submit jobs to workers
        for idx, (code, test_config) in enumerate(zip(responses, test_data)):
            submit_queue.put((WorkerSignal.RUN, idx, (code, test_config)))

        # Collect results
        rewards = np.zeros(len(responses))
        for _ in range(len(responses)):
            idx, result = result_queue.get()
            rewards[idx] = result["reward"]

        # Convert format_rewards to numpy and reshape
        if format_rewards is not None:
            format_rewards = np.array(format_rewards)
        else:
            format_rewards = np.ones(len(responses))
        print("format rewards", format_rewards)
        print("rewards", rewards)
        # Multiply rewards with format rewards
        combined_rewards = rewards * format_rewards
        
        # Match the math grader format
        output_dict = {
            "rewards": combined_rewards.reshape((combined_rewards.shape[0], 1)).tolist(),
        }

        return jsonify(output_dict)

    except Exception as e:
        app.logger.error("An error occurred: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5568, debug=True)
