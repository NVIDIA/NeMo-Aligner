from flask import Flask, request, jsonify
import numpy as np
from nemo_aligner.utils.verifiers.llm_verifier import LLMVerifier
import subprocess
import time
import os
import signal
import atexit

app = Flask(__name__)

def start_vllm_server(model_path, port=5000):
    """Start the vLLM server as a subprocess"""
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--trust-remote-code",
        "--seed=1",
        "--host=0.0.0.0",
        "--port=5000",
        "--served-model-name", "model",
        "--tensor-parallel-size=8",
        "--max-model-len=32768",
        "--gpu-memory-utilization", "0.95",
        "--enforce-eager",
        "--disable-log-requests",
        "--download-dir=/lustre/fsw/portfolios/llmservice/users/jiaqiz/hf_cache"
    ]
    
    process = subprocess.Popen(cmd)
    
    # Register cleanup function
    def cleanup():
        process.send_signal(signal.SIGTERM)
        process.wait()
    atexit.register(cleanup)
    
    # Wait for server to start
    time.sleep(300)  # Adjust based on model loading time
    return process

# Start vLLM server
vllm_process = start_vllm_server(os.environ.get("MODEL_PATH", "meta-llama/Llama-3.1-70B-Instruct"))

# Initialize verifier with vLLM API endpoint
verifier = LLMVerifier(
    api_base="http://localhost:5000/v1",
    max_tokens=256,
    temperature=0.0
)

@app.route('/llm_judge', methods=['POST'])
def verify_responses():
    """
    Endpoint to evaluate responses using LLM judge.
    Expects JSON payload with:
    {
        "prompts": [prompt1, prompt2, ...],
        "responses": [response1, response2, ...],
        "ground_truths": [gt1, gt2, ...]
    }
    """
    try:
        data = request.get_json()
        prompts = data.get("prompts")
        responses = data.get("responses")
        ground_truths = data.get("ground_truths")

        # Validate inputs
        if not all([prompts, responses, ground_truths]):
            return jsonify({"error": "Missing required fields"}), 400
        if not len(prompts) == len(responses) == len(ground_truths):
            return jsonify({"error": "Length mismatch in inputs"}), 400

        # Get verification results
        results = verifier.verify(prompts, responses, ground_truths)
        
        # Convert to rewards format
        rewards = np.array([1.0 if r["passed"] else 0.0 for r in results])
        
        full_responses = [r["full_response"] for r in results]
        print(full_responses)
        # todo: add format reward
        
        output_dict = {
            "rewards": rewards.reshape((rewards.shape[0], 1)).tolist(),
        }

        return jsonify(output_dict)

    except Exception as e:
        app.logger.error("An error occurred: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5569, debug=False)  # Set debug=False when running vLLM server
    finally:
        # Cleanup vLLM server
        vllm_process.send_signal(signal.SIGTERM)
        vllm_process.wait()