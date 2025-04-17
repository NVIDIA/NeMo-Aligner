from flask import Flask, request, jsonify
import numpy as np
import torch
import os
import signal
import atexit
import logging

from nemo_aligner.utils.verifiers.reward_model_verifier import RewardModelVerifier

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the reward model verifier
model_name = os.environ.get("REWARD_MODEL_PATH", "nvidia/Llama-3.1-Nemotron-70B-Reward-HF")
logger.info(f"Initializing reward model from {model_name}")

try:
    verifier = RewardModelVerifier(
        model_name=model_name,
        device="auto",
        dtype=torch.bfloat16
    )
    logger.info("Reward model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize reward model: {str(e)}")
    raise

@app.route('/reward_model_judge', methods=['POST'])
def verify_responses():
    """
    Endpoint to evaluate responses using reward model.
    Expects JSON payload with:
    {
        "prompts": [prompt1, prompt2, ...],
        "responses": [response1, response2, ...],
        "format_rewards": [reward1, reward2, ...]  # Optional format rewards
    }
    """
    try:
        data = request.get_json()
        prompts = data.get("prompts")
        responses = data.get("responses")
        format_rewards = data.get("format_rewards", [1.0] * len(prompts))

        # Validate inputs
        if not all([prompts, responses]):
            return jsonify({"error": "Missing required fields"}), 400
        if len(prompts) != len(responses):
            return jsonify({"error": "Length mismatch in inputs"}), 400

        # Get verification results from reward model
        results = verifier.verify(prompts, responses)
        
        # Extract raw reward scores
        reward_scores = np.array([r["reward_score"] for r in results])
        
        # Apply format rewards as a multiplier
        format_rewards = np.array(format_rewards)
        combined_rewards = np.where(format_rewards == 0, -100, reward_scores * format_rewards)
        
        # Log some results for debugging
        for i in range(min(3, len(prompts))):
            logger.info(f"Sample {i}:")
            logger.info(f"  Prompt: {prompts[i][:50]}...")
            logger.info(f"  Response: {responses[i][:50]}...")
            logger.info(f"  Raw score: {reward_scores[i]}")
            logger.info(f"  Combined reward: {combined_rewards[i]}")
        
        output_dict = {
            "rewards": combined_rewards.reshape((combined_rewards.shape[0], 1)).tolist(),
        }

        return jsonify(output_dict)

    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        port = int(os.environ.get("REWARD_MODEL_PORT", 5570))
        logger.info(f"Starting reward model server on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.error(f"Server error: {str(e)}")