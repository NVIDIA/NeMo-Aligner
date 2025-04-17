from flask import Flask, request, jsonify
import numpy as np
from nemo_aligner.utils.verifiers.genrm_verifier import GenRMVerifier
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize the GenRM verifier
verifier = GenRMVerifier()

@app.route('/genrm_verifier', methods=['POST'])
def verify_genrm():
    """
    Endpoint to evaluate GenRM score predictions.
    Expects JSON payload with:
    {
        "pred_responses": [response1, response2, ...],
        "metadata": [metadata1, metadata2, ...],
        "format_rewards": [reward1, reward2, ...] (optional)
    }
    
    Each metadata entry should contain:
    - "num_responses": 1 or 2
    - "helpfulness_1": ground truth score for first response
    - "helpfulness_2": ground truth score for second response (if num_responses=2)
    - "preference_ranking": ground truth preference ranking (if num_responses=2)
    """
    try:
        data = request.get_json()
        responses = data.get("pred_responses")
        metadata = data.get("metadata")
        format_rewards = data.get("format_rewards", [1.0] * len(responses))  # Default to 1.0 if not provided

        # Validate inputs
        if not all([responses, metadata]):
            return jsonify({"error": "Missing required fields"}), 400
        if not len(responses) == len(metadata):
            return jsonify({"error": "Length mismatch in inputs"}), 400

        # Get verification results
        results = verifier.verify(responses, metadata)
        
        # Extract scores from results
        scores = np.array([r["score"] for r in results])
        
        # Apply format rewards
        format_rewards = np.array(format_rewards)
        combined_rewards = scores * format_rewards
        
        app.logger.info(f"Scores: {scores}")
        app.logger.info(f"Format rewards: {format_rewards}")
        app.logger.info(f"Combined rewards: {combined_rewards}")
        
        output_dict = {
            "rewards": combined_rewards.reshape((combined_rewards.shape[0], 1)).tolist(),
        }

        return jsonify(output_dict)

    except Exception as e:
        app.logger.error("An error occurred: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5571, debug=True)
