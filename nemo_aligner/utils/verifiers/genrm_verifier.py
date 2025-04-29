from typing import List, Dict, Any
import logging
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


class GenRMVerifier:
    def __init__(self):
        pass
    
    def extract_answer(self, string):
        """Extract Answer String from \\boxed expression."""
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            retval = None
        else:
            retval = string[idx : right_brace_idx + 1]

        if retval:
            left = "\\boxed{"
            try:
                assert retval[: len(left)] == left
                assert retval[-1] == "}"
                return retval[len(left) : -1]
            except AssertionError:
                return None

        return None

    def distance_abs(self, a, b):
        try:
            d = abs(int(a) - int(b))
        except Exception as e:
            logging.error(f"Error calculating distance: {e}, a: {a}, b: {b}")
            d = 100
        return d
    
    def verify(self, responses: List[str], 
               metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Verify GenRM score predictions
        
        Args:
            responses: List of model responses to verify
            metadata: List of metadata for each response
            
        Returns:
            List of dictionaries containing verification results
        """
        results = []
        
        for response, meta in zip(responses, metadata):
            distance = 0
            try:
                # get individual helpfulness scores
                indidual_scores_paragraph = response.split("[The Begin of Individual Scores]")[-1].split("[The End of Individual Scores]")[0]
                individual_scores = self.extract_answer(indidual_scores_paragraph).split(",")
                if meta["num_responses"] == 1:
                    gt_individual = meta["helpfulness_1"] 
                    distance = self.distance_abs(individual_scores[0], gt_individual)
                elif meta["num_responses"] == 2:
                    gt_individual_1 = meta.get("helpfulness_1", None)
                    gt_individual_2 = meta.get("helpfulness_2", None)
                    distance = 0
                    if gt_individual_1 is not None and gt_individual_2 is not None:
                        distance = self.distance_abs(individual_scores[0], gt_individual_1) + self.distance_abs(individual_scores[1], gt_individual_2)

                    # get preference ranking score
                    preference_ranking_paragraph = response.split("[The Begin of Ranking Score]")[-1].split("[The End of Ranking Score]")[0]
                    preference_ranking = self.extract_answer(preference_ranking_paragraph)
                    gt_preference_ranking = meta["preference_ranking"]
                    distance += self.distance_abs(preference_ranking, gt_preference_ranking)

                else:
                    raise ValueError(f"Unsupported number of responses for genrm: {meta['num_responses']}")
                
                
                    
                results.append({
                    "score": -distance
                })
            
            except Exception as e:
                logging.error(f"Error verifying response: {response}")
                logging.error(f"Error: {e}")
                results.append({
                    "score": -100
                })
        return results