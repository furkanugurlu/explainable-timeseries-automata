import json
import numpy as np
from typing import List, Dict, Any
from src.utils.distance import find_closest_pattern

class AutomataExplainer:
    """
    Provides step-by-step detailed traceability of Probabilistic Automata logic.
    Generates highly interpretable JSON structures justifying each decision.
    """
    def __init__(self, model):
        """
        Args:
            model: Instance of Trained ProbabilisticAutomata
        """
        self.model = model
        
    def analyze_sequence_step(self, current_idx: int, window_patterns: List[str]) -> Dict[str, Any]:
        """
        Deeply evaluates a single window of patterns and outputs a full decision breakdown.
        """
        num_steps = len(window_patterns)
        if num_steps < 2:
            raise ValueError("Window must contain at least 2 patterns for transition analysis.")
            
        # Step-by-step details collection
        path_prob = 1.0
        detailed_transitions = []
        used_probs = []
        state_statuses = []
        
        for i in range(num_steps - 1):
            orig_from = window_patterns[i]
            orig_to = window_patterns[i+1]
            
            # Detect mapping (Simulating internal model logic to get state-by-state trace)
            mapped_from = orig_from
            mapped_to = orig_to
            from_status = 'known'
            to_status = 'known'
            
            # Map from state
            if orig_from not in self.model.states and self.model.states:
                mapped_from = find_closest_pattern(orig_from, list(self.model.states))
                from_status = 'unseen'
            
            # Map to state
            if orig_to not in self.model.states and self.model.states:
                mapped_to = find_closest_pattern(orig_to, list(self.model.states))
                to_status = 'unseen'
                
            # Re-track state list to keep track of metadata for reporting
            state_statuses.append({
                'orig': orig_from,
                'mapped': None if from_status == 'known' else mapped_from,
                'status': from_status
            })
            
            # Get underlying probability
            prob = self.model.get_transition_prob(orig_from, orig_to)
            
            path_prob *= prob
            used_probs.append(prob)
            
            # Formatting specific text representation: 'aab -> abc : 0.72'
            tr_string = f"{mapped_from} -> {mapped_to} : {prob:.4f}"
            detailed_transitions.append(tr_string)
            
        # Include the final pattern metadata
        final_pattern = window_patterns[-1]
        final_mapped = final_pattern
        final_status = 'known'
        if final_pattern not in self.model.states and self.model.states:
            final_mapped = find_closest_pattern(final_pattern, list(self.model.states))
            final_status = 'unseen'
        state_statuses.append({
            'orig': final_pattern,
            'mapped': None if final_status == 'known' else final_mapped,
            'status': final_status
        })
            
        # Threshold from model or defaults
        threshold = self.model.anomaly_threshold
        is_anomaly = bool(path_prob < threshold)
        
        # Confidence Score Calculation: 
        # Geometric mean of transition probabilities (or average of transition probabilities) 
        # to avoid raw multiplication underflow bias.
        avg_prob = float(np.mean(used_probs)) if used_probs else 0.0
        confidence_score = avg_prob # 0 to 1 scale representing local node security

        # Construct the structured report for the target step (the end of window)
        # As per prompt: time_step, state, pattern, status, mapped_to, transitions, path_probability, decision, confidence_score
        # 'pattern' here will refer to final incoming pattern precipitating result.
        
        explanation = {
            "time_step": current_idx,
            "window_sequence": window_patterns,
            "state": window_patterns[-2] if len(window_patterns) >= 2 else window_patterns[0],
            "pattern": final_pattern,
            "status": final_status,
            "mapped_to": final_mapped if final_status == 'unseen' else None,
            "transitions": detailed_transitions,
            "path_probability": float(path_prob),
            "probability": float(path_prob),
            "decision": "anomaly" if is_anomaly else "normal",
            "confidence_score": round(confidence_score, 4)
        }
        
        return explanation

    def explain_anomalies(self, full_pattern_sequence: List[str], window_len: int = 5, json_output: bool = True) -> List[Dict[str, Any]]:
        """
        Loops through entire pattern sequence, performs explanation only on windows detected as anomalies.
        """
        all_explanations = []
        
        if len(full_pattern_sequence) < window_len:
            return []
            
        # Iterate through the windows
        for i in range(len(full_pattern_sequence) - window_len + 1):
            window = full_pattern_sequence[i : i + window_len]
            
            # Perform analysis
            result = self.analyze_sequence_step(current_idx=i, window_patterns=window)
            
            # Filter for Anomalies ONLY if preferred or keep all. 
            # The request implies analyzing the decision of model (especially for anomalies). 
            # We will keep all in result list, but users can filter downstream.
            if result['decision'] == "anomaly":
                all_explanations.append(result)
                
        if json_output:
            return json.dumps(all_explanations, indent=4)
            
        return all_explanations

if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    _log = _logging.getLogger(__name__)

    from src.models.automata_model import ProbabilisticAutomata

    model = ProbabilisticAutomata()
    model.states = {'abc', 'bcd', 'cda'}
    model.probabilities = {
        'abc': {'bcd': 0.8},
        'bcd': {'cda': 0.9},
        'cda': {'abc': 0.7}
    }
    model.anomaly_threshold = 0.5

    explainer = AutomataExplainer(model)

    _log.info("Generating sample explanation for sequence with one unknown transition...")
    test_seq = ['abc', 'bce', 'cda']

    explanation_json = explainer.explain_anomalies(test_seq, window_len=3)
    _log.info("--- EXPLANATION OUTPUT ---")
    _log.info(explanation_json)
