import logging
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
from src.utils.distance import find_closest_pattern

logger = logging.getLogger(__name__)

class ProbabilisticAutomata:
    """
    Builds a frequency-based Probabilistic Automata derived from SAX patterns.
    Transitions are tracked between consecutive sequential patterns.
    """
    def __init__(self, smoothing_epsilon: float = 1e-5):
        # Transition counts: transitions[from_state][to_state]
        self.transitions = defaultdict(Counter)
        # Transition probabilities
        self.probabilities = defaultdict(dict)
        # Unique set of observed states
        self.states = set()
        # Pseudocount for unseen transitions to prevent strict zero probabilities
        self.epsilon = smoothing_epsilon
        # Automatic threshold computed during fitting
        self.anomaly_threshold = 0.0

    def fit(self, patterns: List[str], window_len: int = 4):
        """
        Learns transition matrix from a sequence of states (patterns).
        Consecutive elements in the list represent a sequence step.
        window_len controls the sliding window size used to compute the training threshold.
        """
        if len(patterns) < 2:
            raise ValueError("Pattern sequence too short to build transitions.")

        # Reset state
        self.transitions.clear()
        self.probabilities.clear()
        self.states = set(patterns)

        # Count transitions between adjacent patterns
        # Pattern[t] -> Pattern[t+1]
        for i in range(len(patterns) - 1):
            current_state = patterns[i]
            next_state = patterns[i + 1]
            self.transitions[current_state][next_state] += 1

        # Convert counts to frequency-based probabilities
        for from_state, to_counts in self.transitions.items():
            total_out = sum(to_counts.values())
            for to_state, count in to_counts.items():
                self.probabilities[from_state][to_state] = count / total_out

        logger.info(f"Automata fitted. States: {len(self.states)}")

        train_probs = self.calculate_sequence_probabilities(patterns, window_len=window_len)
        if train_probs:
            self.anomaly_threshold = np.percentile(train_probs, 5)
            logger.info(f"Anomaly threshold (5th pct): {self.anomaly_threshold:.4e}")

    def get_transition_prob(self, from_state: str, to_state: str) -> float:
        """
        Returns transition probability.
        If states are unseen, uses Levenshtein distance to map to closest known state 
        to allow processing continuity, then applies a penalty/smoothing.
        """
        mapped_from = from_state
        mapped_to = to_state
        
        # 1. Unseen State Remediation via Levenshtein Distance
        if from_state not in self.states and self.states:
            mapped_from = find_closest_pattern(from_state, list(self.states))
            logger.debug(f"Unseen state mapped: '{from_state}' -> '{mapped_from}'")

        if to_state not in self.states and self.states:
            mapped_to = find_closest_pattern(to_state, list(self.states))
            logger.debug(f"Unseen state mapped: '{to_state}' -> '{mapped_to}'")
            
        # 2. Look up in mapped probability table
        if mapped_from in self.probabilities and mapped_to in self.probabilities[mapped_from]:
            prob = self.probabilities[mapped_from][mapped_to]
            # Apply penalty if mapping occurred to signal suspicion
            if mapped_from != from_state or mapped_to != to_state:
                return prob * 0.5 # Scaling down because it's substituted
            return prob
        
        # 3. Transition remains unseen even after remediation (Smoothing applied)
        return self.epsilon

    def calculate_sequence_probabilities(self, patterns: List[str], window_len: int = 5) -> List[float]:
        """
        Calculates sliding window path probabilities by multiplying sequential transitions.
        Returns a list of floats representing Path Probability of segments.
        """
        if len(patterns) < window_len:
            return []
            
        path_probs = []
        
        # Slide over pattern sequence to evaluate block health
        for i in range(len(patterns) - window_len + 1):
            segment = patterns[i : i + window_len]
            
            # Initialize path probability
            path_prob = 1.0
            
            # Multiply consecutive transitions
            for j in range(len(segment) - 1):
                curr = segment[j]
                nxt = segment[j + 1]
                trans_prob = self.get_transition_prob(curr, nxt)
                path_prob *= trans_prob
                
            path_probs.append(path_prob)
            
        return path_probs

    def predict_anomalies(
        self, test_patterns: List[str], window_len: int = 5, threshold: float = None
    ) -> Tuple[List[float], List[int], List[bool]]:
        """
        Evaluates test sequence and detects anomalies based on Path Probability.

        Returns:
            tuple: (
                path_probabilities_list,
                binary_labels_list  (1 = Anomaly, 0 = Normal),
                unseen_flags_list   (True if window contains ≥1 pattern not seen in training)
            )
        """
        active_threshold = threshold if threshold is not None else self.anomaly_threshold

        path_probs = self.calculate_sequence_probabilities(test_patterns, window_len)
        labels = [1 if p < active_threshold else 0 for p in path_probs]

        # Mark windows that contain at least one pattern outside the training state set.
        # These windows used Levenshtein remapping — relevant for Detection Rate / Mapping Accuracy.
        unseen_flags = [
            any(p not in self.states for p in test_patterns[i: i + window_len])
            for i in range(len(test_patterns) - window_len + 1)
        ]

        return path_probs, labels, unseen_flags

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Probabilistic Automata...")
    
    # Mock SAX pattern sequences
    # Let's assume the normal cycle is a->b->c->a->b...
    train_patterns = ['ab', 'bc', 'ca', 'ab', 'bc', 'ca', 'ab', 'bc', 'ca', 'ab']
    
    # Create model
    automata = ProbabilisticAutomata()
    
    # Train
    automata.fit(train_patterns)
    
    # Check a normal transition
    print(f"Prob('ab' -> 'bc'): {automata.get_transition_prob('ab', 'bc')}")
    
    # Test normal sequence
    test_normal = ['ab', 'bc', 'ca', 'ab']
    prob_norm, label_norm = automata.predict_anomalies(test_normal, window_len=3, threshold=1e-4)
    print(f"Normal Seq Probs: {prob_norm} | Labels: {label_norm}")
    
    # Test anomalous sequence (includes an 'aa' transition which never happens in train)
    test_anomaly = ['ab', 'aa', 'bc', 'ca']
    prob_anom, label_anom = automata.predict_anomalies(test_anomaly, window_len=3, threshold=1e-4)
    print(f"Anomalous Seq Probs: {prob_anom} | Labels: {label_anom}")
