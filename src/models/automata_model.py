import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

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

    def fit(self, patterns: List[str]):
        """
        Learns transition matrix from a sequence of states (patterns).
        Consecutive elements in the list represent a sequence step.
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
                
        print(f"Automata fitted successfully. Total States: {len(self.states)}")
        
        # Pre-calculate self probability baseline to assist threshold setting automatically
        # Using mean path probabilities of training sequence segments
        train_probs = self.calculate_sequence_probabilities(patterns, window_len=5)
        if train_probs:
            # Set threshold to e.g., 5th percentile of training path probabilities (Low probability = anomaly)
            self.anomaly_threshold = np.percentile(train_probs, 5)
            print(f"Dynamic Anomaly Threshold established at 5th percentile: {self.anomaly_threshold:.4e}")

    def get_transition_prob(self, from_state: str, to_state: str) -> float:
        """Returns transition probability. Applies smoothing if states are known but pair is unseen."""
        # 1. Direct lookup
        if from_state in self.probabilities and to_state in self.probabilities[from_state]:
            return self.probabilities[from_state][to_state]
        
        # 2. State exists but transition is novel (Smoothing applied)
        if from_state in self.states:
            return self.epsilon
            
        # 3. State itself is unseen (Highest severity anomaly marker)
        return self.epsilon * 0.1

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

    def predict_anomalies(self, test_patterns: List[str], window_len: int = 5, threshold: float = None) -> Tuple[List[float], List[int]]:
        """
        Evaluates test sequence and detects anomalies based on Path Probability.
        
        Returns:
            tuple: (path_probabilities_list, binary_labels_list where 1 = Anomaly, 0 = Normal)
        """
        # Use automatic threshold if none provided
        active_threshold = threshold if threshold is not None else self.anomaly_threshold
        
        path_probs = self.calculate_sequence_probabilities(test_patterns, window_len)
        
        # Lower probability than threshold indicates an Anomaly
        # Mark as 1 (Anomaly) if probability is below threshold, else 0 (Normal)
        labels = [1 if p < active_threshold else 0 for p in path_probs]
        
        return path_probs, labels

if __name__ == "__main__":
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
