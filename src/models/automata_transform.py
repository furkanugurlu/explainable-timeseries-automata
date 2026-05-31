import logging
import numpy as np
import string
from typing import List, Dict, Any, Tuple
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

class SAXTransformer:
    """
    Implements Piecewise Aggregate Approximation (PAA) and Symbolic Aggregate approXimation (SAX)
    tailored for building a Probabilistic Automata.
    """
    def __init__(self, window_size: int = None, alphabet_size: int = None, config: dict = None):
        cfg = config if config else load_config()
        
        # Use provided arguments or fall back to config defaults
        self.window_size = window_size if window_size else cfg['automata']['defaults']['window_size']
        self.alphabet_size = alphabet_size if alphabet_size else cfg['automata']['defaults']['alphabet_size']
        
        # Check validity
        if self.alphabet_size > 26:
            raise ValueError("Alphabet size cannot exceed 26 (number of English letters).")
            
        # Alphabet setup ('a', 'b', 'c', ...)
        self.alphabet = list(string.ascii_lowercase[:self.alphabet_size])
        
        # Placeholders for the distribution breakpoints derived ONLY from Train data
        self.breakpoints = None

    def to_paa(self, data: np.ndarray) -> np.ndarray:
        """
        Converts raw time series into PAA (Piecewise Aggregate Approximation).
        Divides data into non-overlapping blocks of size 'window_size' and averages them.
        """
        # Flatten just in case
        flat_data = data.flatten()
        
        # Calculate number of blocks
        num_blocks = len(flat_data) // self.window_size
        
        if num_blocks == 0:
            # Edge case: series shorter than window
            return np.array([flat_data.mean()])
            
        # Discard trailing elements that don't fill a full block or pad. 
        # Standard practice is often to just discard the tiny fraction at the end.
        trimmed_len = num_blocks * self.window_size
        reshaped_data = flat_data[:trimmed_len].reshape(num_blocks, self.window_size)
        
        # Compute block means
        paa_data = np.mean(reshaped_data, axis=1)
        return paa_data

    def fit(self, train_data: np.ndarray):
        """
        Finds the breakpoints for the SAX transformation using ONLY the training data.
        Uses empirical quantiles from the training PAA values to satisfy the leakage rule.
        """
        # Convert to PAA first
        paa_train = self.to_paa(train_data)
        
        # Calculate quantiles corresponding to uniform distribution bins across alphabet_size
        # For 3 letters ('a','b','c'), we need 2 cut-points at 33.3% and 66.6%
        percentiles = np.linspace(0, 100, self.alphabet_size + 1)[1:-1]
        
        # Find actual threshold values in the data
        self.breakpoints = np.percentile(paa_train, percentiles)
        
        logger.info(f"Transformer fitted. Alphabet: {self.alphabet} | Breakpoints: {self.breakpoints}")
        return self

    def to_sax(self, paa_data: np.ndarray) -> List[str]:
        """
        Converts PAA float sequence to character sequence using defined breakpoints.
        """
        if self.breakpoints is None:
            raise RuntimeError("Transformer not fitted. Call fit(train_data) before transforming.")
            
        # Use np.digitize to map float values to indices
        # Digitized bins map from 0 to alphabet_size - 1
        symbol_indices = np.digitize(paa_data, self.breakpoints)
        
        # Map indices to alphabet characters
        sax_symbols = [self.alphabet[idx] for idx in symbol_indices]
        
        return sax_symbols

    def fit_transform(self, train_data: np.ndarray) -> List[str]:
        """Fits model on training data and converts to SAX symbols immediately."""
        self.fit(train_data)
        paa = self.to_paa(train_data)
        return self.to_sax(paa)

    def transform(self, data: np.ndarray) -> List[str]:
        """Transforms given data into SAX sequence using existing breakpoints."""
        paa = self.to_paa(data)
        return self.to_sax(paa)

    def extract_patterns(self, sax_symbols: List[str], pattern_length: int = 3) -> List[str]:
        """
        Generates patterns using sliding window method over the symbol sequence.
        Example: ['a', 'b', 'c', 'a'] with length 2 -> ['ab', 'bc', 'ca']
        """
        patterns = []
        # Basic sliding window over iterable
        for i in range(len(sax_symbols) - pattern_length + 1):
            window = sax_symbols[i : i + pattern_length]
            # Join symbols to create a cohesive string pattern
            patterns.append("".join(window))
            
        return patterns

if __name__ == "__main__":
    print("Testing SAX Transformer...")
    # Create dummy sequential time series
    dummy_train = np.array([0.1, 0.2, 0.3, 0.9, 1.1, 1.0, 0.4, 0.5, 0.6])
    dummy_test = np.array([0.8, 1.2, 0.3])
    
    # Instantiate (uses config implicitly, but we can override)
    sax = SAXTransformer(window_size=2, alphabet_size=3)
    
    # 1. Fit & Transform Train
    train_symbols = sax.fit_transform(dummy_train)
    print(f"Train Data:\n{dummy_train}")
    print(f"PAA Result (means of size 2):\n{sax.to_paa(dummy_train)}")
    print(f"Train SAX Sequence: {train_symbols}")
    
    # 2. Transform Test
    test_symbols = sax.transform(dummy_test)
    print(f"\nTest Data:\n{dummy_test}")
    print(f"Test SAX Sequence: {test_symbols}")
    
    # 3. Pattern Extraction (Sliding Window over Symbols)
    patterns = sax.extract_patterns(train_symbols, pattern_length=2)
    print(f"\nExtracted Patterns (Slide len 2): {patterns}")
