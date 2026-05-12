import unittest
import sys
import os
from pathlib import Path

# Ensure project root is in path for module loading
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from src.utils.distance import levenshtein_distance, find_closest_pattern

class TestLevenshteinLogic(unittest.TestCase):
    
    def test_levenshtein_identical(self):
        """Identical strings should yield a distance of 0."""
        self.assertEqual(levenshtein_distance("abc", "abc"), 0)
        
    def test_levenshtein_substitution(self):
        """Substituting one character yields a distance of 1."""
        self.assertEqual(levenshtein_distance("abc", "adc"), 1)
        
    def test_levenshtein_deletion(self):
        """Deleting one character yields a distance of 1."""
        self.assertEqual(levenshtein_distance("abc", "ab"), 1)
        
    def test_levenshtein_insertion(self):
        """Inserting one character yields a distance of 1."""
        self.assertEqual(levenshtein_distance("abc", "abcd"), 1)
        
    def test_levenshtein_complex(self):
        """Complex edits should match standard Levenshtein values."""
        # kitten -> sitting has 3 edits
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)

    def test_closest_pattern_exact(self):
        """If target exists in dict, exact match should be returned."""
        dictionary = ["abc", "xyz", "efg"]
        result = find_closest_pattern("xyz", dictionary)
        self.assertEqual(result, "xyz")
        
    def test_closest_pattern_unseen(self):
        """If target doesn't exist, the mathematically closest one should be picked."""
        dictionary = ["aaaa", "bbbb", "cccc"]
        
        # 'aaab' is 1 edit away from 'aaaa' and 3 edits away from 'bbbb'
        result = find_closest_pattern("aaab", dictionary)
        self.assertEqual(result, "aaaa")
        
    def test_closest_pattern_tie_breaking(self):
        """Handling when there is a pool of options."""
        dictionary = ["ab", "ba", "ca"]
        
        # 'cb' is 1 edit from 'ab', 1 edit from 'ca'. 
        # Should return the first one that minimizes distance.
        result = find_closest_pattern("cb", dictionary)
        self.assertIn(result, ["ab", "ca"])

if __name__ == '__main__':
    unittest.main()
