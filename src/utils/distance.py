import numpy as np

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculates the Levenshtein (edit) distance between two strings.
    Represents the number of single-character edits (insertions, deletions or substitutions)
    required to change s1 into s2.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def find_closest_pattern(target: str, dictionary: list) -> str:
    """
    Finds the pattern in the dictionary with the minimum Levenshtein distance to the target.
    """
    if not dictionary:
        raise ValueError("Dictionary is empty.")
        
    min_dist = float('inf')
    closest = dictionary[0]
    
    for candidate in dictionary:
        dist = levenshtein_distance(target, candidate)
        if dist < min_dist:
            min_dist = dist
            closest = candidate
            
        # Early break if exact match found
        if min_dist == 0:
            break
            
    return closest
