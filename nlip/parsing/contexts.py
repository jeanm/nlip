from itertools import zip_longest

def _sort_tuples(indices, window, k):
    return sorted((tuple(element) for element in enumerate(window)
            if element[1] <= k and element[0] not in indices),
            key=lambda x: x[1])

def get_window(index, length, k=2):
    """Return a list of tuples (pos, dist_from_centre) for index's window"""
    start_win = max(0, index - k)
    end_win = min(length - 1, index + k)
    return sorted(((pos,abs(index-pos)) for pos in range(start_win, end_win+1)
            if pos != index), key=lambda x: x[1])

def get_phrase_window(indices, length, k=None):
    if k is None:
        k = length
    distances = (min([abs(l-i) for i in indices]) for l in range(length))
    return _sort_tuples(indices, distances, k)
