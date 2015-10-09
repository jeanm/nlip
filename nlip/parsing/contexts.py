from itertools import zip_longest
def get_window(index, length, k=2):
    """Return a list of tuples (pos, dist_from_centre) for index's window"""
    start_win = max(0, index - k)
    end_win = min(length - 1, index + k)
    return sorted(((pos,abs(index-pos)) for pos in range(start_win, end_win+1)
            if pos != index), key=lambda x: x[1])

def get_phrase_window(indices, length, k=2):
    window = [k+1] * length  # maps position in sentence to smallest distance from index word
    indices = set(indices)  # used to make sure I don't include indices in the window
    for index in indices:
        for pos, distance in get_window(index, length, k):
            if distance < window[pos]:  # only update distance if lower than what we already found
                window[pos] = distance
    # positions are sorted by smallest distance from any index word, and
    # do not include index words themselves
    return sorted((tuple(element) for element in enumerate(window)
            if element[1] <= k and element[0] not in indices),
            key=lambda x: x[1])

def get_sentence_window(indices, length):
    return [i for i in range(length) if i not in indices]
