def apk(true, hat, k):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)

    predicted : list
                A list of predicted elements (order does matter)

    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists


          A little tutorial about enumerate(WOCA)
            for i in range(len(L)):
                item = L[i]
                # ... compute some result based on item ...
                L[i] = result

        This can be rewritten using enumerate() as:

            for i, item in enumerate(L):
                # ... compute some result based on item ...
                L[i] = result

    """

    if len(hat) > k:
        hat = hat[-k:]

    score = 0.000
    num_hits = 0.000

    """
    if len(hat) == 1:
        for p in hat:
            if p in true and p in hat:
                return 1
            else:
                return 0
    else:
        for i, p in enumerate(hat):
            if p in true and p in hat:
                num_hits += 1.000
                score += num_hits / (i + 1.000)
        if num_hits == 0.000:
            return 0
        else:
            return score / num_hits
    """
    if hat[-1] in true:
        for i, p in enumerate(hat):
            if p in true and p in hat:
                num_hits += 1.000
                score += num_hits / (i + 1.000)
        return score / num_hits
    else:
        for i, p in enumerate(hat):
            if p in true and p in hat:
                num_hits += 1.000
                score += num_hits / (i + 1.000)
        return score / (num_hits + 1)
