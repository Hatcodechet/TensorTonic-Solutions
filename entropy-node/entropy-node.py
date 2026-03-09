import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.array(y)

    #count occurences of each classes
    values, counts = np.unique(y, return_counts=True)

    #compute probabilities
    probs = counts/len(y)

    #remove zero prob to avoid log(0)
    probs = probs[probs > 0]
    #compute entropy
    entropy = -np.sum(probs * np.log2(probs))
    return entropy