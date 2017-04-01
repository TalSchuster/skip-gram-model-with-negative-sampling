import numpy as np

def knn(vector, matrix, k=10):
    """
    Finds the k-nearest rows in the matrix with comparison to the vector.
    Use the cosine similarity as a distance metric.

    Arguments:
    vector -- A D dimensional vector
    matrix -- V x D dimensional numpy matrix.

    Return:
    nearest_idx -- A numpy vector consists of the rows indices of the k-nearest neighbors in the matrix
    """

    nearest_idx = []
    
    matNorm = np.linalg.norm(matrix, axis=1)
    vecNorm = np.linalg.norm(vector)
    dist = np.dot(matrix, vector) / (matNorm*vecNorm)
    
    # Reverse order for cosine similarity
    order = dist.argsort()[::-1] 
    nearest_idx = order[:k]    
    
    return nearest_idx

def test_knn():
    """
    Use this space to test your knn implementation by running:
        python knn.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    mat = np.array([[2, 3.5], [0, 1], [2, 3] ])
    vec = np.array([2,3])
    ans = knn(vec, mat, 2)
    assert 0 in ans
    assert 1 not in ans
    assert 2 in ans
    
    mat = np.array([[2, 3.5], [0, 1], [2, 3] ])
    vec = np.array([0,0.95])
    ans = knn(vec, mat, 1)
    assert 0 not in ans
    assert 1 in ans
    assert 2 not in ans

if __name__ == "__main__":
    test_knn()


