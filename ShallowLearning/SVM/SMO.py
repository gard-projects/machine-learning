import numpy as np

# Sequential Minimal Optimization (SMO) algorithm for training SVM

def polynomial_kernel(X, Y=None, r=0, d=3, gamma=None):
    '''
    Parameters
    ----------
    X : np.ndarray
        A matrix with n samples and m features, shape (n, m)
    Y : np.ndarray
        A matrix with n samples and m features, shape (n, m)
    r : float
        Coefficient of the polynomial kernel
    d : int
        Degree of the polynomial kernel
    gamma : float
        Coefficient of the polynomial kernel, used for scaling
        
    Returns
    -------
    The polynomial kernel matrix for the given data, shape (n, n)
    '''
    if Y is None:
        Y = X
        
    K = np.dot(X, Y.T)
    if gamma is None:
        gamma = 1 / (X.shape[1]*np.var(X))
    
    return (gamma * K + r)**d
    
def linear_kernel(X, Y=None):
    '''
    Parameters
    ----------
    X : np.ndarray
        A matrix with n samples and m features, shape (n, m)
    Y : np.ndarray
        A matrix with n samples and m features, shape (n, m)
        
    Returns
    -------
    The linear kernel matrix for the given data, shape (n, n)
    '''
    if Y is None:
        Y = X
    return np.dot(X,Y.T)

def rbf_kernel(X, Y=None, gamma=0.5):
    if gamma is None:
        gamma = 1 / (X.shape[1]*np.var(X))

    if Y is None:
        Y = X
    return np.exp(-gamma * np.linalg.norm(X[:, np.newaxis] - Y[np.newaxis, :], axis=2)**2)

def compute_error(alphas: np.ndarray, y: np.ndarray, b: float, i: int, kernel_cache: np.ndarray=None):
    '''
    Parameters
    ----------
    alphas : np.ndarray
        The alpha values for the SVM model
    y : np.ndarray
        The labels for the data
    b : float
        The bias term for the SVM model
    i : int
        The index of the data point
    kernel_cache : np.ndarray
        The kernel matrix for the data points
        
    Returns
    -------
    The error for the given data point
    '''
    
    return (np.dot((alphas * y), kernel_cache[:,i]) + b) - y[i]

def update_multipliers(i, j, alphas, errors, C, b, y, kernel_cache: np.ndarray):
    '''
    Parameters
    ----------
    i : int
        The index of the first data point
    j : int
        The index of the second data point
    alphas : np.ndarray
        The alpha values for the SVM model
    errors : np.ndarray
        The error values for the data points
    C : float
        The regularization parameter
    b : float
        The bias term for the SVM model
    y : np.ndarray
        The labels for the data
    kernel_cache : np.ndarray
        The kernel matrix for the data points

    Returns
    -------
    A tuple containing a boolean value indicating if the alphas were updated, and the new bias term
    '''
    
    
    # Skip the iteration if the alphas are the same
    if i == j:
        return False, b

    alpha_i_old, alpha_j_old = alphas[i], alphas[j]
    y_i, y_j = y[i], y[j]
    E_i, E_j = errors[i], errors[j]
    
    # Compute the bounds for a_j (L, H)
    if y_i != y_j:
        L = max(0, alpha_j_old - alpha_i_old)
        H = min(C, C + alpha_j_old - alpha_i_old)
    else:
        L = max(0, alpha_j_old + alpha_i_old - C)
        H = min(C, alpha_j_old + alpha_i_old)
        
    # No need to optimize if L == H
    if L == H:
        return False, b

    # Find eta parameter
    eta = 2 * kernel_cache[i, j] - kernel_cache[i, i] - kernel_cache[j, j]
    if eta >= 0:
        return False, b
    
    # Update a_j
    alphas[j] -= y_j * (E_i - E_j) / eta
    alphas[j] = max(L, min(H, alphas[j]))
    # Check if update is significant
    if abs (alphas[j] - alpha_j_old) < 1e-5:
        return False, b
    
    # Update a_i
    alphas[i] += y_i * y_j * (alpha_j_old - alphas[j])
    # Update b (bias)
    b1 = b - E_i - y_i * (alphas[i] - alpha_i_old) * kernel_cache[i, i] - y_j * (alphas[j] - alpha_j_old) * kernel_cache[i, j]
    b2 = b - E_j - y_i * (alphas[i] - alpha_i_old) * kernel_cache[i, j] - y_j * (alphas[j] - alpha_j_old) * kernel_cache[j, j]
    
    # Determine which b to use
    if 0 < alphas[i] < C:
        b = b1
    elif 0 < alphas[j] < C:
        b = b2
    else:
        b = (b1 + b2) / 2
        
    return True, b
    
def sequential_minimal_optimization(X: np.ndarray, y: np.ndarray, max_iters: int=100, tol: float=1e-3, C: float=0.5):
    '''
    Parameters
    ----------
    X : np.ndarray
        A matrix with n samples and m features, shape (n, m)
    y : np.ndarray
        A vector with n labels, shape (n, 1)
    max_iters : int
        The maximum number of iterations to run the algorithm
    tol : float
        The tolerance for the algorithm, what to consider as a significant solution to the dual problem
    C : float
        The regularization parameter
        
    Returns
    -------
    alphas : np.ndarray
        The alpha values for the SVM model
    b : float
        The bias term for the SVM model
    '''
    
    n = X.shape[0]
    alphas = np.zeros(n)
    b = 0
    
    kernel_cache = rbf_kernel(X)
    errors = np.array([compute_error(alphas, y, b, i, kernel_cache) for i in range(n)])
    
    # Choose the first alpha, then determine the second alpha
    for _ in range(max_iters):
        alpha_pairs_changed = 0
        
        for i in range(n):
            E_i = errors[i]
            if (y[i] * E_i < -tol and alphas[i] < C) or (y[i] * E_i > tol and alphas[i] > 0):
                # Fetch the second alpha
                j = np.random.randint(low=0, high=n-1)
                updated, new_b = update_multipliers(i, j, alphas, errors, C, b, y, kernel_cache)
                
                if updated:   
                   alpha_pairs_changed += 1
                   b = new_b
                   errors = np.array([compute_error(alphas, y, b, k, kernel_cache) for k in range(n)])
                   
            # End algorithm when a suitable solution is found
            if alpha_pairs_changed == 0:
                break

    return alphas, b