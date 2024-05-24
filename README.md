# LDA_linear_probe

## Task

Fine-tuning a linear classification head based on an off-the-shelf feature extraction backbone model to adapt it to a new task.

## Existing methods

Attach a new linear head, which is basically a linear layer `x @ w + b`, and train it on dataset of the new task with cross entropy loss, while the backbone may be finetuned simultaneously in some cases or not in other cases.

## Limitations

- Slow. 
- Not optimal. 

## Proposed method

Keep the feature extractor backbone fixed, and solve the linear mapping by LDA. To be more specific, run through the training dataset of the new task, extract all the output features of the backbone model, and solve the projection matrix by multiclass LDA (Linear Discriminant Analysis)

code
```python
def PCA(X: np.ndarray, eps: float = 1e-2):
    '''
    PCA dimension reduction

    input:
        X: [N, D]
    
    output:
        X: [N, D1], D1 <= D
        X_mean: [D], can be used for test, `X_test_pca = (X_test - X_mean) @ U`
        U: [D, D1], can be used for test
    '''
    X_mean = np.mean(X, axis = 0)
    X -= X_mean
    A = X.T @ X / X.shape[0]
    e, U = np.linalg.eigh(A)
    mask = e > eps
    U0 = U[:, mask]
    X1 = X @ U0
    return X1, X_mean, U0

def LDA(X: np.ndarray, y: np.ndarray):
    '''
    input:
        X: [N, D]
        y: [N]
    output:
        entropy: float
    '''
    # PCA is required to avoid zero singular value in the covarian matrix
    X, X_mean, U0 = PCA(X)
    
    ys = set(y)
    K = len(ys)
    X_means = []
    Sws = []
    for y0 in ys:
        X1 = X[y == y0] # [N1, D]
        N0 = X1.shape[0]
        X1_mean = np.mean(X1, axis = 0) # [D]
        X_means.append(X1_mean)
        X2 = X1 - X1_mean # [N1, D]
        Sw0 = X2.T @ X2 / N0 # [D, D]
        Sws.append(Sw0)
    
    X_means = np.array(X_means)
    X_means -= np.mean(X_means, axis = 0)
    Sb = X_means.T @ X_means / K # [D, D]
    Sws = np.array(Sws) # [K, D, D]
    Sw = np.mean(Sws, axis = 0) # [D, D]
    # return np.trace(Sb) / np.trace(Sw)
    e, U = np.linalg.eigh(Sw)
    # assert np.sum(((U * e) @ U.T - Sw) ** 2) < 1e-5
    P = U / np.sqrt(e)
    Sb1 = P.T @ Sb @ P
    e1, U1 = np.linalg.eigh(Sb1)
    inds = np.argsort(e1)[::-1]
    e2 = e1[inds]
    # print(e2)
    U2 = U1[:, inds]
    return X_mean, U0 @ P @ U2

def test(X_train, y_train, X_test):
    '''
    input:
        X_train: [N1, D]
        y_train: [N1]
        X_test: [N2, D]

    output:
        y_test_pred: [N2]
    '''
    X_mean, U = LDA(X_train, y_train)
    X_train1 = (X_train - X_mean) @ U
    X_test1 = (X_test - X_mean) @ U
    X_cls = np.array([np.mean(X_train1[y_train == y], axis = 0) for y in sorted(list(set(y_train)))])

    D = np.sum(X_test1 ** 2, axis = 1, keepdims=True) + np.sum(X_cls ** 2, axis = 1, keepdims=True).T - 2 * X_test1 @ X_cls.T
    y_test_pred = np.argmin(D, axis = 1)
    return y_test_pred
```
