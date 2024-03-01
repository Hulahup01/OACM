import numpy as np


def main_simplex_method(c, x, A, B):
    m, n = A.shape
    x_new = x.copy()
    assert (np.linalg.matrix_rank(A) == m)
    while True:
        AB = A[:, B]
        AB_inv = np.linalg.inv(AB)
        cB = c[B]
        u = cB @ AB_inv
        delta = u @ A - c
        if np.all(delta >= 0):
            return [x_new, B]
        j0 = np.where(delta < 0)[0][0]
        z = AB_inv @ A[:, j0]
        theta = np.array([x_new[B[i]] / z[i] if z[i] > 0 else np.inf for i in range(m)])
        theta0 = np.min(theta)
        if theta0 == np.inf:
            raise Exception("The objective function is not top-bounded on a set of valid plans")
        k = np.argmin(theta)
        j_star = B[k]
        B[k] = j0
        for i in range(m):
            if i != k:
                x_new[B[i]] -= theta0 * z[i]
        x_new[j0] = theta0
        x_new[j_star] = 0


def initial_simplex_method(c, A, b):
    m, n = A.shape
    print(f"m: {m}     n: {n}\n")
    
    # Step 1
    for i in range(len(b)):
        if b[i] < 0:
            b[i] *= -1
            A[i] *= -1

    print(f"Matrix A:\n{A}\n")
    print(f"Vector c:\n{c}\n")
    print(f"Vector b:\n{b}\n")

    # Step 2
    c_tilde = np.concatenate((np.zeros(n), np.full((m,), -1)))
    print(f"Vector c_tilde:\n{c_tilde}\n")
    A_tilde = np.concatenate((A, np.eye(m)), axis=1)
    print(f"Matrix A_tilde:\n{A_tilde}\n")
    
    # Step 3
    x_tilde = np.concatenate((np.zeros(n), b))
    print(f"Vector x_tilde:\n{x_tilde}\n")
    B = np.arange(n, n + m)
    print(f"Vector B:\n{B}\n")

    # Step 4
    print(f"-===Getting the optimal plan===-\n")
    x_tilde, B = main_simplex_method(c_tilde, x_tilde, A_tilde, B)
    print(f"Vector x_tilde:\n{x_tilde}\n")
    print(f"Vector B:\n{B}\n")

    # Step 5
    if not np.all(x_tilde[n:n+m] == 0):
        print("The task is not compatible!")

    # Step 6 
    x = x_tilde[0:n]
    print(f"Vector x:\n{x}\n")

    iter = 1 
    while True:
        print(f"=======ITER{iter}=======\n")
        # Step 7
        if np.all((B >= 0) & (B < n - 1)):
            print("========END========\n")
            return x, B, A, b
        
        # Step 8
        j_k = B[-1]
        i = j_k - n 
        k = B.shape[0] - 1
        print(f"Index j_k:\n{j_k}\n")
        print(f"Index k:\n{k}\n")
        print(f"Index i:\n{i}\n")    

        # Step 9
        A_B_inv =  np.linalg.inv(A_tilde[:, B])
        l = np.array([A_B_inv @ A_tilde[:, j] for j in np.setdiff1d(np.arange(n), B)])
        print(f"Vectors l(j):\n{l}\n")  

        if np.any(np.array([l_i[k] for l_i in l]) != 0):
            # Step 10
            j = np.where([l_i[k] for l_i in l]) != 0
            B[k] = j
        else:
            # Step 11
            A = np.delete(A, i, axis=0)
            A_tilde = np.delete(A_tilde, i, axis=0)
            b = np.delete(b, i, axis=0)
            B = np.delete(B, k, axis=0)
        print("===================\n")
        iter += 1


if __name__ == '__main__':
    c = np.array([1, 0, 0])
    A = np.array([
        [1, 1, 1],
        [2, 2, 2]
    ])
    b = np.array([0, 0])
    
    x, B, A_t, b_t = initial_simplex_method(c.copy(), A.copy(), b.copy())
    print(f"Matrix A:\n{A_t}\n")
    print(f"vector b_t:\n{b_t}\n")
    print(f"Vector x:\n{x}\n")
    print(f"Vector B:\n{B}\n")

    