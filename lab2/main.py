import numpy as np


def simplex_method(c, x, A, B):
    m, n = A.shape
    x_new = x.copy()

    assert (np.linalg.matrix_rank(A) == m)
    iter = 1
    while True:
        print(f"=======ITER{iter}=======\n")

        # Step 1
        AB = A[:, B]
        print(f"Matrix AB:\n{AB} \n")
        AB_inv = np.linalg.inv(AB)
        print(f"Matrix AB_inv:\n{AB_inv}\n")

        # Step 2
        cB = c[B]
        print(f"Vector cB:\n{cB}\n")

        # Step 3
        u = cB @ AB_inv
        print(f"Vector u:\n{u}\n")

        # Step 4
        delta = u @ A - c
        print(f"delta:\n{delta}\n")

        # Step 5
        if np.all(delta >= 0):
            print("========END========\n")
            return x_new

        # Step 6
        j0 = np.where(delta < 0)[0][0]
        print(f"Index j0:\n{j0}\n")

        # Step 7
        z = AB_inv @ A[:, j0]
        print(f"Vector z:\n{z}\n")

        # Step 8
        theta = np.array([x_new[B[i]] / z[i] if z[i] > 0 else np.inf for i in range(m)])
        print(f"Vector theta:\n{theta}\n")

        # Step 9
        theta0 = np.min(theta)
        print(f"Component theta0:\n{theta0}\n")

        # Step 10
        if theta0 == np.inf:
            raise Exception("The objective function is not top-bounded on a set of valid plans")

        # Step 11
        k = np.argmin(theta)
        print(f"Index k:\n{k}\n")
        j_star = B[k]
        print(f"Index j*:\n{j_star}\n")

        # Step 12
        B[k] = j0
        print(f"Vector B:\n{B}\n")

        # Step 12
        for i in range(m):
            if i != k:
                x_new[B[i]] -= theta0 * z[i]

        x_new[j0] = theta0
        x_new[j_star] = 0
        print(f"Vector new x:\n{x_new}\n")
        B.sort()
        print(f"Vector B:\n{B}\n")
        print("===================\n")
        iter += 1


if __name__ == '__main__':
    c = np.array([1, 1, 0, 0, 0])
    x = np.array([0, 0, 1, 3, 2])
    A = np.array([
        [-1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1]
    ])
    B = np.array([2, 3, 4])
    
    print(f"Matrix A:\n{A}\n")
    print(f"Vector c:\n{c}\n")
    print(f"Vector x:\n{x}\n")
    print(f"Vector B:\n{B}\n")

    result = simplex_method(c, x, A, B)

    print("The optimal plan", result)