import numpy as np


def simplex_dual(c, A, b, B):
    m, n = A.shape

    print(f"Vector c:\n{c}\n")
    print(f"Matrix A:\n{A}\n")
    print(f"Vector b:\n{b}\n")

    iter = 1
    while True:
        print(f"=======ITER{iter}=======\n")
        print(f"Vector B:\n{B}\n")
        # Step 1
        AB = A[:, B]
        print(f"Matrix AB:\n{AB}\n")
        AB_inv: np.ndarray = np.linalg.inv(AB)
        print(f"Matrix AB_inv:\n{AB_inv}\n")

        # Step 2
        cB = c[B]
        print(f"Vector cB:\n{cB}\n")

        # Step 3
        y = cB @ AB_inv
        print(f"Vector y:\n{y}\n")

        # Step 4
        kB = AB_inv @ b
        print(f"Vector kB:\n{kB}\n")
        kk = np.array([kB[np.where(B == i)][0] if i in B else 0 for i in range(n)])
        print(f"Vector k:\n{kk}\n")

        # Step 5
        if np.all(kk >= 0):
            print("========END========\n")
            return kk, B

        # Step 6
        j_k = np.where(kk < 0)[0][-1]
        print(f"Index j_k:\n{j_k}\n")
        k = np.where(B == j_k)[0][0]
        print(f"Index k:\n{k}\n")

        # Step 7
        delta_y = AB_inv[k]
        print(f"Vector delta_y:\n{delta_y}\n")
        mu = np.array([delta_y @ A[:, j] if j in np.setdiff1d(np.arange(n), B) else 0 for j in range(n)])
        print(f"Vector µ:\n{mu}\n")

        # Step 8
        if np.all(mu[np.setdiff1d(np.arange(n), B)] >= 0):
            raise ValueError("The task is not compatible!")

        # Step 9
        sigma = np.array([(c[j] - A[:, j] @ y) / mu[j] for j in np.setdiff1d(np.arange(n), B) if mu[j] < 0])
        print(f"Vector sigma:\n{sigma}\n")
        sigma_0 = min(sigma)
        print(f"sigma0:\n{sigma_0}\n")

        # Step 10
        j_0 = np.where(sigma == sigma_0)[0][0]
        print(f"Index j_0:\n{j_0}\n")
        B[k] = j_0
        B.sort()
        iter+=1


if __name__ == "__main__":
    c = np.array([-4, -3, -7, 0, 0])
    A = np.array([
        [-2, -1, -4, 1, 0],
        [-2, -2, -2, 0, 1],
    ])
    b = np.array([-1, -3. / 2])
    B = np.array([3, 4])

    x, B = simplex_dual(c, A, b, B)

    print(f"The optimal plan:\n x: {x}\n B: {B}")