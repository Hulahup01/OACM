import numpy as np

if __name__ == "__main__":
    n = int(input("Enter size of square matrix: "))
    ind = int(input("Enter i: "))
    x = np.array(list(map(int, (input("Enter vector x: \n").split()))))
    A = np.empty((n, n))
    print("Enter matrix A:")
    for i in range(n):
        row = np.array(list(map(int, input().split())))
        A[i] = row

    A_inv = np.linalg.inv(A)

    A_swap = A.copy()
    A_swap[:, ind - 1] = x

    print("==========================")
    print(f"A: \n {A} \n")
    print(f"A_inv: \n {A_inv} \n")
    print(f"x: \n {x} \n")
    print(f"A_swap: \n {A_swap} \n")
    print("==========================")

    l_vec = A_inv @ x
    l_ind = l_vec[ind - 1]

    if l_ind == 0:
        raise Exception("Matrix is not invertible ")
    else:
        print("\nMatrix is invertible\n")

    l_vec_swap = l_vec.copy()
    l_vec_swap[ind - 1] = -1

    l_hat = (-1 / l_ind) * l_vec_swap

    Q = np.eye(n)
    Q[:, ind - 1] = l_hat

    A_swap_inv = Q @ A_inv

    print(f"A_swap_inv: \n {A_swap_inv} \n")в






