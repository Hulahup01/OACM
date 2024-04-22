import numpy as np


def get_basis_vector(c, B):
    i = 0
    c_b = [0 for _ in B]
    for index in B:
        c_b[i] = c[index]
        i += 1
    return c_b


def check(delta_x):
    for i in range(len(delta_x)):
        if delta_x[i] < 0:
            return i
    return -1


def main():
    c = np.array([-8, -6, -4, -6])
    A = np.array([[1, 0, 2, 1], [0, 1, -1, 2]])
    x = np.array([2, 3, 0, 0])
    Jb = np.array([0, 1])
    Jbs = np.array([0, 1])
    D = np.array([[2, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]])
    iter = 0
    print(f'Matrix A:\n{A}\n')
    print(f'Matrix D:\n{D}\n')
    print(f'Vector Jb:\n{Jb}\n')
    print(f'Vector c:\n{c}\n')
    print(f'Vector x:\n{x}\n')
    while True:
        iter += 1
        print(f"\n===========ITER{iter}============\n")
        A_b = A[:, Jb]
        A_b_inv = np.linalg.inv(A_b)
   
        c_x = c + np.dot(x, D)
        c_b = get_basis_vector(c_x, Jb)
        c_b = [i * (-1) for i in c_b]
        u_x = np.dot(c_b, A_b_inv)
        delta_x = np.dot(u_x, A) + c_x
        print(f'Vector c(x):\n{c_x}\n')
        print(f'Vector u(x):\n{u_x}\n')
        print(f'Vector ∆(x):\n{delta_x}\n')
        
        j0 = check(delta_x)
        if j0 == -1:
            print('[!] Plan is optimal\n')
            print("============END============\n")
            print(f"Optimal plan: \n{x}")
            return x
     
        print(f'Index j0: \n{j0}\n')
        l = np.zeros(len(x))
        l[j0] = 1
        A_b_ext = A[:, Jbs]

        H = np.bmat(
            [[D[Jbs, :][:, Jbs], A_b_ext.T], [A_b_ext, np.zeros((len(A), len(A)))]]
        )
        print(f'Matrix H:\n{H}\n')
        H_inv = np.array(np.linalg.inv(H))
        print(f'Matrix H_inv:\n{H_inv}\n')

        b_starred = np.concatenate((D[Jbs, j0], A[:, j0]))
        print(f'Vector b*:\n{b_starred}\n')
        x_temp = np.dot(-H_inv, b_starred)
        print(f'Vector x:\n{x_temp}\n')

        l[: len(Jbs)] = x_temp[: len(Jbs)]
        print(f'Vector l:\n{l}\n')

        delta = np.dot(np.dot(l, D), l)
        print(f'Delta δ:\n{delta}\n')
        theta = {}
        theta[j0] = np.inf if delta == 0 else np.abs(delta_x[j0]) / delta
        print(f'Theta_j0 θ_{j0}:\n{theta[j0]}\n')

        for j in Jbs:
            if l[j] < 0:
                theta[j] = -x[j] / l[j]
            else:
                theta[j] = np.inf
        theta = dict(sorted(theta.items()))
        print(f'Theta θ:\n{theta}\n')
        j_s = min(theta, key=theta.get)
        theta_0 = theta[j_s]
        print(f'Min theta_{j_s} θ_{j_s}:\n{theta_0}\n')

        if theta_0 == np.inf:
            print("Целевая функция задачи не ограничена снизу на множестве допустимых планом")

        x = x + theta_0 * l
        print("Updated plan x:", x)
        if j_s == j0:
            Jbs = np.append(Jbs, j_s)
        elif j_s in Jbs and j_s not in Jb:
            Jbs = np.delete(Jbs, j_s)
        elif j_s in Jb:
            third_condition = False
            s = Jb.index(j_s)

            for j_plus in set(Jbs).difference(Jb):
                if (np.dot(A_b_inv, A[:, j_plus]))[s] != 0:
                    third_condition = True
                    Jb[s] = j_plus
                    Jbs = np.delete(Jbs, j_s)

            if not third_condition:
                Jb[s] = j0
                Jbs[Jbs.index(j_s)] = j0
            print("Updated constraint supports: ", Jbs)


if __name__ == "__main__":
    main()