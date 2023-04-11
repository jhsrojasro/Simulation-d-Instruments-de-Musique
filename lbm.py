import numpy as np
from evtk import hl as vtkhl

LATTICE_D = 2
LATTICE_Q = 9

SIZE_X = 48
SIZE_Y = 64
# 6 2 5
# 3 0 1
# 7 4 8


ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
lattice_w = np.array([4/9] + [1/9] * 4 + [1/36] * 4)

cs2 = sum(lattice_w * ex * ex)

cpt = iter(range(1000000)) #image counter

def IDXY(x, y, q):
    return (x * SIZE_Y + y) * LATTICE_Q + q


def init():
    N = np.ones((SIZE_X, SIZE_Y, LATTICE_Q))

    for q in range(LATTICE_Q):
        N[:, :, q] = lattice_w[q]
    
    return N

def stream(N, P):
    return N.reshape((SIZE_X*SIZE_Y*LATTICE_Q))[P].reshape((SIZE_X, SIZE_Y, LATTICE_Q))
    

def permutation(N):
    N_1 = np.zeros_like(N)
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            for q in range(LATTICE_Q):
                xp = (x + ex[q]) % SIZE_X
                xy = (y + ey[q]) % SIZE_Y
                N_1[xp, xy, q] = N[x, y, q]
    return N_1

def flow_properties(N):
    rho = np.sum(N, axis=2)
    u = np.sum(N * ex, axis=2) / rho
    v = np.sum(N * ey, axis=2) / rho
    return rho, u, v

def save_to_vtk(N, name):
    rho, u, v = flow_properties(N)
    u   = np.reshape(u  , (SIZE_X, SIZE_Y, 1), order='C')
    v   = np.reshape(v  , (SIZE_X, SIZE_Y, 1), order='C')
    rho = np.reshape(rho, (SIZE_X, SIZE_Y, 1), order='C')
    vtkhl.imageToVTK(f"images/{name}_{next(cpt)}",
            pointData={"pressure": rho, "u": u, "v": v})

# def equilibrium_distribution(rho, u, v):
#     ...
#     return Neq


N = init()

N[:,0, :] = N[:, 0 , :] * 1.1

# P_X = (np.arange(0, SIZE_X) + 1) % SIZE_X

# P_X = np.array([SIZE_X-1] + list(range(0, (SIZE_X -1))))
P = np.arange(0, SIZE_X*SIZE_Y*LATTICE_Q)

# save_to_vtk(N, 'rond')

for t in  range(200):
    N = stream(N)
    save_to_vtk(N, 'rond')
