import numpy as np
from evtk import hl as vtkhl
import imageio

LATTICE_D = 2
LATTICE_Q = 9

# Définition du problème à simuler dans un fichier image
def open_image(filename):
    image = imageio.imread(filename)
    SIZE_X = image.shape[0]
    SIZE_Y = image.shape[1]
    walls = [(i,j) for i in range(SIZE_X) for j in range(SIZE_Y) 
     if sum(image[i,j,:]) < 20]
    walls = np.array(walls)
    return SIZE_X,SIZE_Y, walls


SIZE_X, SIZE_Y, walls = open_image("images/dessing.png")
# SIZE_X = 160
# SIZE_Y = 400
# nwalls = 40
# walls = np.ones([nwalls, LATTICE_D], dtype = np.int64);
# walls[:, 0] = np.arange(0, nwalls) + 5
# walls[:, 1] = 40

ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype = np.int64)
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype = np.int64)
lattice_w = np.array([4/9] + [1/9] * 4 + [1/36] * 4)


opposite_bb = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype = np.int64)

cs2 = sum(lattice_w * ex * ex)

nu = 0.02
tau = nu/cs2 + 1/2


cpt = iter(range(1000000)) #image counter

def IDXY(x, y, q):
    return ((x * SIZE_Y) + y) * LATTICE_Q + q


def save_to_vtk(N, name):
    rho, u, v = flow_properties(N)
    u   = np.reshape(u  , (SIZE_X, SIZE_Y, 1), order='C')
    v   = np.reshape(v  , (SIZE_X, SIZE_Y, 1), order='C')
    rho = np.reshape(rho, (SIZE_X, SIZE_Y, 1), order='C')
    vtkhl.imageToVTK(f"images/{name}_{next(cpt)}",
            pointData={"pressure": rho, "u": u, "v": v})

def init():
    N = np.ones([SIZE_X, SIZE_Y, LATTICE_Q])

    for q in range(LATTICE_Q):
        N[:, :, q] = lattice_w[q]

    return N

def flow_properties(N):
    rho = np.sum(N, axis = 2)
    u = np.sum(N * ex, axis = 2) / rho
    v = np.sum(N * ey, axis = 2) / rho
    return rho, u, v


def stream(N, P):
    return N.reshape(SIZE_X * SIZE_Y * LATTICE_Q)[P].reshape(SIZE_X, SIZE_Y, LATTICE_Q)


def calc_permutations():
    P = np.zeros([SIZE_X * SIZE_Y * LATTICE_Q], dtype = np.int64)
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            for q in range(LATTICE_Q):
                xp = (x + ex[q]) % SIZE_X
                yp = (y + ey[q]) % SIZE_Y
                P[IDXY(xp, yp, q)] = IDXY(x, y, q)
    return P


def equilibrium_distribution(rho, u, v):
    def p(t2, t1):
        return np.tensordot(t2, t1, axes = 0)
    qones = np.ones(LATTICE_Q) 
    vci = p(u, ex) + p(v, ey)
    row = p(rho, lattice_w)
    vsq = p(u*u + v*v, qones)
    Neq = row * (1 + (vci)/cs2 + (vci)**2 /(2*cs2**2) - (vsq)/(2*cs2))
    return Neq

def collide(N):
    rho, u, v = flow_properties(N)
    Neq = equilibrium_distribution(rho, u, v)
    return N - (N - Neq)/tau

def bounceback(N, walls):
    for x, y in walls:
        for q in range(LATTICE_Q):
            qbb = opposite_bb[q]
            xp, yp = (x - ex[q]) % SIZE_X, (y - ey[q]) % SIZE_Y
            N[xp, yp, qbb], N[x, y, q] = N[x, y, q], N[xp, yp, qbb]
    return N

def blow(N, domaine, alpha, v0):
    dir_droit = [6, 2, 5]
    dir_gauche = [7, 4, 8]
    for x, y in domaine:
        for i in range(3):
            N0 = N[x, y, dir_droit[i]]
            N1 = N[x, y, dir_gauche[i]]
            vloc = (N0 - N1) / (N0 + N1)
            dv = vloc - v0
            adv = alpha * dv
            N[x, y, dir_droit[i]] += adv
            N[x, y, dir_gauche[i]] -= adv
    return N

def impose_vel(N, domaine, v0):
    N_limit = equilibrium_distribution(1.0, v0, 0.0)
    for x, y in domaine:
        N[x, y, :] = N_limit
    return N

if __name__ == "__main__":
    rho = np.ones([SIZE_X, SIZE_Y])
    u = np.zeros([SIZE_X, SIZE_Y])
    v = np.zeros([SIZE_X, SIZE_Y])

    u[:, 5:10] = 0.1

    N = equilibrium_distribution(rho, u, v)
    # N[:, 0, :] = N[:, 0, :] * 1.1
    bord_gauche = [(x,0) for x in range(SIZE_X)]
    P = calc_permutations()
    save_to_vtk(N, "rond")
    domaine_limits = [(x, 0) for x in range(SIZE_X)]
    v0 = 0.05

    # print(N[14, 20, :])

    for t in range(2000):
        N = collide(N)
        N = stream(N, P)
        N = impose_vel(N, domaine_limits,  v0)
        N = bounceback(N, walls)
        if t%10 == 0:
            save_to_vtk(N, "rond")
