import numpy as np
from evtk import hl as vtkhl

LATTICE_D = 2
LATTICE_Q = 9

SIZE_X = 48
SIZE_Y = 64

# 6 2 5
# 3 0 1
# 7 4 8

ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)

opposite_bb = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)

lattice_w = np.array([4/9] + [1/9] * 4 + [1/36] * 4)
cs2 = sum(lattice_w * ex * ex)
nu = 0.1
tau = nu/cs2 + 1/2
cpt = iter(range(1000000)) #image counter

nwalls = 40
walls = np.ones([nwalls, LATTICE_D])
walls[:, 0] = np.arange(0, nwalls) + 5
walls[:, 1] =  40

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
    P = precalculate_permutations()
    return N

def flow_properties(N):
    rho = np.sum(N, axis = 2)
    u = np.sum(N * ex, axis = 2) / rho
    v = np.sum(N * ey, axis = 2) / rho
    return rho, u, v


def stream(N, P):
    return N.reshape(SIZE_X * SIZE_Y * LATTICE_Q)[P].reshape(SIZE_X, SIZE_Y, LATTICE_Q)

def precalculate_permutations():
    P = np.zeros(SIZE_X * SIZE_Y * LATTICE_Q, dtype=np.int32)
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            for q in range(LATTICE_Q):
                xp = (x + ex[q]) % SIZE_X
                yp = (y + ey[q]) % SIZE_Y
                P[IDXY(xp, yp, q)] = IDXY(x, y, q)
    return P   

def stream_avec_boucles(N):
    R = np.zeros_like(N)
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            for q in range(LATTICE_Q):
                xp = (x + ex[q]) % SIZE_X
                yp = (y + ey[q]) % SIZE_Y
                R[xp, yp, q] = N[x, y, q]
    return R

### Calculer la distribution Neq à l’équilibre à partir des tableaux rho, u et v (chacun de taille SIZE_X * SIZE_Y)
def equilibrium_distribution(rho, u, v):
    def p(t2,t1):
        return np.tensordot(t2,t1,axes=0)
    qones = np.ones(LATTICE_Q)
    vci = p(u,ex) + p(v,ey)
    row = p(rho, lattice_w)
    vsq = p(u**2 + v**2, qones)
    Neq = row * (1 + vci/cs2 + (vci)**2/(2*cs2**2) - (vsq)/(2*cs2))
    
    return Neq

def collide(N):
    rho, u, v = flow_properties(N)
    Neq = equilibrium_distribution(rho, u, v)
    return N - (N - Neq) / tau

def bounce_back(N, walls):
    for i in range(nwalls):
        x = int(walls[i, 0])
        y = int(walls[i, 1])
        for q in range(LATTICE_Q):
            qbb = opposite_bb[q]
            xp, yp = ((x + ex[q]) % SIZE_X, (y + ey[q]) % SIZE_Y)
            N[xp, yp, qbb], N[x,y,q] = N[x, y, q], N[xp, yp, qbb] 
    return N

### Calculer le tableau des permutaions "P" une fois pour toutes
### ATTENTION, l'exemple ci-après ne fais pas de streaming (elle laisse les données là où elles sont), regarder la fonction "stream" pour s’en inspirer.
P = precalculate_permutations()

rho = np.ones((SIZE_X, SIZE_Y))
u = np.zeros((SIZE_X, SIZE_Y))
v = np.zeros((SIZE_X, SIZE_Y))
N = equilibrium_distribution(rho, u, v)

N[10,20, :] = N[10,20, :] * 1.1

save_to_vtk(N, "rond")

# print(N[14,20, :])

for t in range(200):
    N = collide(N)
    N = stream(N, P)
    N = bounce_back(N, walls)
    save_to_vtk(N, "rond")
 