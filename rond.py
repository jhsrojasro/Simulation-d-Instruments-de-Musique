import numpy as np

SIZE_X = 100
SIZE_Y = 30
LATTICE_Q = 9
# 6 2 5
# 3 0 1
# 7 4 8

N = np.ones((SIZE_X, SIZE_Y, LATTICE_Q), dtype=np.float64)

N[2][2] = np.arange(0,LATTICE_Q)
print(N)

def stream(N):
    SIZE_X, SIZE_Y, LATTICE_Q = N.shape
    N_1 = np.zeros_like(N)
    ex = [0,1,0,-1,0,1,-1,-1,1]
    ey = [0,0,1,0,-1,1,1,-1,-1]
    omega = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
    for i in range(SIZE_X):
        for j in range(SIZE_Y):
            for q in range(LATTICE_Q):
                xp = (x + ex[q]) % SIZE_X
                xy = (y + ey[q]) % SIZE_Y
                N_1[xp,xy,q] = N[i,j,q]
    return N_1

"""
    Generate random convex polygon
    @param n: number of vertices
"""
def convex_polygon(n):
    # generate random points
    points = np.random.rand(n, 2)
    # compute convex hull
    hull = ConvexHull(points)
    # return points on convex hull
    return points[hull.vertices]

def ConvexHull(points):
    # find leftmost point
    leftmost_point = np.argmin(points[:,0])
    # find extreme points
    extreme_points = [leftmost_point]
    while True:
        endpoint = extreme_points[-1]
        endpoint_candidates = [i for i in range(len(points)) if i != endpoint]
        next_endpoint = endpoint_candidates[0]
        for candidate in endpoint_candidates:
            # find next endpoint
            if is_left(points[endpoint], points[candidate], points[next_endpoint]) > 0:
                next_endpoint = candidate
        if next_endpoint == extreme_points[0]:
            break
        extreme_points.append(next_endpoint)
    return extreme_points

def is_left(p0, p1, p2):
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])