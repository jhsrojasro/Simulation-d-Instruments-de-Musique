import numpy as np
import time


N = 1000
X = list(range(N))
A = np.array(X)
# print( type(X) )
# print( type(A) )
# print(A.dtype)
# print(type(X[0]))

# print(A + A)
# print(A * A)
# print(2 * A)
A1 = 1.5 * A
# print(A1.dtype)
A2 = np.zeros((N,N))
A3 = np.zeros_like(A)
A4 = np.ones(N, dtype=np.float64)

U = [[i*N +j for j in range(N)]  for i in range(N)]
V = [[0 for j in range(N)]  for i in range(N)]

t1 = time.clock_gettime_ns(time.CLOCK_PROCESS_CPUTIME_ID)
for i in range(N):
    for j in range(N):
        V[i][j] = sum([U[i][k] * U[k][j] for k in range(N)])

t2 = time.clock_gettime_ns(time.CLOCK_PROCESS_CPUTIME_ID)


M = np.array(range(N*N))
M.shape = (N,N)
S = np.zeros_like(M)

t3 = time.clock_gettime_ns(time.CLOCK_PROCESS_CPUTIME_ID)

for i in range(N):
    for j in range(N):
        S[i][j] = np.sum(M[i,:] * M[:,j])

t4 = time.clock_gettime_ns(time.CLOCK_PROCESS_CPUTIME_ID)

R = np.matmul(M,M)

t5 = time.clock_gettime_ns(time.CLOCK_PROCESS_CPUTIME_ID)

print(t2 - t1, t4 - t3, t5 - t4)