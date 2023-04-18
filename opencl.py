import pyopencl as cl
import numpy as np

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx) # objet dont on envoie les taches
prog = cl.Program(ctx, open("addition.cl").read()).buld()
a_np = np.random.rand(10).astype(np.float32)
b_np = np.random.rand(10).astype(np.float32)

mf = cl.mem_flags

a_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

prog.addition(queue, (10,), None, a_g, b_g, np.float32(1.03))
prog.difference(queue, (10,), None a_g, b_g)
res_np = np.random.rand(10).astype(np.float32)

cl.enqueue_copy(queue, res_np, a_g)
print(res_np)
print(np.allclose(res_np, a_g))