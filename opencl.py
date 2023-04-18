import pyopencl as cl
import numpy as np

if __name__ == '__main__':
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    program = cl.Program(ctx, """
    kernel void addition(__global const int *a, __global const int *b){
        
    }
    """)