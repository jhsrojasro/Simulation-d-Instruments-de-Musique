kernel void addition(__global const float *A, __global const float *B){
  int i = get_global_id(0);
  A[i] = A[i] + B[i];
}