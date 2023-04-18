#define RHO 1.0f

constant float beta = 1.03f; // 
constant int dimension = 3;
constant float omega[3] = {1 / 2.0f, 1.04f, 1.05f};

kernel void addition(__global const float *A, __global const float *B, float alpha){
  int i = get_global_id(0);
  A[i] = A[i] + B[i];
  for(int k=0; k<3; k=k+1){
    A[i] = A[i] + omega[k];
  }
  float sum = 0;
  for(int k=0; K<3; k++){
      sum += omega[k];
  }
  A[i] -= sum;
}

kernel void difference(__global float *A, __global const float *B){
  int i = get_global_id(0);
  A[i] = A[i] - B[i];
}

