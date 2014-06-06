// Copyright (c) 2014, Oren Rippel and Ryan P. Adams
// All rights reserved.

// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdint.h>
#include <omp.h>
#include <mkl.h>
#include <math.h>
#include <offload.h>

#ifndef MIC_DEV
#define MIC_DEV 0
#endif

#define ALLOC alloc_if(1) free_if(0)
#define FREE alloc_if(0) free_if(1)
#define ALLOC_FREE alloc_if(1) free_if(1)
#define REUSE alloc_if(0) free_if(0)

#define ti(a, b, c, d, B, C, D) ((a)*(B)*(C)*(D) + (b)*(C)*(D) + (c)*(D) + (d))
// #define it(i, dim, B, C, D) do{
//     if (dim == 0) i / (B*C*D); \
//     else if (dim == 1) (i % (B*C*D)) / (C*D); \
//     else if (dim == 2) (i % (C*D)) / D; \
//     else if (dim == 3) i % D; \
// } while(0)

void tester(){
  int N = 100000;
  float *A = _mm_malloc(N*sizeof(float), 64);
  float *B = _mm_malloc(N*sizeof(float), 64);

  #pragma omp parallel for
    for (int n = 0; n < N; n++)
      A[n] = 0.f;

  #pragma omp parallel for
    for (int n = 0; n < N; n++)
      B[n] = A[n];

  float S = 0.f;
  #pragma omp parallel for
    for (int n = 0; n < N; n++)
      S = S + B[n];

  printf("%f", S);
}


float *allocate_host(int N){
    // float *A = _mm_malloc(N*sizeof(float), 64);
    float *A = (float *) malloc(N*sizeof(float));
    if (A == NULL){
        fprintf(stderr, "Out of memory.\n");
    }

    return A;
}

int *allocate_host_int(int N){
    int *A = (int *) malloc(N*sizeof(int));
    if (A == NULL){
        fprintf(stderr, "Out of memory.\n");
    }

    return A;
}

void fill_zeros(int N, float *restrict A, int offloaded){
  #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
  in(A:length(0) REUSE)
  { 
    int n;
    #pragma omp parallel for private(n)
        for (n = 0; n < N; n++)
          A[n] = 0.f;
  }
}

void fill_zeros_int(int N, int *restrict A, int offloaded){
  #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
  in(A:length(0) REUSE)
  { 
    int n;
    #pragma omp parallel for private(n)
        for (n = 0; n < N; n++)
          A[n] = 0;
  }
}

void fill_ones(int N, float *restrict A, int offloaded){
  #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
  in(A:length(0) REUSE)
  { 
    int i;
    #pragma omp parallel for private(i)
        for (i = 0; i < N; i++)
          A[i] = 1.f;
  }
}

// // convert linear index to tensor index
// __attribute__((target(mic:MIC_DEV))) int ti(int a, int b, int c, int d, int B, int C, int D){
//     return (a*B*C*D + b*C*D + c*D + d);
// }

// convert tensor index to linear index
__attribute__((target(mic:MIC_DEV))) int it(int i, int dim, int B, int C, int D){
    int s;

    if (dim == 0) s = i / (B*C*D);
    else if (dim == 1) s = (i % (B*C*D)) / (C*D);
    else if (dim == 2) s = (i % (C*D)) / D;
    else if (dim == 3) s = i % D;

    return s;
}

__attribute__((target(mic:MIC_DEV))) float *zeros_mic(int N){
    // float *restrict A = _mm_malloc(N*sizeof(float), 64);
    float *restrict A = (float *) malloc(N*sizeof(float));
    
    int i;
    #pragma omp parallel for private(i)
        for (i = 0; i < N; i++)
          A[i] = 0.f;

    return A;
}


__attribute__((target(mic:MIC_DEV))) float *ones_mic(int N){
    // float *restrict A = _mm_malloc(N*sizeof(float), 64);
    float *restrict A = (float *)malloc(N*sizeof(float));
    
    int i;
    #pragma omp parallel for private(i)
        for (i = 0; i < N; i++)
          A[i] = 1.f;

    return A;
}

// VSLStreamStatePtr initialize_stream(VSLStreamStatePtr stream){
//     #pragma offload target(mic:MIC_DEV) \ 
//     inout(stream)
//     { 
//         vslNewStream(&stream, VSL_BRNG_MCG31, 1);
//     }
//     return stream;
// }

void fill_randn(int skip_num, int N, float *restrict A, float mu, float sigma){
  #pragma offload target(mic:MIC_DEV) \ 
  in(A:length(0) REUSE)
  { 
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MCG31, 1);
    vslSkipAheadStream(stream, skip_num);
    vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, N, A, mu, sigma); 
  }
}

void fill_uniform(int skip_num, int N, float *restrict A){
  #pragma offload target(mic:MIC_DEV) \ 
  in(A:length(0) REUSE)
  { 
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MCG31, 1);
    vslSkipAheadStream(stream, skip_num);
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N, A, 0.0, 1.0);
  }
}

float *slice_inds(int N, int *restrict indices, float *restrict A, int indices_offloaded, int offloaded){
  float *restrict A_sliced = allocate_host(N);
  if (indices_offloaded == 0){
    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \
    in(A:length(0) REUSE) \
    in(indices:length(N) ALLOC_FREE) \
    nocopy(A_sliced:length(N) ALLOC)
    {
      int n;
      #pragma omp parallel for private(n)
      for (n = 0; n < N; n++)
        A_sliced[n] = A[indices[n]];
    }
  }
  else{
    #pragma offload target(mic:MIC_DEV) if(offloaded == 1)\
    in(A:length(0) REUSE) \
    in(indices:length(0) REUSE) \
    nocopy(A_sliced:length(N) ALLOC)
    {
      int n;
      #pragma omp parallel for private(n)
      for (n = 0; n < N; n++)
        A_sliced[n] = A[indices[n]];
    }
  }

  return A_sliced;
}

float *slice_cols(int N, int *restrict indices, int ROWS, int COLS, float *restrict A, int indices_offloaded, int offloaded){
  float *restrict A_sliced = allocate_host(N*ROWS);

  if (indices_offloaded == 0){
    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \
    in(A:length(0) REUSE) \
    in(indices:length(N) ALLOC_FREE) \
    nocopy(A_sliced:length(N*ROWS) ALLOC)
    {
      int n, r;
      #pragma omp parallel for collapse(2) private(n, r)
      for (r = 0; r < ROWS; r++)
        for (n = 0; n < N; n++)
          A_sliced[r*N + n] = A[r*COLS + indices[n]];
    }
  }
  else{
    #pragma offload target(mic:MIC_DEV) if(offloaded == 1)\
    in(A:length(0) REUSE) \
    in(indices:length(0) REUSE) \
    nocopy(A_sliced:length(N*ROWS) ALLOC)
    {
      int n, r;
      #pragma omp parallel for collapse(2) private(n, r)
      for (r = 0; r < ROWS; r++)
        for (n = 0; n < N; n++)
          A_sliced[r*N + n] = A[r*COLS + indices[n]];
    }
  }

  return A_sliced;
}

float *slice_rows(int N, int *restrict indices, int ROWS, int COLS, float *restrict A, int indices_offloaded, int offloaded){
  float *restrict A_sliced = allocate_host(N*COLS);

  if (indices_offloaded == 0){
    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \
    in(A:length(0) REUSE) \
    in(indices:length(N) ALLOC_FREE) \
    nocopy(A_sliced:length(N*COLS) ALLOC)
    {
      int c, n;
      #pragma omp parallel for collapse(2) private(n, c)
      for (c = 0; c < COLS; c++)
        for (n = 0; n < N; n++)
          A_sliced[n*COLS + c] = A[indices[n]*COLS + c];
    }
  }
  else{
    #pragma offload target(mic:MIC_DEV) if(offloaded == 1)\
    in(A:length(0) REUSE) \
    in(indices:length(0) REUSE) \
    nocopy(A_sliced:length(N*COLS) ALLOC)
    {
      int c, n;
      #pragma omp parallel for collapse(2) private(n, c)
      for (c = 0; c < COLS; c++)
        for (n = 0; n < N; n++)
          A_sliced[n*COLS + c] = A[indices[n]*COLS + c];
    }
  }

  return A_sliced;
}

void print_slice_small(int ROWS, int COLS, float *A){
      
      printf("[");
      for (int r = 0; r < ROWS; r++){
          printf("[");
          for (int c = 0; c < COLS; c++){
              printf("%f", A[r*COLS + c]);
              if (c < COLS - 1) printf("   ");
              }
              printf("]");
              if (r < ROWS-1) printf("\n");    
          }
        printf("]\n");
}

void print_slice_big(float *A){
  int COLS = 6, ROWS = 6;
      
      printf("[");
      for (int r = 0; r < 3; r++){
          printf("[");
          for (int c = 0; c < 3; c++){
              printf("%f   ", A[r*COLS + c]);
          }
          printf("...   ");
          for (int c = COLS-3; c < COLS; c++){
              printf("%f", A[r*COLS + c]);
              if (c < COLS - 1) printf("   ");
          }
          printf("]\n");
      }
      printf("...\n");

      for (int r = ROWS-3; r < ROWS; r++){
          printf("[");
          for (int c = 0; c < 3; c++){
              printf("%f   ", A[r*COLS + c]);
          }
          printf("...   ");
          for (int c = COLS-3; c < COLS; c++){
              printf("%f", A[r*COLS + c]);
              if (c < COLS - 1) printf("   ");
          }
          printf("]");
          if (r < ROWS-1) printf("\n");
      }

      printf("]");
      printf("\n");
}

void print_slice_big_col(int ROWS, float *A){
      int COLS = 6;

      printf("[");
      for (int r = 0; r < ROWS; r++){
          printf("[");
          for (int c = 0; c < 3; c++){
              printf("%f", A[r*COLS + c]);
              if (c < COLS - 1) printf("   ");
          }
          printf("...   ");
          for (int c = COLS-3; c < COLS; c++){
              printf("%f", A[r*COLS + c]);
              if (c < COLS - 1) printf("   ");
          }
          printf("]");
          if (r < ROWS-1) printf("\n");
      }
      printf("]");
      printf("\n");
}

void print_slice_big_row(int COLS, float *A){
      int ROWS = 6;

      printf("[");
      for (int r = 0; r < 3; r++){
          printf("[");
          for (int c = 0; c < COLS; c++){
              printf("%f", A[r*COLS + c]);
              if (c < COLS - 1) printf("   ");
          }
          printf("]\n");
      }
      printf("...\n");

      for (int r = ROWS-3; r < ROWS; r++){
          printf("[");
          for (int c = 0; c < COLS; c++){
              printf("%f", A[r*COLS + c]);
              if (c < COLS - 1) printf("   ");
          }
          printf("]");
          if (r < ROWS-1) printf("\n");
      }
      printf("]");
      printf("\n");
}

void print_slice(int ROWS, int COLS, float *A, int offloaded){
      float *restrict B;

      if (ROWS <= 6 && COLS <= 6){
        B = allocate_host(ROWS*COLS);
        #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \
          in(A:length(0) REUSE) \
          out(B:length(ROWS*COLS) ALLOC_FREE)
          {
            for (int n = 0; n < ROWS*COLS; n++) B[n] = A[n];
          }

        print_slice_small(ROWS, COLS, B);
      }
      
      else if (ROWS <= 6 && COLS > 6){
          B = allocate_host(6*ROWS);

          #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \
          in(A:length(0) REUSE) \
          out(B:length(6*ROWS) ALLOC_FREE)
          {
            for (int r = 0; r < ROWS; r++)
                for (int c = 0; c < 3; c++)
                    B[r*6 + c] = A[r*COLS + c];

            for (int r = 0; r < ROWS; r++)
                for (int c = COLS-3; c < COLS; c++)
                    B[r*6 + c-COLS+6] = A[r*COLS + c];
          }

        print_slice_big_col(ROWS, B);
      }
      
      else if (ROWS > 6 && COLS <= 6){
        B = allocate_host(6*COLS);

          #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \
          in(A:length(0) REUSE) \
          out(B:length(6*COLS) ALLOC_FREE)
          {
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < COLS; c++)
                    B[r*COLS + c] = A[r*COLS + c];

            for (int r = ROWS-3; r < ROWS; r++)
                for (int c = 0; c < COLS; c++)
                    B[(r-ROWS+6)*COLS + c] = A[r*COLS + c];
          }

        print_slice_big_row(COLS, B);
      }
      
      else {
          B = allocate_host(36);

          #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \
          in(A:length(0) REUSE) \
          out(B:length(36) ALLOC_FREE)
          {
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    B[r*6 + c] = A[r*COLS + c];

            for (int r = 0; r < 3; r++)
                for (int c = COLS-3; c < COLS; c++)
                    B[r*6 + c-COLS+6] = A[r*COLS + c];

            for (int r = ROWS-3; r < ROWS; r++)
                for (int c = 0; c < 3; c++)
                    B[(r-ROWS+6)*6 + c] = A[r*COLS + c];

            for (int r = ROWS-3; r < ROWS; r++)
                for (int c = COLS-3; c < COLS; c++)
                    B[(r-ROWS+6)*6 + c-COLS+6] = A[r*COLS + c];

          }
          print_slice_big(B);
      }
}

void print_slice_mic(int ROWS, int COLS, float *A){
    printf("Object on MIC:\n");

    float *restrict B = allocate_host(36);

    #pragma offload target(mic:MIC_DEV)\
    in(A:length(0) REUSE) \
    out(B:length(36) ALLOC_FREE)
    {
      for (int r = 0; r < 3; r++)
          for (int c = 0; c < 3; c++)
              B[r*6 + c] = A[r*COLS + c];

      for (int r = 0; r < 3; r++)
          for (int c = COLS-3; c < COLS; c++)
              B[r*6 + c-COLS+6] = A[r*COLS + c];

      for (int r = ROWS-3; r < ROWS; r++)
          for (int c = 0; c < 3; c++)
              B[(r-ROWS+6)*6 + c] = A[r*COLS + c];

      for (int r = ROWS-3; r < ROWS; r++)
          for (int c = COLS-3; c < COLS; c++)
              B[(r-ROWS+6)*6 + c-COLS+6] = A[r*COLS + c];

    }

  // print_slice(6, 6, B);
  free(B);
}

void offload_mic(int N, float *restrict A){
    _Offload_status mic_status;
    OFFLOAD_STATUS_INIT(mic_status);
    
    #pragma offload_transfer target(mic:MIC_DEV) status(mic_status) \ 
    in(A:length(N) ALLOC)

    if (!mic_status.result == OFFLOAD_SUCCESS){
        printf("Offload failed.\n");
        if (mic_status.result == OFFLOAD_OUT_OF_MEMORY) {
            printf("Offload failed due to insufficient memory.\n"); }
    }
}

void offload_mic_int(int N, int *restrict A){
    _Offload_status mic_status;
    OFFLOAD_STATUS_INIT(mic_status);
    
    #pragma offload_transfer target(mic:MIC_DEV) status(mic_status) \ 
    in(A:length(N) ALLOC)

    if (!mic_status.result == OFFLOAD_SUCCESS){
        printf("Offload failed.\n");
        if (mic_status.result == OFFLOAD_OUT_OF_MEMORY) {
            printf("Offload failed due to insufficient memory.\n"); }
    }
}

void pull_mic(int N, float *restrict A){

    #pragma offload_transfer target(mic:MIC_DEV) \ 
    out(A:length(N) REUSE)
}

float *unalign_host(int N, float *restrict A){

    float *restrict B = allocate_host(N);
    int n;
    #pragma omp parallel for private(n)
    for (n = 0; n < N; n++) B[n] = A[n];
    return B;
}

float output_float(float *restrict A){
  float S = A[0];
  return S;
}

void copy(int N, float *restrict A, float *restrict B, int offloaded){
    if (offloaded == 0){
        int n;
        #pragma omp parallel for private(n)
        for (n = 0; n < N; n++) A[n] = B[n];
    }

    else{
        #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
        in(B:length(0) REUSE) \
        nocopy(A:length(N) ALLOC)
        { 
            cblas_scopy(N, B, 1, A, 1);
        }
    }
}

// copies B into A on host
void replace_host(int N, float *restrict A, float *restrict B){

    int n;
    #pragma omp parallel for private(n)
    for (n = 0; n < N; n++) A[n] = B[n];
    // cblas_scopy(N, B, 1, A, 1);
}

void replace_mic(int N, float *restrict A, float *restrict B){
    #pragma offload target(mic:MIC_DEV) \ 
    in(B:length(0) REUSE) \
    in(A:length(0) REUSE)
    { 
      cblas_scopy(N, B, 1, A, 1);
    }
}

void replace_host_int(int N, int *restrict A, int *restrict B){

    int n;
    #pragma omp parallel for private(n)
    for (n = 0; n < N; n++) A[n] = B[n];
    // cblas_scopy(N, B, 1, A, 1);
}

void replace_mic_int(int N, int *restrict A, int *restrict B){
    #pragma offload target(mic:MIC_DEV) \ 
    in(B:length(0) REUSE) \
    in(A:length(0) REUSE)
    { 
      int n;
      #pragma omp parallel for private(n)
      for (n = 0; n < N; n++) A[n] = B[n];
    }
}

// copies B into A on host
void replace_partial_host(int N_A, int SHIFT_A, float *restrict A, int N_B, int SHIFT_B, float *restrict B){

    int n;
    #pragma omp parallel for private(n)
    for (n = 0; n < N_B; n++)
        A[SHIFT_A + n] = B[SHIFT_B + n];
    // cblas_scopy(N, B, 1, A, 1);
}

void replace_partial_mic(int N_A, int SHIFT_A, float *restrict A, int N_B, int SHIFT_B, float *restrict B){
    #pragma offload target(mic:MIC_DEV) \ 
    in(B:length(0) REUSE) \
    in(A:length(0) REUSE)
    { 
      cblas_scopy(N_B, B + SHIFT_B, 1, A + SHIFT_A, 1);
    }
}

float *get_partial(int N, int SHIFT, float *restrict A){
  float *restrict S = A + SHIFT;

  return S;
}

// copies A into existing memory of A on MIC
void push_mic(int N, float *restrict A){
    _Offload_status mic_status;
    OFFLOAD_STATUS_INIT(mic_status);

    #pragma offload_transfer target(mic:MIC_DEV) status(mic_status) \ 
    in(A:length(N) REUSE)

    if (!mic_status.result == OFFLOAD_SUCCESS){
        printf("Offload failed.\n");
        if (mic_status.result == OFFLOAD_OUT_OF_MEMORY) {
            printf("Offload failed due to insufficient memory.\n"); }
    }
}

float *cast_float(int N, int *restrict A, int offloaded){
  float *restrict A_float = allocate_host(N);

  #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \
  in(A:length(0) FREE) \
  nocopy(A_float:length(N) ALLOC)
  {
    int n;
    #pragma omp parallel for private(n)
    for (n = 0; n < N; n++) A_float[n] = (float) A[n];
  }
  
  free(A);
  return A_float;
}

int *cast_int(int N, float *restrict A, int offloaded){
  int *restrict A_int = allocate_host_int(N);

  #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \
  in(A:length(0) FREE) \
  nocopy(A_int:length(N) ALLOC)
  {
    int n;
    #pragma omp parallel for private(n)
    for (n = 0; n < N; n++) A_int[n] = (int) A[n];
  }
  
  free(A);
  return A_int;
}


void free_host(int N, float *A){
    // _mm_free(A);
    free(A);
}

void free_host_int(int N, int *A){
    // _mm_free(A);
    free(A);
}

void free_mic(int N, float *restrict A){
    #pragma offload_transfer target(mic:MIC_DEV) \ 
    nocopy(A:length(N) FREE)
}

void free_mic_int(int N, int *restrict A){
    #pragma offload_transfer target(mic:MIC_DEV) \ 
    nocopy(A:length(N) FREE)
}

void expo(int N, float *restrict A){
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
    
        int n;
        #pragma omp parallel for private(n)
            for (n = 0; n < N; n++) vsExp(1, A+n, A+n);
     }
}

void clip(int N, float *restrict A, float LOWER, float UPPER){
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
        int n;
        #pragma omp parallel for private(n)
            for (n = 0; n < N; n++){
              if (A[n] < LOWER) A[n] = LOWER;
              if (A[n] > UPPER) A[n] = UPPER;
            }
    }
}

void clip_low(int N, float *restrict A, float LOWER){
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
        int n;
        #pragma omp parallel for private(n)
            for (n = 0; n < N; n++){
              if (A[n] < LOWER) A[n] = LOWER;
            }
    }
}

void flooro(int N, float *restrict A){
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
    
        int n;
        #pragma omp parallel for private(n)
            for (n = 0; n < N; n++)
              vsFloor(1, A+n, A+n);
     }
}

void sign(int N, float *restrict A){
  #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
        int n;
        #pragma omp parallel for private(n)
            for (n = 0; n < N; n++){
              if (A[n] >= 0) A[n] = 1.f;
              else A[n] = -1.f;
              
            } 
    }
}

float *equal(int N, float *restrict A, float *restrict B){
  float *restrict S = allocate_host(N);
  float max_AB; 

  #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE) \
    in(B:length(0) REUSE) \
    nocopy(S:length(N) ALLOC)
    {   
        int n;
        #pragma omp parallel for private(n)
        for (n = 0; n < N; n++){
          max_AB = fmaxf(fabsf(A[n]), fabsf(B[n]));
          if (fabsf(A[n] - B[n]) <= 0.00001*max_AB) S[n] = 1.f;
          else S[n] = 0.f;
        }
    }
    return S;
}

float *leq(int N, float *restrict A, float B){
  float *restrict S = allocate_host(N);

  #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE) \
    nocopy(S:length(N) ALLOC)
    {   
        int n;
        #pragma omp parallel for private(n)
        for (n = 0; n < N; n++){
          if (A[n] <= B) S[n] = 1.f;
          else S[n] = 0.f;
        }
    }
    return S;
}

float *geq(int N, float *restrict A, float B){
  float *restrict S = allocate_host(N);

  #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE) \
    nocopy(S:length(N) ALLOC)
    {   
        int n;
        #pragma omp parallel for private(n)
        for (n = 0; n < N; n++){
          if (A[n] >= B) S[n] = 1.f;
          else S[n] = 0.f;
        }
    }
    return S;
}

float *greater(int N, float *restrict A, float B){
  float *restrict S = allocate_host(N);

  #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE) \
    nocopy(S:length(N) ALLOC)
    {   
        int n;
        #pragma omp parallel for private(n)
        for (n = 0; n < N; n++){
          if (A[n] > B) S[n] = 1.f;
          else S[n] = 0.f;
        }
    }
    return S;
}

float *elementwise_or(int N, float *restrict A, float *restrict B){
  float *restrict S = allocate_host(N);

  #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE) \
    in(B:length(0) REUSE) \
    nocopy(S:length(N) ALLOC)
    {   
        int n;
        #pragma omp parallel for private(n)
        for (n = 0; n < N; n++){
          if ((A[n] != 0.f) || (B[n] != 0.f)) S[n] = 1.f;
          else S[n] = 0.f;
        }
    }
    return S;
}

float *labels_to_vectors(int N, int K, float *restrict A){
    float *restrict S = allocate_host(N*K);

    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE) \
    nocopy(S:length(N*K) ALLOC)
    {   
    
        int n;
        #pragma omp parallel for private(n)
        for (n = 0; n < K*N; n++)
          S[n] = 0;

    
        #pragma omp parallel for private(n)
        for (n = 0; n < N; n++)
          S[n*K + (int) A[n]] = 1;
    }
    return S;
}

void lg(int N, float *restrict A){
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
    
        int n;
        #pragma omp parallel for private(n)
            for (n = 0; n < N; n++)
              vsLn(1, A+n, A+n);
        // vsLn(N, A, A);
     }
}

void abso(int N, float *restrict A){
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
        int n;
        #pragma omp parallel for private(n)
        for (n = 0; n < N; n++)
          vsAbs(1, A+n, A+n);
        // vsAbs(N, A, A);
     }
}

void sqrto(int N, float *restrict A){
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
    
        int n;
        #pragma omp parallel for private(n)
        for (n = 0; n < N; n++)
          vsSqrt(1, A+n, A+n);
        // vsSqrt(N, A, A);
     }
}

float normo(int N, float *restrict A){
    float S;
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
        S = cblas_snrm2(N, A, 1);
     }
     return S;
}

void powo(int N, float *restrict A, float b){
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
    
        int n;
        #pragma omp parallel for private(n)
        for (n = 0; n < N; n++)
          vsPowx(1, A+n, b, A+n);
          // vsPowx(N, A, b, A);
     }
}

void T(int ROWS, int COLS, float *restrict A){
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
        mkl_simatcopy('R', 'T', ROWS, COLS, 
          1.0, A, COLS, ROWS);
     }
}

void scale(int N, float *restrict A, float c){
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE) 
    { 
        cblas_sscal(N, c, A, 1);
     }
}

void mult(int ROWS_A, int COLS_A, float *restrict A, int ROWS_X, int COLS_X, float *restrict X){    
    if (COLS_X == 1 && ROWS_X > 1){
        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \ 
        in(X:length(0) REUSE)
        { 
            int r, c;
            #pragma omp parallel for collapse(2) private(r, c)
            for (r = 0; r < ROWS_A; r++)
               for (c = 0; c < COLS_A; c++)
                A[r*COLS_A + c] = A[r*COLS_A + c] * X[r];
        }
    }
    else if (ROWS_X == 1 && COLS_X > 1){
        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \ 
        in(X:length(0) REUSE)
        { 
        
            int r, c;
            #pragma omp parallel for collapse(2) private(r, c)
            for (c = 0; c < COLS_A; c++)
              for (r = 0; r < ROWS_A; r++)
                A[r*COLS_A + c] = A[r*COLS_A + c] * X[c];
        }
    }
    else if (ROWS_X == 1 && COLS_X == 1){
        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        in(X:length(0) REUSE)
        { 
          cblas_sscal(ROWS_A*COLS_A, X[0], A, 1);
        }
    } 
    else if (ROWS_X == ROWS_A && COLS_X == COLS_A){
            #pragma offload target(mic:MIC_DEV) \ 
            in(A:length(0) REUSE) \
            in(X:length(0) REUSE)
            { 
            
                 int n;
                 #pragma omp parallel for private(n)
                  for (n = 0; n < ROWS_A*COLS_A; n++)
                    vsMul(1, A+n, X+n, A+n);
                    // vsMul(ROWS_A*COLS_A, A, X, A);
             }
    } 
    else printf("Update matrix dimensions don\'t match.");
}

void invert(int N, float *restrict A){
  #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
    
        int n;
        #pragma omp parallel for private(n)
            for (n = 0; n < N; n++)
              vsInv(1, A+n, A+n);
              // vsInv(N, A, A);
     }
}

void divide(int ROWS_A, int COLS_A, float *restrict A, int ROWS_X, int COLS_X, float *restrict X){    
    if (COLS_X == 1 && ROWS_X > 1){
        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \ 
        in(X:length(0) REUSE)
        { 
            int r, c;
            #pragma omp parallel for collapse(2) private(r, c)
            for (r = 0; r < ROWS_A; r++)
               for (c = 0; c < COLS_A; c++)
                A[r*COLS_A + c] = A[r*COLS_A + c] / X[r];
        }
    }
    else if (ROWS_X == 1 && COLS_X > 1){
        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \ 
        in(X:length(0) REUSE)
        { 
            int r, c;
            #pragma omp parallel for collapse(2) private(r, c)
            for (c = 0; c < COLS_A; c++)
              for (r = 0; r < ROWS_A; r++)
                A[r*COLS_A + c] = A[r*COLS_A + c] / X[c];
        }
    }
    else if (ROWS_X == 1 && COLS_X == 1){
        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        in(X:length(0) REUSE)
        { 
          cblas_sscal(ROWS_A*COLS_A, 1.0/X[0], A, 1);
        }
    } 
    else if (ROWS_X == ROWS_A && COLS_X == COLS_A){
            #pragma offload target(mic:MIC_DEV) \ 
            in(A:length(0) REUSE) \
            in(X:length(0) REUSE)
            { 

                int n;
                #pragma omp parallel for private(n)
                  for (n = 0; n < ROWS_A*COLS_A; n++)
                    vsDiv(1, A+n, X+n, A+n);
                // vsDiv(ROWS_A*COLS_A, A, X, A);
             }
    } 
    else printf("Update matrix dimensions don\'t match.");
}

void update(int ROWS_A, int COLS_A, float *restrict A, int ROWS_X, int COLS_X, float *restrict X, float ALPHA){    
    if (COLS_X == 1 && ROWS_X > 1){
        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \ 
        in(X:length(0) REUSE)
        { 
          float *restrict Y = ones_mic(COLS_A);
          cblas_sger(CblasRowMajor, ROWS_A, COLS_A, 
            ALPHA, X, 1, Y, 1, 
            A, COLS_A);
          // _mm_free(Y);
          free(Y);
        }
    }
    else if (ROWS_X == 1 && COLS_X > 1){
        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \ 
        in(X:length(0) REUSE)
        { 
          float *restrict Y = ones_mic(ROWS_A);
          cblas_sger(CblasRowMajor, ROWS_A, COLS_A, 
            ALPHA, Y, 1, X, 1, 
            A, COLS_A);
          // _mm_free(Y);
          free(Y);
        }
    }
    else if (ROWS_X == 1 && COLS_X == 1){
        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        in(X:length(0) REUSE)
        { 
          float *restrict Y = ones_mic(ROWS_A*COLS_A);
          cblas_saxpy(ROWS_A*COLS_A, X[0]*ALPHA, Y, 1, A, 1);
          // _mm_free(Y);
          free(Y);
        }
    } 
    else if (ROWS_X == ROWS_A && COLS_X == COLS_A){
        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \ 
        in(X:length(0) REUSE)
        { 
            cblas_saxpy(ROWS_A*COLS_A, ALPHA, X, 1, A, 1);
        }
    } 
    else printf("Update matrix dimensions don\'t match.");
}

void update_const(int N, float *restrict A, float c){
    #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE)
        { 
          float *restrict Y = ones_mic(N);
          cblas_saxpy(N, c, Y, 1, A, 1);
          // _mm_free(Y);
          free(Y);
        }
}

// void fill_bernoulli(int skip_num, int N, float *restrict A, float p){
//     fill_uniform(skip_num, N, A);
//     update_const(N, A, p);
//     flooro(N, A);
// }

void fill_bernoulli(int skip_num, int N, float *restrict A, float p){
    fill_uniform(skip_num, N, A);
    update_const(N, A, p);
    flooro(N, A);

    // #pragma offload target(mic:MIC_DEV) \ 
    // in(A:length(0) REUSE)
    // { 
    //   int *B = (int *)malloc(N*sizeof(int));
    //   VSLStreamStatePtr stream;
    //   vslNewStream(&stream, VSL_BRNG_MCG31, 1);
    //   vslSkipAheadStream(stream, skip_num);
    //   viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, N, B, p);

    //   for (int n = 0; n < N; n++) A[n] = (float) B[n];

    //   free(B);
    // }
}


float *dot(int ROWS_A, int COLS_A, int T_A, float *restrict A, int COLS_B, int T_B, float *restrict B){    
    // float *restrict C = (float *)_mm_malloc(ROWS_A*COLS_B*sizeof(float), 64); 
    float *restrict C = allocate_host(ROWS_A*COLS_B);

    char TRANSPOSE_A = 'N', TRANSPOSE_B = 'N';
    
    if (T_A == 1) TRANSPOSE_A = 'T';
    if (T_B == 1) TRANSPOSE_B = 'T';
    
    float ALPHA = 1.0f, BETA = 0.0f;

    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE) \ 
    in(B:length(0) REUSE) \
    nocopy(C:length(ROWS_A*COLS_B) ALLOC)
    { 
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ROWS_A, COLS_B, COLS_A, ALPHA,
           (float *)A, COLS_A, (float *)B, COLS_B, BETA, (float *)C, COLS_B);
    } 

    return C;
}

float *dot_vec(int N, float *restrict A, float *restrict B){
  float *restrict S = allocate_host(2);
  
  #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        in(B:length(0) REUSE) \
        nocopy(S:length(2) ALLOC)
    {
        S[0] = cblas_sdot(N, A, 1, B, 1);
      }

    return S;
}

void dot_replace(int ROWS_A, int COLS_A, int T_A, float *restrict A, int ROWS_B, int COLS_B, int T_B, float *restrict B, float BETA, float *restrict C){    

    CBLAS_TRANSPOSE TRANSPOSE_A = CblasNoTrans, TRANSPOSE_B = CblasNoTrans;
    int ROWS_LEFT = ROWS_A, ROWS_RIGHT = ROWS_B, COLS_LEFT = COLS_A, COLS_RIGHT = COLS_B;
    
    if (T_A == 1){ 
      TRANSPOSE_A = CblasTrans;
      ROWS_LEFT = COLS_A;
      COLS_LEFT = ROWS_A; }

    if (T_B == 1){ 
      TRANSPOSE_B = CblasTrans;
      ROWS_RIGHT = COLS_B;
      COLS_RIGHT = ROWS_B; }

    float ALPHA = 1.0f;

    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE) \ 
    in(B:length(0) REUSE) \
    in(C:length(0) REUSE)
    { 
      cblas_sgemm(CblasRowMajor, TRANSPOSE_A, TRANSPOSE_B, ROWS_LEFT, COLS_RIGHT, COLS_LEFT, ALPHA,
           (float *)A, COLS_A, (float *)B, COLS_B, BETA, (float *)C, COLS_RIGHT);
    } 
}

void __attribute__((target(mic:MIC_DEV))) dot_mic(int ROWS_A, int COLS_A, float *restrict A, int COLS_B, float *restrict B, float *restrict C){    

    char TRANSPOSE_A = 'N', TRANSPOSE_B = 'N';
    float ALPHA = 1.0f, BETA = 0.0f;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ROWS_A, COLS_B, COLS_A, ALPHA,
           (float *)A, COLS_A, (float *)B, COLS_B, BETA, (float *)C, COLS_B);

}

float *sum_axis(int ROWS_A, int COLS_A, float *restrict A, int AXIS){
    float *restrict S;

    if (AXIS == 0){
        S = allocate_host(COLS_A);

        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        nocopy(S:length(COLS_A) ALLOC)
        {   
            float *restrict Y = ones_mic(ROWS_A);
            dot_mic(1, ROWS_A, Y, COLS_A, A, S);
            free(Y);
        }
    }
    else if (AXIS == 1){
        S = allocate_host(ROWS_A);

        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        nocopy(S:length(ROWS_A) ALLOC)
        {   
            float *restrict Y = ones_mic(COLS_A);
            dot_mic(ROWS_A, COLS_A, A, 1, Y, S);
            free(Y);
        }
    }
    else if (AXIS == 2){ 
        S = allocate_host(2);

        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        nocopy(S:length(2) ALLOC)
        {   
            float *restrict Y = ones_mic(ROWS_A*COLS_A);
            S[0] = cblas_sdot(ROWS_A*COLS_A, A, 1, Y, 1);
            free(Y);
        }
    }

    return S;
}

int *max_axis(int ROWS_A, int COLS_A, float *restrict A, int AXIS){
    int *restrict S;

    float A_MIN = 268435456;
    #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE)
        {
          int n;
          #pragma omp parallel for private(n)
          for (n = 0; n < ROWS_A*COLS_A; n++)
            {if (A[n] < A_MIN) A_MIN = A[n];}
          
          #pragma omp parallel for private(n)
          for (n = 0; n < ROWS_A*COLS_A; n++)
            A[n] = A[n] - A_MIN;
        }

    if (AXIS == 0){
        S = allocate_host_int(COLS_A);

        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        nocopy(S:length(COLS_A) ALLOC)
        {   
            int n;
            #pragma omp parallel for private(n)
              for (n = 0; n < COLS_A; n++){
                  S[n] = cblas_isamax(ROWS_A, A + n, COLS_A);
                  S[n] = S[n]*COLS_A + n;
                  }
        }
    }
    else if (AXIS == 1){
        // S = _mm_malloc(ROWS_A*sizeof(float), 64); 
        S = allocate_host_int(ROWS_A);

        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        nocopy(S:length(ROWS_A) ALLOC)
        {   
            int n;
            #pragma omp parallel for private(n)
              for (n = 0; n < ROWS_A; n++){
                S[n] = cblas_isamax(COLS_A, A + n*COLS_A, 1);
                S[n] = S[n] + n*COLS_A;
              }
        }
    }
    else if (AXIS == 2){
        // S = _mm_malloc(2*sizeof(float), 64); 
        S = allocate_host_int(2);

        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        nocopy(S:length(2) ALLOC)
        {   
            S[0] = cblas_isamax(ROWS_A*COLS_A, A, 1);
        }
    }

    #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE)
        {
          int n;
          #pragma omp parallel for private(n)
          for (n = 0; n < ROWS_A*COLS_A; n++)
            A[n] = A[n] + A_MIN;
        }

    return S;
}

void index_global_to_local(int ROWS, int COLS, int *restrict A, int AXIS){
    if (AXIS == 0){
        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE)
        {   
            int n;
            #pragma omp parallel for private(n)
            for (n = 0; n < COLS; n++){
                A[n] = (A[n] - n)/COLS;}
        }
    }
    else if (AXIS == 1){
        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE)
        {   
            int n;
            #pragma omp parallel for private(n)
              for (n = 0; n < ROWS; n++){
                A[n] = A[n] - n*COLS;
              }
        }
    }
}

float sumo(int N, float *restrict A){
        float S;
        
        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE)
        {   
            float *restrict Y = ones_mic(N);
            S = cblas_sdot(N, A, 1, Y, 1);
            // _mm_free(Y);
            free(Y);
        }
        return S;
}

int *convolve_and_pool(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int pool_radius, int stride, int *ARGMAXS, int argmaxs_fixed){

  int output_H = H - Y + 1;
  int output_W = W - X + 1;
  int pooled_H = ceil(((float) output_H)/pool_radius);
  int pooled_W = ceil(((float) output_W)/pool_radius);

  #pragma offload target(mic:MIC_DEV) \ 
  in(INPUTS:length(0) REUSE) \ 
  in(FILTERS:length(0) REUSE) \ 
  in(OUTPUTS:length(0) REUSE) \
  in(ARGMAXS:length(0) REUSE)
  {
      float convolution;
      int n, k, h, w, c, y, x, lin_index, h_arg, w_arg;
      if (argmaxs_fixed == 0){
          #pragma omp parallel for private(n)
          for (n = 0; n < N*K*pooled_H*pooled_W; n++) OUTPUTS[n] = -1.0e10;

          #pragma omp parallel for collapse(4) private(n, k, h, w, convolution, c, y, x)
          for (n = 0; n < N; n++){
              for (k = 0; k < K; k++){                  
                  // loop over 2D pre-pooled map given n, k
                  for (h = 0; h < output_H; h += stride){
                      for (w = 0; w < output_W; w += stride){
                          // Code needs to not be parallelized from here on; make sure variables are not shared

                          // compute convolution for a particular set of n, k, h, w
                          // then push to pooled layer
                          convolution = 0.f;
                          for (c = 0; c < C; c++)
                            for (y = 0; y < Y; y++)
                              for (x = 0; x < X; x++)
                                convolution += INPUTS[ti(n, c, h + y, w + x, C, H, W)]
                                             * FILTERS[ti(k, c, y, x, C, Y, X)];

                          // #pragma omp critical
                          if (convolution > OUTPUTS[ti(n, k, h/pool_radius, w/pool_radius, K, pooled_H, pooled_W)]){
                            OUTPUTS[ti(n, k, h/pool_radius, w/pool_radius, K, pooled_H, pooled_W)] = convolution;
                            ARGMAXS[ti(n, k, h/pool_radius, w/pool_radius, K, pooled_H, pooled_W)] = ti(n, k, h, w, K, output_H, output_W);
                          }
                        
                      }
                  }
              }
          }
      }

      else{
          #pragma omp parallel for private(n)
          for (n = 0; n < N*K*pooled_H*pooled_W; n++) OUTPUTS[n] = 0.f;

          #pragma omp parallel for collapse(7) private(n, k, h, w, c, y, x, lin_index, h_arg, w_arg)
          // loop over pooled elements
          for (n = 0; n < N; n++)
              for (k = 0; k < K; k++)
                    for (h = 0; h < pooled_H; h++)
                        for (w = 0; w < pooled_W; w++)
                            // add up convolution
                              for (c = 0; c < C; c++)
                                  for (y = 0; y < Y; y++)
                                      for (x = 0; x < X; x++){
                                          lin_index = ARGMAXS[ti(n, k, h, w, K, pooled_H, pooled_W)];
                                          h_arg = it(lin_index, 2, K, output_H, output_W);
                                          w_arg = it(lin_index, 3, K, output_H, output_W);
                                          OUTPUTS[ti(n, k, h, w, K, pooled_H, pooled_W)] += INPUTS[ti(n, c, h_arg + y, w_arg + x, C, H, W)]
                                                       * FILTERS[ti(k, c, y, x, C, Y, X)];

                                      }
      }
         
  }

  return ARGMAXS;
}

void convolve_gradient(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, int pool_radius, float *D_INPUTS, float *D_FILTERS){

  int output_H = H - Y + 1;
  int output_W = W - X + 1;
  int pooled_H = ceil(((float) output_H)/pool_radius);
  int pooled_W = ceil(((float) output_W)/pool_radius);

  #pragma offload target(mic:MIC_DEV) \ 
  in(INPUTS:length(0) REUSE) \ 
  in(FILTERS:length(0) REUSE) \ 
  in(ARGMAXS:length(0) REUSE) \
  in(D_INPUTS:length(0) REUSE) \
  in(D_POOLED_OUTPUTS:length(0) REUSE) \
  in(D_FILTERS:length(0) REUSE)
  {
      int k, c, x, y, n, h, w, h_pooled, w_pooled, lin_index, h_arg, w_arg;

      // filter gradient computation
      #pragma omp parallel for collapse(7) private(k, c, x, y, n, h_pooled, w_pooled, lin_index, h_arg, w_arg)
      // loop over elements of filter
      for (k = 0; k < K; k++)
          for (c = 0; c < C; c++)
              for (x = 0; x < X; x++)
                  for (y = 0; y < Y; y++)

                      for (n = 0; n < N; n++)
                          // loop over 2D map of pooled output
                          for (h_pooled = 0; h_pooled < pooled_H; h_pooled++)
                            for (w_pooled = 0; w_pooled < pooled_W; w_pooled++){
                                // argmax is over the output image
                                lin_index = ARGMAXS[ti(n, k, h_pooled, w_pooled, K, pooled_H, pooled_W)];
                                h_arg = it(lin_index, 2, K, output_H, output_W);
                                w_arg = it(lin_index, 3, K, output_H, output_W);

                                D_FILTERS[ti(k, c, y, x, C, Y, X)] += D_POOLED_OUTPUTS[ti(n, k, h_pooled, w_pooled, K, pooled_H, pooled_W)]
                                       * INPUTS[ti(n, c, h_arg + y, w_arg + x, C, H, W)];

                                D_INPUTS[ti(n, c, h_arg + y, w_arg + x, C, H, W)] += D_POOLED_OUTPUTS[ti(n, k, h_pooled, w_pooled, K, pooled_H, pooled_W)]
                                       * FILTERS[ti(k, c, y, x, C, Y, X)];
                            }

      // // input gradient computation
      // #pragma omp parallel for collapse(7) private(n, c, h_pooled, w_pooled, k, x, y, lin_index, h_arg, w_arg)
      // // loop over elements of input
      // for (n = 0; n < N; n++)
      //     for (c = 0; c < C; c++)
      //         for (h_pooled = 0; h_pooled < pooled_H; h_pooled++)
      //             for (w_pooled = 0; w_pooled < pooled_W; w_pooled++)
      //                 for (k = 0; k < K; k++)
      //                     for (x = 0; x < X; x++)
      //                         for (y = 0; y < Y; y++){
      //                           lin_index = ARGMAXS[ti(n, k, h_pooled, w_pooled, K, pooled_H, pooled_W)];
      //                           h_arg = it(lin_index, 2, K, output_H, output_W);
      //                           w_arg = it(lin_index, 3, K, output_H, output_W);
  
      //                           D_INPUTS[ti(n, c, h_arg + y, w_arg + x, C, H, W)] += D_POOLED_OUTPUTS[ti(n, k, h_pooled, w_pooled, K, pooled_H, pooled_W)]
      //                                  * FILTERS[ti(k, c, y, x, C, Y, X)];
      //                       }
  
  }
}

void convolve(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int tight){
  // C channels per input, per filter

  #pragma offload target(mic:MIC_DEV) \ 
  in(INPUTS:length(0) REUSE) \ 
  in(FILTERS:length(0) REUSE) \ 
  in(OUTPUTS:length(0) REUSE)
  {
      float *output_scratch = (float *) malloc((H + Y - 1)*(W + X - 1)*sizeof(float));

      // convolution mode can also be set manually to be VSL_CONV_MODE_FFT, VSL_CONV_MODE_DIRECT
      int INPUTS_SHAPE[] = {H, W};
      int FILTERS_SHAPE[] = {Y, X};
      int OUTPUTS_SHAPE[] = {H + Y - 1, W + X - 1};

      int INPUTS_STRIDE[] = {W, 1};
      int FILTERS_STRIDE[] = {-X, -1};
      int OUTPUTS_STRIDE[] = {W + X - 1, 1};
      
      int output_H = H + Y - 1;
      int output_W = W + X - 1;
      if (tight == 1){
          output_H = H - Y + 1;
          output_W = W - X + 1;
      }

      VSLConvTaskPtr ConvTask;
      
      // #pragma omp parallel for
      for (int n = 0; n < N; n++){
          for (int c = 0; c < C; c++){
              float *input = &INPUTS[(n*C + c)*H*W];
              
              vslsConvNewTaskX(&ConvTask, VSL_CONV_MODE_AUTO, 2, INPUTS_SHAPE, FILTERS_SHAPE, OUTPUTS_SHAPE, input, INPUTS_STRIDE);
              
              for (int k = 0; k < K; k++){
                  float *filter = &FILTERS[(k*C + c)*X*Y];
                  float *output = &OUTPUTS[(n*K + k)*output_H*output_W];

                  vslsConvExecX(ConvTask, filter, FILTERS_STRIDE, output_scratch, OUTPUTS_STRIDE);

                  // max-pooling here, before tightening convolution even
                  // need to output argmax's of indices (corresponding to indices of padded array?)
                  if (tight == 1){
                      for (int h = 0; h < output_H; h++)
                        for (int w = 0; w < output_W; w++)
                          output_scratch[h*output_W + w] = output_scratch[(h + Y - 1)*(W + X - 1) + (w + X - 1)];
                  }

                  if (c == 0) cblas_scopy(output_H*output_W, output_scratch, 1, output, 1);
                  else cblas_saxpy(output_H*output_W, 1., output_scratch, 1, output, 1);
              }
          }
          vslConvDeleteTask(&ConvTask);
      }

    free(output_scratch);      
  }

}

void check_mic_status(){
    _Offload_status mic_status;
    OFFLOAD_STATUS_INIT(mic_status);

    int NUM_MIC;
    #pragma offload target(mic) status(mic_status) mandatory
    { NUM_MIC = _Offload_get_device_number(); }
         
    if (NUM_MIC < 0)
        printf("Found no MICs.");

    if (mic_status.result == OFFLOAD_SUCCESS) {
        printf("Offload test was successful.\n\n"); } 
    else {
        printf("Offload failed.\n");
        if (mic_status.result == OFFLOAD_OUT_OF_MEMORY) {
            printf("Offload failed due to insufficient memory.\n"); }
    }
}

void ping_each_core(){
    #pragma offload target(mic:MIC_DEV)
   {
       #pragma omp parallel
       {
             #ifdef __MIC__
                printf("MIC: greetings from thread %d out of %d.\n",
                       omp_get_thread_num(), omp_get_num_threads());
                fflush(0);
             #else
                printf("HOST: greetings from thread %d out of %d.\n",
                       omp_get_thread_num(), omp_get_num_threads());
                fflush(0);             
            #endif
       }
  }
}
