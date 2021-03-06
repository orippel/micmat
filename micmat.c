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
#include <assert.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <immintrin.h>

#include <generated_macros.h>
#include <macros.h>

void speed_tester(int S, float *restrict INPUTS, float *restrict OUTPUTS){
    #pragma offload target(mic:MIC_DEV)\ 
    in(INPUTS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE)
    {

        __cilkrts_end_cilk();
        if (0!= __cilkrts_set_param("nworkers", "236"))
        {
            printf("Failed to set worker count\n");
            exit(0);
        }
        __cilkrts_init();

        int i, j, k, chunk;
        int CHUNK_SIZE = 1000;
        // int CHUNKS = N/CHUNK_SIZE;
        // printf("Chunks %d \n", CHUNKS);

        // #pragma simd
        // #pragma ivdep
        // #pragma vector aligned
        // #pragma omp parallel for \
        //     schedule(dynamic, 10) \
        //     default(none) \
        //     private(i, j, k) \
        //     shared(INPUTS, OUTPUTS, CHUNK_SIZE)

        cilk_for (int i = 0; i < 5000; i++){
            // float *outputs_pointer = OUTPUTS + i*S;

            cilk_for (int j = 0; j < 5000; j++){
                cilk_for (int k = 0; k < 5000; k++){
                    OUTPUTS[i*5000 + j] += INPUTS[i*5000 + k] * INPUTS[k*5000 + j];
                }

                // outputs_pointer++;
            }
        }
    }
}

void tester(){
  int N = 100000;
  float *A = _mm_malloc(N*sizeof(float), ALIGN);
  float *B = _mm_malloc(N*sizeof(float), ALIGN);

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
    float *A = _mm_malloc(N*sizeof(float), ALIGN);
    __assume_aligned(A, ALIGN);
    // float *A = (float *) malloc(N*sizeof(float));
    if (A == NULL){
        fprintf(stderr, "Out of memory.\n");
    }

    return A;
}

float *allocate_host_unaligned(int N){
    // float *A = _mm_malloc(N*sizeof(float), ALIGN);
    float *A = (float *) malloc(N*sizeof(float));
    if (A == NULL){
        fprintf(stderr, "Out of memory.\n");
    }

    return A;
}

int *allocate_host_int(int N){
    // int *A = (int *) malloc(N*sizeof(int));
    int *A = _mm_malloc(N*sizeof(int), ALIGN);
    __assume_aligned(A, ALIGN);

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
    int num_cache_lines_64 = N/64;
    int N_aligned = N/64*64;
    int N_remaining = N - N_aligned;   
 
    #pragma omp parallel for private(n) shared(N, N_aligned, num_cache_lines_64)
        for (n = 0; n < num_cache_lines_64; n++)
    {
#ifdef __MIC__
        __m512 v_zero = _mm512_set1_ps(0.0);
                _mm512_storenrngo_ps((float *)(A + 64*n),        v_zero);
                _mm512_storenrngo_ps((float *)(A + 64*n + 16),   v_zero);
                _mm512_storenrngo_ps((float *)(A + 64*n + 32),   v_zero);
                _mm512_storenrngo_ps((float *)(A + 64*n + 48),   v_zero);

#endif
          //A[n] = 0.f;
    }
     A[N_aligned : N_remaining] = 0.f;
  }
}

void fill_zeros_int(int N, int *restrict A, int offloaded){
  #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
  in(A:length(0) REUSE)
  { 
    int n;
    int num_cache_lines_64 = N/64;
    int N_aligned = N/64*64;
    int N_remaining = N - N_aligned;   

    #pragma omp parallel for private(n) shared(N, N_aligned, num_cache_lines_64)
        for (n = 0; n < num_cache_lines_64; n++)
    {
#ifdef __MIC__
        __m512i v_zero = _mm512_set1_epi32(0);
                _mm512_extstore_epi32((float *)(A + 64*n),       v_zero, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
                _mm512_extstore_epi32((float *)(A + 64*n + 16),   v_zero, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE); 
                _mm512_extstore_epi32((float *)(A + 64*n + 32),   v_zero, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
                _mm512_extstore_epi32((float *)(A + 64*n + 48),   v_zero, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);

#endif
    }
     A[N_aligned : N_remaining] = 0;
  }
}

void fill_ones(int N, float *restrict A, int offloaded){
  #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
  in(A:length(0) REUSE)
  { 
    int n;
    int num_cache_lines_64 = N/64;
    int N_aligned = N/64*64;
    int N_remaining = N - N_aligned;   
 
    #pragma omp parallel for private(n) shared(N, N_aligned, num_cache_lines_64)
        for (n = 0; n < num_cache_lines_64; n++)
    {
#ifdef __MIC__
        __m512 v_one = _mm512_set1_ps(1.0f);
                _mm512_storenrngo_ps((float *)(A + 64*n),        v_one);
                _mm512_storenrngo_ps((float *)(A + 64*n + 16),   v_one);
                _mm512_storenrngo_ps((float *)(A + 64*n + 32),   v_one);
                _mm512_storenrngo_ps((float *)(A + 64*n + 48),   v_one);

#endif
    }
     A[N_aligned : N_remaining] = 1.f;
  }
}

void fill_ones_int(int N, int *restrict A, int offloaded){
  #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
  in(A:length(0) REUSE)
  { 
    int n;
    int num_cache_lines_64 = N/64;
    int N_aligned = N/64*64;
    int N_remaining = N - N_aligned;   
 
    #pragma omp parallel for private(n) shared(N, N_aligned, num_cache_lines_64)
        for (n = 0; n < num_cache_lines_64; n++)
    {
#ifdef __MIC__
        __m512i v_one = _mm512_set1_epi32(1);
                _mm512_extstore_epi32((float *)(A + 64*n),       v_one, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
                _mm512_extstore_epi32((float *)(A + 64*n + 16),   v_one, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE); 
                _mm512_extstore_epi32((float *)(A + 64*n + 32),   v_one, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
                _mm512_extstore_epi32((float *)(A + 64*n + 48),   v_one, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);

#endif
    }
     A[N_aligned : N_remaining] = 1;
  }
}

// void fill_zeros(int N, float *restrict A, int offloaded){
//   #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
//   in(A:length(0) REUSE)
//   { 
//     A[0:N] = 0;
//     // int n;
//     // #pragma omp parallel for private(n)
//     //     for (n = 0; n < N; n++)
//     //       A[n] = 0.f;
//   }
// }

// void fill_zeros_int(int N, int *restrict A, int offloaded){
//   #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
//   in(A:length(0) REUSE)
//   { 
//     int n;
//     #pragma omp parallel for private(n)
//         for (n = 0; n < N; n++)
//           A[n] = 0;
//   }
// }

// void fill_ones(int N, float *restrict A, int offloaded){
//   #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
//   in(A:length(0) REUSE)
//   { 
//     A[0:N] = 1.f;
//     // int i;
//     // #pragma omp parallel for private(i)
//     //     for (i = 0; i < N; i++)
//     //       A[i] = 1.f;
//   }
// }

// void fill_ones_int(int N, int *restrict A, int offloaded){
//   #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
//   in(A:length(0) REUSE)
//   { 
//     int n;
//     #pragma omp parallel for private(n)
//         for (n = 0; n < N; n++)
//           A[n] = 1;
//   }
// }

void fill_const(int N, float *restrict A, float c, int offloaded){
  #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
  in(A:length(0) REUSE)
  { 
    int i;
    #pragma omp parallel for private(i)
        for (i = 0; i < N; i++)
          A[i] = c;
  }
}

void fill_const_int(int N, int *restrict A, int c, int offloaded){
  #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
  in(A:length(0) REUSE)
  { 
    int n;
    #pragma omp parallel for private(n)
        for (n = 0; n < N; n++)
          A[n] = c;
  }
}


// // convert linear index to tensor index
// __attribute__((target(mic:MIC_DEV))) int ti(int a, int b, int c, int d, int B, int C, int D){
//     return (a*B*C*D + b*C*D + c*D + d);
// }

// // convert tensor index to linear index
// __attribute__((target(mic:MIC_DEV))) int it(int i, int dim, int B, int C, int D){
//     int s;

//     if (dim == 0) s = i / (B*C*D);
//     else if (dim == 1) s = (i % (B*C*D)) / (C*D);
//     else if (dim == 2) s = (i % (C*D)) / D;
//     else if (dim == 3) s = i % D;

//     return s;
// }

__attribute__((target(mic:MIC_DEV))) float *zeros_mic(int N){
    float *restrict A = _mm_malloc(N*sizeof(float), ALIGN);
    // float *restrict A = (float *) malloc(N*sizeof(float));
    
    int i;
    #pragma omp parallel for private(i)
        for (i = 0; i < N; i++)
          A[i] = 0.f;

    return A;
}


__attribute__((target(mic:MIC_DEV))) float *ones_mic(int N){
    float *restrict A = _mm_malloc(N*sizeof(float), ALIGN);
    // float *restrict A = (float *)malloc(N*sizeof(float));
    
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

void slice_inds_replace(int N, int *restrict indices, float *restrict A, int indices_offloaded, int offloaded, float *restrict scratch){
  if (indices_offloaded == 0){
    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \
    in(A:length(0) REUSE) \
    in(indices:length(N) ALLOC_FREE) \
    nocopy(scratch:length(0) REUSE)
    {
      int n;
      #pragma omp parallel for private(n)
      for (n = 0; n < N; n++)
        scratch[n] = A[indices[n]];
    }
  }
  else{
    #pragma offload target(mic:MIC_DEV) if(offloaded == 1)\
    in(A:length(0) REUSE) \
    in(indices:length(0) REUSE) \
    nocopy(scratch:length(0) REUSE)
    {
      int n;
      #pragma omp parallel for private(n)
      for (n = 0; n < N; n++)
        scratch[n] = A[indices[n]];
    }
  }
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
  // free(B);
    _mm_free(B);
}

void offload_mic(int N, float *restrict A){
    _Offload_status mic_status;
    OFFLOAD_STATUS_INIT(mic_status);
    
    #pragma offload_transfer target(mic:MIC_DEV) status(mic_status) \ 
    in(A:length(N) ALLOC)

    if (!mic_status.result == OFFLOAD_SUCCESS){
        printf("Offload failed.\n");
        if (mic_status.result == OFFLOAD_OUT_OF_MEMORY)
            printf("Offload failed due to insufficient memory.\n");
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

    // float *restrict B = allocate_host(N);
    float *restrict B = allocate_host_unaligned(N);
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

void shift_int(int N, int shift, int *restrict A){
    
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
      A[shift:N] = A[0:N];
    }
}

// copies B into A on host
void replace_host(int N, float *restrict A, float *restrict B){

    A[0:N] = B[0:N];
    // int n;
    // #pragma omp parallel for private(n)
    // for (n = 0; n < N; n++) A[n] = B[n];
    //  (N, B, 1, A, 1);
}

void replace_mic(int N, float *restrict A, float *restrict B){
    #pragma offload target(mic:MIC_DEV) \ 
    in(B:length(0) REUSE) \
    in(A:length(0) REUSE)
    { 
      // cblas_scopy(N, B, 1, A, 1);
        int n;
        int num_cache_lines_64 = N/64;
        int N_aligned = N/64*64;
        int N_remaining = N - N_aligned;  
        // printf("N %d, N_aligned %d, N_remaining %d, num_cache_lines_64 %d \n", N, N_aligned, N_remaining, num_cache_lines_64);
 
        #pragma omp parallel for private(n) shared(N, N_aligned, num_cache_lines_64)
        for (n = 0; n < num_cache_lines_64; n++)
        {
    #ifdef __MIC__
                    __m512 load_1 = _mm512_extload_ps(B + 64*n, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                    __m512 load_2 = _mm512_extload_ps(B + 64*n + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                    __m512 load_3 = _mm512_extload_ps(B + 64*n + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                    __m512 load_4 = _mm512_extload_ps(B + 64*n + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                    _mm512_storenrngo_ps((float *)(A + 64*n),        load_1);
                    _mm512_storenrngo_ps((float *)(A + 64*n + 16),   load_2);
                    _mm512_storenrngo_ps((float *)(A + 64*n + 32),   load_3);
                    _mm512_storenrngo_ps((float *)(A + 64*n + 48),   load_4);

    #endif
        }
        A[N_aligned : N_remaining] = B[N_aligned : N_remaining];
    }
}

void replace_host_int(int N, int *restrict A, int *restrict B){

    A[0:N] = B[0:N];
    // int n;
    // #pragma omp parallel for private(n)
    // for (n = 0; n < N; n++) A[n] = B[n];
    // cblas_scopy(N, B, 1, A, 1);
}

void replace_mic_int(int N, int *restrict A, int *restrict B){
    #pragma offload target(mic:MIC_DEV) \ 
    in(B:length(0) REUSE) \
    in(A:length(0) REUSE)
    { 
        A[0:N] = B[0:N];
      // int n;
      // #pragma omp parallel for private(n)
      // for (n = 0; n < N; n++) A[n] = B[n];
    }
}

// copies B into A on host
void replace_partial_host(int N_A, int SHIFT_A, float *restrict A, int N_B, int SHIFT_B, float *restrict B){

    A[SHIFT_A: N_B] = B[SHIFT_B : N_B];
    // int n;
    // #pragma omp parallel for private(n)
    // for (n = 0; n < N_B; n++)
    //     A[SHIFT_A + n] = B[SHIFT_B + n];
}

void replace_partial_mic(int N_A, int SHIFT_A, float *restrict A, int N_B, int SHIFT_B, float *restrict B){
    #pragma offload target(mic:MIC_DEV) \ 
    in(B:length(0) REUSE) \
    in(A:length(0) REUSE)
    { 
      // cblas_scopy(N_B, B + SHIFT_B, 1, A + SHIFT_A, 1);
        A += SHIFT_A;
        B += SHIFT_B;
        
        int n;
        int num_cache_lines_64 = N_B/64;
        int N_aligned = N_B/64*64;
        int N_remaining = N_B - N_aligned;   
 
        #pragma omp parallel for private(n) shared(N_B, N_aligned, num_cache_lines_64)
        for (n = 0; n < num_cache_lines_64; n++)
    {
#ifdef __MIC__
                __m512 load_1 = _mm512_extload_ps(B + 64*n, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 load_2 = _mm512_extload_ps(B + 64*n + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 load_3 = _mm512_extload_ps(B + 64*n + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 load_4 = _mm512_extload_ps(B + 64*n + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                _mm512_storenrngo_ps((float *)(A + 64*n),        load_1);
                _mm512_storenrngo_ps((float *)(A + 64*n + 16),   load_2);
                _mm512_storenrngo_ps((float *)(A + 64*n + 32),   load_3);
                _mm512_storenrngo_ps((float *)(A + 64*n + 48),   load_4);

#endif
    }
        A[N_aligned : N_remaining] = B[N_aligned : N_remaining];

        // int n;
        // #pragma omp parallel for private(n)
        // for (n = 0; n < N_B; n++){
                    // A[SHIFT_A + n] = B[SHIFT_B + n];}
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

void free_host(int N, float *A){
    _mm_free(A);
    // free(A);
}

void free_host_int(int N, int *A){
    _mm_free(A);
    // free(A);
}

float *cast_float(int N, int *restrict A, int offloaded){
  float *restrict A_float = allocate_host(N);

  #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \
  in(A:length(0) FREE) \
  nocopy(A_float:length(N) ALLOC)
  {
    int n;
    // #pragma vector aligned
    #pragma omp parallel for private(n)
    for (n = 0; n < N; n++) A_float[n] = (float) A[n];
  }
  
  free_host_int(N, A);
  return A_float;
}

int *cast_int(int N, float *restrict A, int offloaded){
  int *restrict A_int = allocate_host_int(N);
  #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \
  in(A:length(0) FREE) \
  nocopy(A_int:length(N) ALLOC)
  {
    int n;
    // #pragma vector aligned
    #pragma omp parallel for private(n)
    for (n = 0; n < N; n++) A_int[n] = (int) A[n];
  }
  
  free_host(N, A);
  return A_int;
}

void cast_int_replace(int N, float *restrict A, int *restrict OUTPUT, int offloaded){
    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \
    in(A:length(0) REUSE) \
    in(OUTPUT:length(0) REUSE)  
    {
        // OUTPUT[0:N] = (int) A[0:N];
        int n;
        int num_cache_lines_64 = N/64;
        int N_aligned = N/64*64;
        int N_remaining = N - N_aligned;   
 
        #pragma omp parallel for private(n) shared(N, N_aligned, num_cache_lines_64)
        for (n = 0; n < num_cache_lines_64; n++)
        {
        #ifdef __MIC__
                        __m512 load_1 = _mm512_extload_ps(A + 64*n, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                        __m512 load_2 = _mm512_extload_ps(A + 64*n + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                        __m512 load_3 = _mm512_extload_ps(A + 64*n + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                        __m512 load_4 = _mm512_extload_ps(A + 64*n + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                        _mm512_extstore_epi32(OUTPUT + 64*n, _mm512_cvtfxpnt_round_adjustps_epi32(load_1, _MM_FROUND_TO_NEG_INF, _MM_EXPADJ_NONE),      _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
                        _mm512_extstore_epi32(OUTPUT + 64*n + 16, _mm512_cvtfxpnt_round_adjustps_epi32(load_2, _MM_FROUND_TO_NEG_INF, _MM_EXPADJ_NONE), _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
                        _mm512_extstore_epi32(OUTPUT + 64*n + 32, _mm512_cvtfxpnt_round_adjustps_epi32(load_3, _MM_FROUND_TO_NEG_INF, _MM_EXPADJ_NONE), _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
                        _mm512_extstore_epi32(OUTPUT + 64*n + 48, _mm512_cvtfxpnt_round_adjustps_epi32(load_4, _MM_FROUND_TO_NEG_INF, _MM_EXPADJ_NONE), _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);

        #endif
        }
        OUTPUT[N_aligned : N_remaining] = (int) A[N_aligned : N_remaining];
    }
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

// void clip_low(int N, float *restrict A, float LOWER){
//     #pragma offload target(mic:MIC_DEV) \ 
//     in(A:length(0) REUSE)
//     { 
//         if (A[0:N] < LOWER){
//             A[0:N] = LOWER;
//         }
//         // int n;
//         // #pragma omp parallel for private(n) shared(LOWER)
//         //     for (n = 0; n < N; n++){
//         //       if (A[n] < LOWER) A[n] = LOWER;
//         //     }
//     }
// }

void clip_low(int N, float *restrict A, float LOWER){
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
    int n;  
        int num_cache_lines_64 = N/64;
        int N_aligned = N/64*64;
        int N_remaining = N - N_aligned;   
 
        #pragma omp parallel for private(n) shared(N, N_aligned, LOWER, num_cache_lines_64)
        for (n = 0; n < num_cache_lines_64; n++)
    {
#ifdef __MIC__
                __m512 A_load_1 = _mm512_extload_ps(A + 64*n, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 A_load_2 = _mm512_extload_ps(A + 64*n + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 A_load_3 = _mm512_extload_ps(A + 64*n + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 A_load_4 = _mm512_extload_ps(A + 64*n + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        __m512 v_LOWER = _mm512_set1_ps(LOWER);
        // __mmask16 mask_1 = _mm512_cmplt_ps_mask(A_load_1, v_LOWER);
        // __mmask16 mask_2 = _mm512_cmplt_ps_mask(A_load_2, v_LOWER);
        // __mmask16 mask_3 = _mm512_cmplt_ps_mask(A_load_3, v_LOWER);
        // __mmask16 mask_4 = _mm512_cmplt_ps_mask(A_load_4, v_LOWER);
        __mmask16 mask_1 = _mm512_cmp_ps_mask(A_load_1, v_LOWER, _MM_CMPINT_LT);
        __mmask16 mask_2 = _mm512_cmp_ps_mask(A_load_2, v_LOWER, _MM_CMPINT_LT);
        __mmask16 mask_3 = _mm512_cmp_ps_mask(A_load_3, v_LOWER, _MM_CMPINT_LT);
        __mmask16 mask_4 = _mm512_cmp_ps_mask(A_load_4, v_LOWER, _MM_CMPINT_LT);
        A_load_1 = _mm512_mask_mov_ps(A_load_1, mask_1, v_LOWER); 
        A_load_2 = _mm512_mask_mov_ps(A_load_2, mask_2, v_LOWER); 
        A_load_3 = _mm512_mask_mov_ps(A_load_3, mask_3, v_LOWER); 
        A_load_4 = _mm512_mask_mov_ps(A_load_4, mask_4, v_LOWER);
                _mm512_extstore_ps((float *)(A + 64*n),        A_load_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                _mm512_extstore_ps((float *)(A + 64*n + 16),   A_load_2, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                _mm512_extstore_ps((float *)(A + 64*n + 32),   A_load_3, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                _mm512_extstore_ps((float *)(A + 64*n + 48),   A_load_4, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
#endif
    }
        if(A[N_aligned : N_remaining] < LOWER) A[N_aligned : N_remaining]  = LOWER;

        //for (n = N_aligned; n < N; n++){
        //  if (A[n] < LOWER) A[n] = LOWER;
        //}
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
        if (A[0:N] >= 0.f){
            A[0:N] = 1.f;
        }
        else{
            A[0:N] = -1.f;
        }
        // int n;
        // #pragma omp parallel for private(n)
        //     for (n = 0; n < N; n++){
        //       if (A[n] >= 0) A[n] = 1.f;
        //       else A[n] = -1.f;
              
        //     } 
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
            S[n] = (A[n] <= B) ? 1.f : 0.f;
          // if (A[n] <= B) S[n] = 1.f;
          // else S[n] = 0.f;
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
        #pragma omp parallel for private(n) shared(B)
        for (n = 0; n < N; n++){
            // S[n] = (A[n] >= B) ? 1.f : 0.f;
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
        #pragma omp parallel for private(n) shared(B)
        for (n = 0; n < N; n++){
          // S[n] = (A[n] > B) ? 1.f : 0.f;
          if (A[n] > B) S[n] = 1.f;
          else S[n] = 0.f;
        }
    }
    return S;
}

void greater_replace(int N, float *restrict A, float B){
  #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    {   
        // if (A[0:N] > B){
        //     A[0:N] = 1.f;
        // }
        // else{
        //     A[0:N] = 0.f;
        // }
        int n;
        int num_cache_lines_64 = N/64;
        int N_aligned = N/64*64;
        int N_remaining = N - N_aligned;   
 
        #pragma omp parallel for private(n) shared(B, N, N_aligned, num_cache_lines_64)
        for (n = 0; n < num_cache_lines_64; n++)
    {
#ifdef __MIC__
        __m512 load_1 = _mm512_extload_ps(A + 64*n, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        __m512 load_2 = _mm512_extload_ps(A + 64*n + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        __m512 load_3 = _mm512_extload_ps(A + 64*n + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        __m512 load_4 = _mm512_extload_ps(A + 64*n + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        __m512 v_B = _mm512_set1_ps(B);
        __mmask16 mask_1 = _mm512_cmplt_ps_mask(v_B, load_1); // _mm512_mask_mov_ps??
        __mmask16 mask_2 = _mm512_cmplt_ps_mask(v_B, load_2);
        __mmask16 mask_3 = _mm512_cmplt_ps_mask(v_B, load_3);
        __mmask16 mask_4 = _mm512_cmplt_ps_mask(v_B, load_4);
        __m512 v_one = _mm512_set1_ps(1.0);
        __m512 v_A_1 = _mm512_setzero_ps();
        __m512 v_A_2 = _mm512_setzero_ps();
        __m512 v_A_3 = _mm512_setzero_ps();
        __m512 v_A_4 = _mm512_setzero_ps();
        v_A_1 = _mm512_mask_mov_ps(v_A_1, mask_1, v_one); 
        v_A_2 = _mm512_mask_mov_ps(v_A_2, mask_2, v_one); 
        v_A_3 = _mm512_mask_mov_ps(v_A_3, mask_3, v_one); 
        v_A_4 = _mm512_mask_mov_ps(v_A_4, mask_4, v_one); 
        _mm512_extstore_ps((float *)(A + 64*n),        v_A_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
        _mm512_extstore_ps((float *)(A + 64*n + 16),   v_A_2, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
        _mm512_extstore_ps((float *)(A + 64*n + 32),   v_A_3, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
        _mm512_extstore_ps((float *)(A + 64*n + 48),   v_A_4, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
#endif
    }
    if(A[N_aligned : N_remaining] > B) A[N_aligned : N_remaining] = 1.f; else A[N_aligned : N_remaining] = 0.f;
        //for (n = N_aligned; n < N; n++){
        //    if (A[n] > B) A[n] = 1.f;
        //    else A[n] = 0.f;
        //}

        // int n;
        // #pragma omp parallel for private(n) shared(B)
        // for (n = 0; n < N; n++){
        //     if (A[n] > B) A[n] = 1.f;
        //     else A[n] = 0.f;
        // }
    }
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

float *labels_to_vectors(int N, int K, float *restrict A, float *restrict targets){
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE) \
    in(targets:length(0) REUSE)
    {   
    
        int n;
        
        #pragma omp parallel for private(n)
        for (n = 0; n < N; n++)
          targets[n*K + (int) A[n]] = 1.f;
    }
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

float frac_nan(int N, float *restrict A){
    // float test = 1.0/0.0;
    // printf("NaN %f\n", test);
    
    // int boo = test == test;
    // printf("bool %d\n", boo);

    // boo = test == INFINITY;
    // printf("bool inf %d\n", boo);

    // test = 0.0/0.0;
    // printf("NaN %f\n", test);
    
    // boo = test == test;
    // printf("bool %d\n", boo);    
    float sum = 0.0f;
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
        #pragma omp parallel for reduction(+:sum)
        for (int n = 0; n < N; n++){
            if ((A[n] != A[n]) || (A[n] == INFINITY) || (A[n] == -INFINITY)) sum += 1.f;
        }
        sum /= (float) N;
    }
    return sum;
}

void treat_nan(int N, float *restrict A){
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
        if ((A[0:N] != A[0:N]) || (A[0:N] == INFINITY) || (A[0:N] == -INFINITY)) A[0:N] = 0.f;
    }
}

void powo(int N, float *restrict A, float b){
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
        int n;
        int num_cache_lines_64 = N/64;
        int N_aligned = N/64*64;
        int N_remaining = N - N_aligned;   
        
        #pragma omp parallel for private(n) shared(N, N_aligned, num_cache_lines_64)
        for (n = 0; n < N_aligned; n += 64)
        {
            #ifdef __MIC__
            float *A_pointer = A + n;

            __m512 A_load_1 = _mm512_extload_ps(A_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
            __m512 A_load_2 = _mm512_extload_ps(A_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
            __m512 A_load_3 = _mm512_extload_ps(A_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
            __m512 A_load_4 = _mm512_extload_ps(A_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
            
            __m512 v_pow = _mm512_set1_ps(b);
            
            _mm512_extstore_ps((float *)(A_pointer),        _mm512_pow_ps(A_load_1, v_pow), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
            _mm512_extstore_ps((float *)(A_pointer + 16),   _mm512_pow_ps(A_load_2, v_pow), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
            _mm512_extstore_ps((float *)(A_pointer + 32),   _mm512_pow_ps(A_load_3, v_pow), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
            _mm512_extstore_ps((float *)(A_pointer + 48),   _mm512_pow_ps(A_load_4, v_pow), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
            #endif
        }
        A[N_aligned : N_remaining] = pow(A[N_aligned : N_remaining], b);

        // A[0 : N_remaining] = pow(A[N_aligned : N_remaining], b);
        // int n;
        // #pragma omp parallel for private(n)
        // for (n = 0; n < N; n++)
        //   vsPowx(1, A+n, b, A+n);
        //  // vsPowx(N, A, b, A);
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
    if (ROWS_X == ROWS_A && COLS_X == COLS_A){
        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        in(X:length(0) REUSE)
        { 
            // A[0:ROWS_A*COLS_A] *= X[0:ROWS_A*COLS_A];
            int N = ROWS_A * COLS_A; 
            int n;
            int num_cache_lines_64 = N/64;
            int N_aligned = N/64*64;
            int N_remaining = N - N_aligned;   
            
            #pragma omp parallel for private(n) shared(N, N_aligned, num_cache_lines_64)
            // for (n = 0; n < num_cache_lines_64; n++)
            for (n = 0; n < N_aligned; n += 64)
            {
                #ifdef __MIC__
                float *X_pointer = X + n;
                float *A_pointer = A + n;

                __m512 load_1 = _mm512_extload_ps(X_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 load_2 = _mm512_extload_ps(X_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 load_3 = _mm512_extload_ps(X_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 load_4 = _mm512_extload_ps(X_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 A_load_1 = _mm512_extload_ps(A_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 A_load_2 = _mm512_extload_ps(A_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 A_load_3 = _mm512_extload_ps(A_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 A_load_4 = _mm512_extload_ps(A_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                _mm512_extstore_ps((float *)(A_pointer),        _mm512_mul_ps(A_load_1, load_1), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                _mm512_extstore_ps((float *)(A_pointer + 16),   _mm512_mul_ps(A_load_2, load_2), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                _mm512_extstore_ps((float *)(A_pointer + 32),   _mm512_mul_ps(A_load_3, load_3), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                _mm512_extstore_ps((float *)(A_pointer + 48),   _mm512_mul_ps(A_load_4, load_4), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                #endif
            }
            A[N_aligned : N_remaining] *= X[N_aligned : N_remaining];
        }
    }
    else if (COLS_X == 1 && ROWS_X > 1){
        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \ 
        in(X:length(0) REUSE)
        { 
            // int r, c;
            // #pragma omp parallel for private(r, c)
            // for (r = 0; r < ROWS_A; r++)
            //    for (c = 0; c < COLS_A; c++)
            //         A[r*COLS_A + c] *= X[r];
            int r;
            #pragma omp parallel for private(r)
            for (r = 0; r < ROWS_A; r++)
                A[r*COLS_A : COLS_A] *= X[r];
        }
    }
    else if (ROWS_X == 1 && COLS_X > 1){
        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \ 
        in(X:length(0) REUSE)
        { 
        
            // int r, c;
            // #pragma omp parallel for private(r, c)
            // for (c = 0; c < COLS_A; c++)
            //   for (r = 0; r < ROWS_A; r++)
            //     A[r*COLS_A + c] *= X[c];
            int c;
            #pragma omp parallel for private(c)
            for (c = 0; c < COLS_A; c++)
                    A[c : ROWS_A : COLS_A] *= X[c];
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

                A[0:ROWS_A*COLS_A] /= X[0:ROWS_A*COLS_A];
                // int n;
                // #pragma omp parallel for private(n)
                //   for (n = 0; n < ROWS_A*COLS_A; n++)
                //     vsDiv(1, A+n, X+n, A+n);
                // vsDiv(ROWS_A*COLS_A, A, X, A);
             }
    } 
    else printf("Update matrix dimensions don\'t match.");
}

void update(int a1, int a2, int a3, int a4, int a5, int a6, float *A, int b1, int b2, int b3, int b4, int b5, int b6, float *B, float ALPHA, int offloaded){
    // printf("a1 %d, a2 %d, a3 %d, a4 %d, a5 %d, a6 %d\n", a1, a2, a3, a4, a5, a6);
    // printf("b1 %d, b2 %d, b3 %d, b4 %d, b5 %d, b6 %d\n\n", b1, b2, b3, b4, b5, b6);
    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \
    in(A:length(0) REUSE) \
    in(B:length(0) REUSE)
    {   
        if (a1 == b1 && a2 == b2 & a3 == b3 && a4 == b4 && a5 == b5 && a6 == b6){
            int N = a1*a2*a3*a4*a5*a6; 
            int n;
            int num_cache_lines_64 = N/64;
            int N_aligned = N/64*64;
            int N_remaining = N - N_aligned;   
            
            #pragma omp parallel for private(n) shared(N, N_aligned, num_cache_lines_64)
            // for (n = 0; n < num_cache_lines_64; n++)
            for (n = 0; n < N_aligned; n += 64)
            {
                #ifdef __MIC__
                float *B_pointer = B + n;
                float *A_pointer = A + n;

                __m512 v_alpha = _mm512_set1_ps(ALPHA);

                __m512 B_load_1 = _mm512_extload_ps(B_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 B_load_2 = _mm512_extload_ps(B_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 B_load_3 = _mm512_extload_ps(B_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 B_load_4 = _mm512_extload_ps(B_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 A_load_1 = _mm512_extload_ps(A_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 A_load_2 = _mm512_extload_ps(A_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 A_load_3 = _mm512_extload_ps(A_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 A_load_4 = _mm512_extload_ps(A_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                _mm512_extstore_ps((float *)(A_pointer),        _mm512_add_ps(A_load_1, _mm512_mul_ps(B_load_1, v_alpha)), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                _mm512_extstore_ps((float *)(A_pointer + 16),   _mm512_add_ps(A_load_2, _mm512_mul_ps(B_load_2, v_alpha)), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                _mm512_extstore_ps((float *)(A_pointer + 32),   _mm512_add_ps(A_load_3, _mm512_mul_ps(B_load_3, v_alpha)), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                _mm512_extstore_ps((float *)(A_pointer + 48),   _mm512_add_ps(A_load_4, _mm512_mul_ps(B_load_4, v_alpha)), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                #endif
            }
            A[N_aligned : N_remaining] += ALPHA*B[N_aligned : N_remaining];

            // int a4a5a6 = a4*a5*a6;
            // #pragma omp parallel for shared(a4a5a6)
            // for (int i1i2i3 = 0; i1i2i3 < a1*a2*a3; i1i2i3++){
            //     A[i1i2i3*a4a5a6 : a4a5a6] += ALPHA * B[i1i2i3*a4a5a6 : a4a5a6];
            // }
        }
        // else if (b1 == 1 & b2 == 1 & b3 == 1 && b4 == 1 && b5 == 1 && a6 == b6){
        //     // printf("HI\n");
        //     int N = a6; 
        //     int K = a1*a2*a3*a4*a5;
        //     int a, n;
        //     int num_cache_lines_64 = N/64;
        //     int N_aligned = N/64*64;
        //     int N_remaining = N - N_aligned;   
        //     // printf("N %d, K %d, N_aligned %d\n\n", N, K, N_aligned);
            
        //     #pragma omp parallel for private(n, a) shared(N, K, N_aligned, num_cache_lines_64)
        //     // for (n = 0; n < num_cache_lines_64; n++)
        //     for (n = 0; n < N_aligned; n += 64)
        //     {
        //         #ifdef __MIC__
        //         float *B_pointer = B + n;
        //         __m512 v_alpha = _mm512_set1_ps(ALPHA);
        //         __m512 B_load_1 = _mm512_extload_ps(B_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        //         __m512 B_load_2 = _mm512_extload_ps(B_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        //         __m512 B_load_3 = _mm512_extload_ps(B_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        //         __m512 B_load_4 = _mm512_extload_ps(B_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        //         __m512 B_out_1 = _mm512_mul_ps(B_load_1, v_alpha);
        //         __m512 B_out_2 = _mm512_mul_ps(B_load_2, v_alpha);
        //         __m512 B_out_3 = _mm512_mul_ps(B_load_3, v_alpha);
        //         __m512 B_out_4 = _mm512_mul_ps(B_load_4, v_alpha);
                
        //         for (a = 0; a < K; a++){
        //             float *A_pointer = A + a*N + n;
    
        //             __m512 A_load_1 = _mm512_extload_ps(A_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        //             __m512 A_load_2 = _mm512_extload_ps(A_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        //             __m512 A_load_3 = _mm512_extload_ps(A_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        //             __m512 A_load_4 = _mm512_extload_ps(A_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        //             _mm512_extstore_ps((float *)(A_pointer),        _mm512_add_ps(A_load_1, B_out_1), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
        //             _mm512_extstore_ps((float *)(A_pointer + 16),   _mm512_add_ps(A_load_2, B_out_2), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
        //             _mm512_extstore_ps((float *)(A_pointer + 32),   _mm512_add_ps(A_load_3, B_out_3), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
        //             _mm512_extstore_ps((float *)(A_pointer + 48),   _mm512_add_ps(A_load_4, B_out_4), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
        //         }
        //         #endif
        //     }
        //     #pragma omp parallel for private(n, a) shared(N, N_aligned, N_remaining)
        //     for (a = 0; a < K; a++){
        //         // for (n = N_aligned; n < N_aligned + N_remaining; n++){
        //           A[a*N + N_aligned : N_remaining] += ALPHA * B[N_aligned : N_remaining];
        //         // }    
        //     }
            
        //     // A[N_aligned : N_remaining] += ALPHA*B[N_aligned : N_remaining];

        //     // int a4a5a6 = a4*a5*a6;
        //     // #pragma omp parallel for shared(a4a5a6)
        //     // for (int i1i2i3 = 0; i1i2i3 < a1*a2*a3; i1i2i3++){
        //     //     A[i1i2i3*a4a5a6 : a4a5a6] += ALPHA * B[i1i2i3*a4a5a6 : a4a5a6];
        //     // }
        // }

        else if (b1 == 1 & b2 == 1 & b3 == 1 && a4 == b4 && a5 == b5 && a6 == b6){
            int N = a4*a5*a6; 
            int K = a1*a2*a3;
            int a, n;
            int num_cache_lines_64 = N/64;
            int N_aligned = N/64*64;
            int N_remaining = N - N_aligned;   
            
            #pragma omp parallel for private(n, a) shared(N, K, N_aligned, num_cache_lines_64)
            // for (n = 0; n < num_cache_lines_64; n++)
            for (n = 0; n < N_aligned; n += 64)
            {
                #ifdef __MIC__
                float *B_pointer = B + n;
                __m512 v_alpha = _mm512_set1_ps(ALPHA);
                __m512 B_load_1 = _mm512_extload_ps(B_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 B_load_2 = _mm512_extload_ps(B_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 B_load_3 = _mm512_extload_ps(B_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 B_load_4 = _mm512_extload_ps(B_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 B_out_1 = _mm512_mul_ps(B_load_1, v_alpha);
                __m512 B_out_2 = _mm512_mul_ps(B_load_2, v_alpha);
                __m512 B_out_3 = _mm512_mul_ps(B_load_3, v_alpha);
                __m512 B_out_4 = _mm512_mul_ps(B_load_4, v_alpha);
                
                for (a = 0; a < K; a++){
                    float *A_pointer = A + a*N + n;
    
                    __m512 A_load_1 = _mm512_extload_ps(A_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                    __m512 A_load_2 = _mm512_extload_ps(A_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                    __m512 A_load_3 = _mm512_extload_ps(A_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                    __m512 A_load_4 = _mm512_extload_ps(A_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                    _mm512_extstore_ps((float *)(A_pointer),        _mm512_add_ps(A_load_1, B_out_1), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                    _mm512_extstore_ps((float *)(A_pointer + 16),   _mm512_add_ps(A_load_2, B_out_2), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                    _mm512_extstore_ps((float *)(A_pointer + 32),   _mm512_add_ps(A_load_3, B_out_3), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                    _mm512_extstore_ps((float *)(A_pointer + 48),   _mm512_add_ps(A_load_4, B_out_4), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                }
                #endif
            }
            #pragma omp parallel for private(n, a) shared(N, N_aligned, N_remaining)
            for (a = 0; a < K; a++){
                // for (n = N_aligned; n < N_aligned + N_remaining; n++){
                    A[a*N + N_aligned : N_remaining] += ALPHA * B[N_aligned : N_remaining];
                // }    
            }
        }

        else{
            float (*restrict A_6D)[a2][a3][a4][a5][a6] = (float (*)[a2][a3][a4][a5][a6]) A;
            float (*restrict B_6D)[b2][b3][b4][b5][b6] = (float (*)[b2][b3][b4][b5][b6]) B;

            #pragma omp parallel for
            for (int i1i2i3 = 0; i1i2i3 < a1*a2*a3; i1i2i3++){
                int i1 = i1i2i3 / (a2*a3);
                int i2 = md(i1i2i3, a2*a3) / a3;
                int i3 = md(i1i2i3, a3);

                for (int i4 = 0; i4 < a4; i4++){
                    for (int i5 = 0; i5 < a5; i5++){
                        for (int i6 = 0; i6 < a6; i6++){
                            A_6D[i1][i2][i3][i4][i5][i6] += ALPHA * B_6D[mn(b1-1, i1)][mn(b2-1, i2)][mn(b3-1, i3)][mn(b4-1, i4)][mn(b5-1, i5)][mn(b6-1, i6)];
                        }
                    }
                }
            }
        }
        // A_6D[1:a1][1:a2][1:a3][1:a4][1:a5][1:a6] += B_6D[1:b1][1:b2][1:b3][1:b4][1:b5][1:b6];
        
        // int c123456, c1, c2, c3, c4, c5, c6, mod1, mod2, mod3, mod4, mod5, mod6;
        // #pragma omp parallel for private(c123456, c1, c2, c3, c4, c5, c6, mod1, mod2, mod3, mod4, mod5, mod6)
        // for (c123456 = 0; c123456 < a1*a2*a3*a4*a5*a6; c123456++){
        //     c1 = c123456 / (a2*a3*a4*a5*a6);
        //     c2 = (c123456 % (a2*a3*a4*a5*a6)) / (a3*a4*a5*a6);
        //     c3 = (c123456 % (a3*a4*a5*a6)) / (a4*a5*a6);
        //     c4 = (c123456 % (a4*a5*a6)) / (a5*a6);
        //     c5 = (c123456 % (a5*a6)) / a6;
        //     c6 = c123456 % a6;

        //     mod1 = (c1 < b1-1) ? c1 : b1-1;
        //     mod2 = (c2 < b2-1) ? c2 : b2-1;
        //     mod3 = (c3 < b3-1) ? c3 : b3-1;
        //     mod4 = (c4 < b4-1) ? c4 : b4-1;
        //     mod5 = (c5 < b5-1) ? c5 : b5-1;
        //     mod6 = (c6 < b6-1) ? c6 : b6-1;

        //     // A[ti(c1, c2, c3, c4, a2, a3, a4)] += ALPHA * B[ti(mod1, mod2, mod3, mod4, b2, b3, b4)];
        //     A[((((c1*a2 + c2)*a3 + c3)*a4 + c4)*a5 + c5)*a6 + c6] += ALPHA * B[((((mod1*b2 + mod2)*b3 + mod3)*b4 + mod4)*b5 + mod5)*b6 + mod6];
        // }

    }
}

// void update(int ROWS_A, int COLS_A, float *restrict A, int ROWS_X, int COLS_X, float *restrict X, float ALPHA){    
//     if (COLS_X == 1 && ROWS_X > 1){
//         #pragma offload target(mic:MIC_DEV) \ 
//         in(A:length(0) REUSE) \ 
//         in(X:length(0) REUSE)
//         { 
//           float *restrict Y = ones_mic(COLS_A);
//           cblas_sger(CblasRowMajor, ROWS_A, COLS_A, 
//             ALPHA, X, 1, Y, 1, 
//             A, COLS_A);
//           // _mm_free(Y);
//           free(Y);
//         }
//     }
//     else if (ROWS_X == 1 && COLS_X > 1){
//         #pragma offload target(mic:MIC_DEV) \ 
//         in(A:length(0) REUSE) \ 
//         in(X:length(0) REUSE)
//         { 
//           float *restrict Y = ones_mic(ROWS_A);
//           cblas_sger(CblasRowMajor, ROWS_A, COLS_A, 
//             ALPHA, Y, 1, X, 1, 
//             A, COLS_A);
//           // _mm_free(Y);
//           free(Y);
//         }
//     }
//     else if (ROWS_X == 1 && COLS_X == 1){
//         #pragma offload target(mic:MIC_DEV) \ 
//         in(A:length(0) REUSE) \
//         in(X:length(0) REUSE)
//         { 
//           float *restrict Y = ones_mic(ROWS_A*COLS_A);
//           cblas_saxpy(ROWS_A*COLS_A, X[0]*ALPHA, Y, 1, A, 1);
//           // _mm_free(Y);
//           free(Y);
//         }
//     } 
//     else if (ROWS_X == ROWS_A && COLS_X == COLS_A){
//         #pragma offload target(mic:MIC_DEV) \ 
//         in(A:length(0) REUSE) \ 
//         in(X:length(0) REUSE)
//         { 
//             cblas_saxpy(ROWS_A*COLS_A, ALPHA, X, 1, A, 1);
//         }
//     } 
//     else printf("Update matrix dimensions don\'t match.");
// }

void update_const(int N, float *restrict A, float c){
    #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE)
        { 
            int n;
            int num_cache_lines_64 = N/64;
            int N_aligned = N/64*64;
            int N_remaining = N - N_aligned;   
            
            #pragma omp parallel for private(n) shared(N, N_aligned, num_cache_lines_64)
            for (n = 0; n < N_aligned; n += 64)
            {
                #ifdef __MIC__
                float *A_pointer = A + n;

                __m512 v_c = _mm512_set1_ps(c);

                __m512 A_load_1 = _mm512_extload_ps(A_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 A_load_2 = _mm512_extload_ps(A_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 A_load_3 = _mm512_extload_ps(A_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 A_load_4 = _mm512_extload_ps(A_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                _mm512_extstore_ps((float *)(A_pointer),        _mm512_add_ps(A_load_1, v_c), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                _mm512_extstore_ps((float *)(A_pointer + 16),   _mm512_add_ps(A_load_2, v_c), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                _mm512_extstore_ps((float *)(A_pointer + 32),   _mm512_add_ps(A_load_3, v_c), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                _mm512_extstore_ps((float *)(A_pointer + 48),   _mm512_add_ps(A_load_4, v_c), _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                #endif
            }
            A[N_aligned : N_remaining] += c;
          // float *restrict Y = ones_mic(N);
          // cblas_saxpy(N, c, Y, 1, A, 1);
          // _mm_free(Y);
          // free(Y);
        }
}

// void fill_bernoulli(int skip_num, int N, float *restrict A, float p){
//     fill_uniform(skip_num, N, A);
//     update_const(N, A, p);
//     flooro(N, A);
// }

void fill_bernoulli(int skip_num, int N, int *restrict A, float p){
    // fill_uniform(skip_num, N, A);
    // update_const(N, A, p);
    // flooro(N, A);

    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
      // int *B = (int *)malloc(N*sizeof(int));
      VSLStreamStatePtr stream;
      vslNewStream(&stream, VSL_BRNG_MCG31, 1);
      vslSkipAheadStream(stream, skip_num);
      viRngBernoulli(VSL_RNG_METHOD_UNIFORM_STD, stream, N, A, p);
    }
}

void fill_uniform_int(int skip_num, int N, int *restrict A, int i_start, int i_end){

    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
      VSLStreamStatePtr stream;
      vslNewStream(&stream, VSL_BRNG_MCG31, 1);
      vslSkipAheadStream(stream, skip_num);
      viRngUniform(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, N, A, i_start, i_end);

    }
}

void fill_geometric(int skip_num, int N, float *restrict A, float p){
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    { 
      int *B = _mm_malloc(N*sizeof(int), ALIGN);
      VSLStreamStatePtr stream;
      vslNewStream(&stream, VSL_BRNG_MCG31, 1);
      vslSkipAheadStream(stream, skip_num);
      viRngGeometric(VSL_RNG_METHOD_GEOMETRIC_ICDF, stream, N, B, p);

      for (int n = 0; n < N; n++) A[n] = (float) (B[n] + 1);

        _mm_free(B);
      // free(B);
    }
}

void fill_nested(int N, int K, float *restrict A, float *restrict GEOMETRIC_SAMPLES){
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE) \
    in(GEOMETRIC_SAMPLES:length(0) REUSE)
    { 
        int n, k;
        #pragma omp parallel for collapse(2) private(n, k)
        for (n = 0; n < N; n++)
            for (k = 0; k < K; k++)
                if (k < (int) GEOMETRIC_SAMPLES[n]) A[k + n*K] = 1.f;
                else A[k + n*K] = 0.f;
    }
}

void apply_dropout(int N, int K, float *restrict A, int *restrict MASK){
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE) \
    in(MASK:length(0) REUSE)
    { 
        int n, k;
        #pragma omp parallel for private(n, k)
        for (n = 0; n < N; n++){
            for (k = 0; k < K; k++){
                // A[n*K + k] *= ((float) MASK[k]);      
                A[n*K + k] *= ((float) MASK[n*K + k]);      
            }
        } 
    }
}




float *dot(int ROWS_A, int COLS_A, int T_A, float *restrict A, int COLS_B, int T_B, float *restrict B){    
    // float *restrict C = (float *)_mm_malloc(ROWS_A*COLS_B*sizeof(float), ALIGN); 
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
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE) \ 
    in(B:length(0) REUSE) \
    in(C:length(0) REUSE)
    { 
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
        // cblas_sgemm(CblasRowMajor, TRANSPOSE_A, TRANSPOSE_B, ROWS_LEFT, COLS_RIGHT, COLS_LEFT, ALPHA,
           // (float *)A, COLS_A, (float *)B, COLS_B, BETA, (float *)C, COLS_RIGHT);

        cblas_sgemm(CblasRowMajor, TRANSPOSE_A, TRANSPOSE_B, ROWS_LEFT, COLS_RIGHT, COLS_LEFT, ALPHA,
           A, COLS_A, B, COLS_B, BETA, C, COLS_RIGHT);
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
            // free(Y);
            _mm_free(Y);
        }
    }
    else if (AXIS == 1){ // broadcasting sum along axis 1
        S = allocate_host(ROWS_A);

        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        nocopy(S:length(ROWS_A) ALLOC)
        {   
            float *restrict Y = ones_mic(COLS_A);
            dot_mic(ROWS_A, COLS_A, A, 1, Y, S);
            // free(Y);
            _mm_free(Y);
        }
    }
    else if (AXIS == 2){ 
        S = allocate_host(2);

        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        nocopy(S:length(2) ALLOC)
        {   
            // S[0] = __sec_reduce_add(A[0 : ROWS_A*COLS_A]);

            // float sum = 0.0f;
            // #pragma omp parallel for reduction(+:sum)
            // for (int i = 0; i < ROWS_A*COLS_A; i++) {
            //   _mm_prefetch((char*)(A+i+128), _MM_HINT_T0);
            //   sum += A[i];
            // }
            // S[0] = sum;

            int n;
            int N = ROWS_A*COLS_A;
            int num_cache_lines_64 = N/64;
            int N_aligned = N/64*64;
            int N_remaining = N - N_aligned;   
            
            float sum = 0.0f;
            #pragma omp parallel for reduction(+:sum) private(n) shared(N, N_aligned, num_cache_lines_64)
            for (n = 0; n < num_cache_lines_64; n++)
            {
                #ifdef __MIC__
                _mm_prefetch((char*)(A + 64*n + 128), _MM_HINT_T0);
                float *A_pointer = A + 64*n;
                __m512 A_load_1 = _mm512_extload_ps(A_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 A_load_2 = _mm512_extload_ps(A_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 A_load_3 = _mm512_extload_ps(A_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 A_load_4 = _mm512_extload_ps(A_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                sum += _mm512_reduce_add_ps(A_load_1);
                sum += _mm512_reduce_add_ps(A_load_2);
                sum += _mm512_reduce_add_ps(A_load_3);
                sum += _mm512_reduce_add_ps(A_load_4);
                #endif
            }
            sum += __sec_reduce_add(A[N_aligned : N_remaining]);
            S[0] = sum;
            
        }
    }

    return S;
}

float *sum_axis_replace(int ROWS_A, int COLS_A, float *restrict A, int AXIS, float *restrict S){
    if (AXIS == 0){
        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        in(S:length(0) REUSE)
        {   
            int row, col;
            #pragma omp parallel for \
            private(row, col)
            for (col = 0; col < COLS_A; col++){
                S[col] = 0.f;
                for (row = 0; row < ROWS_A; row++){
                    S[col] += A[row*COLS_A + col];
                }
            }
            // float *restrict Y = ones_mic(ROWS_A);
            // dot_mic(1, ROWS_A, Y, COLS_A, A, S);
            // free(Y);
        }
    }
    else if (AXIS == 1){
        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        in(S:length(0) REUSE)
        {   
            // float *restrict Y = ones_mic(COLS_A);
            // dot_mic(ROWS_A, COLS_A, A, 1, Y, S);
            // // free(Y);
            // _mm_free(Y);

            int row, col;            
            #pragma omp parallel for private(row, col) shared(ROWS_A, COLS_A)
            for (row = 0; row < ROWS_A; row++) S[row] = __sec_reduce_add(A[row*COLS_A : COLS_A]);
        }
    }
    else if (AXIS == 2){ 
        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        in(S:length(0) REUSE)
        {   
            float *restrict Y = ones_mic(ROWS_A*COLS_A);
            S[0] = cblas_sdot(ROWS_A*COLS_A, A, 1, Y, 1);
            // free(Y);
            _mm_free(Y);
        }
    }

    return S;
}

// float *sum_axis(int a1, int a2, int a3, int a4, float *A, int AXIS, int offloaded){
//     int s1 = a1, s2 = a2, s3 = a3, s4 = a4, mod1, mod2, mod3, mod4;
//     if (AXIS == 0) s1 = 1;
//     if (AXIS == 1) s2 = 1;
//     if (AXIS == 2) s3 = 1;
//     if (AXIS == 3) s4 = 1;
//     if (AXIS == -1) s1 = 1, s2 = 1, s3 = 1, s4 = 1;

//     int allocate_size = s1*s2*s3*s4;
//     if (allocate_size == 1) allocate_size = 2;

//     float *S = allocate_host(allocate_size);
//     if (offloaded == 1) offload_mic(allocate_size, S);
//     fill_zeros(allocate_size, S, offloaded);

//     #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \
//     in(A:length(0) REUSE) \
//     nocopy(S:length(allocate_size) ALLOC)
//     {
//         int c1234, c1, c2, c3, c4;
//         #pragma omp parallel for private(c1234, c1, c2, c3, c4, mod1, mod2, mod3, mod4)
//         for (c1234 = 0; c1234 < a1*a2*a3*a4; c1234++){
//             c1 = c1234 / (a2*a3*a4);
//             c2 = (c1234 % (a2*a3*a4)) / (a3*a4);
//             c3 = (c1234 % (a3*a4)) / a4;
//             c4 = c1234 % a4;

//             mod1 = (c1 < s1-1) ? c1 : s1-1;
//             mod2 = (c2 < s2-1) ? c2 : s2-1;
//             mod3 = (c3 < s3-1) ? c3 : s3-1;
//             mod4 = (c4 < s4-1) ? c4 : s4-1;

//             #pragma omp atomic
//             S[ti(mod1, mod2, mod3, mod4, s2, s3, s4)] += A[ti(c1, c2, c3, c4, a2, a3, a4)];
//         }
//     }

//     return S;
// }

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
        S = allocate_host_int(ROWS_A);

        #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        nocopy(S:length(ROWS_A) ALLOC)
        {   
            int n;
            #pragma omp parallel for private(n)
              for (n = 0; n < ROWS_A; n++){
                S[n] = cblas_isamax(COLS_A, A + n*COLS_A, 1);
                S[n] += n*COLS_A;
              }
        }
    }
    else if (AXIS == 2){
        // S = _mm_malloc(2*sizeof(float), ALIGN); 
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

void max_axis_replace(int ROWS_A, int COLS_A, float *restrict A, int AXIS, float *restrict S){
    #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        in(S:length(0) REUSE)
    {

        if (AXIS == 0){
            int n;
            #pragma omp parallel for private(n)
            for (n = 0; n < COLS_A; n++){
                S[n] = __sec_reduce_max(A[n : ROWS_A : COLS_A]);
            }
            
        }
        else if (AXIS == 1){
            int n;
            #pragma omp parallel for private(n)
              for (n = 0; n < ROWS_A; n++){
                S[n] = __sec_reduce_max(A[n*COLS_A : COLS_A]);
            }
            
        }
        else if (AXIS == 2){
            S[0] = cblas_isamax(ROWS_A*COLS_A, A, 1);
        }
    }
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
    float sum;
    
    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE)
    {   
        int n;
        int num_cache_lines_64 = N/64;
        int N_aligned = N/64*64;
        int N_remaining = N - N_aligned;   
        
        sum = 0.0f;
        #pragma omp parallel for reduction(+:sum) private(n) shared(N, N_aligned, num_cache_lines_64)
        for (n = 0; n < num_cache_lines_64; n++)
        {
            #ifdef __MIC__
            _mm_prefetch((char*)(A + 64*n + 128), _MM_HINT_T0);
            float *A_pointer = A + 64*n;
            __m512 A_load_1 = _mm512_extload_ps(A_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
            __m512 A_load_2 = _mm512_extload_ps(A_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
            __m512 A_load_3 = _mm512_extload_ps(A_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
            __m512 A_load_4 = _mm512_extload_ps(A_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
            sum += _mm512_reduce_add_ps(A_load_1);
            sum += _mm512_reduce_add_ps(A_load_2);
            sum += _mm512_reduce_add_ps(A_load_3);
            sum += _mm512_reduce_add_ps(A_load_4);
            #endif
        }
        sum += __sec_reduce_add(A[N_aligned : N_remaining]);
        // float *restrict Y = ones_mic(N);
        // sum = cblas_sdot(N, A, 1, Y, 1);
        // _mm_free(Y);
    }
    return sum;
}

void horizontal_reflection(int N, int C, int H, int W, float *restrict INPUTS, int *restrict SHOULD_FLIP, float *restrict scratch){
    
    #pragma offload target(mic:MIC_DEV) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(SHOULD_FLIP:length(0) REUSE) \
    in(scratch:length(0) REUSE)
    {
        int n, nc, h, w;

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nc, h, w) \
            shared(INPUTS, SHOULD_FLIP, scratch, N, C, H, W)
        for (nc = 0; nc < N*C; nc++){
            int ncHW = nc*H*W;
            float *restrict inputs_pointer = INPUTS + ncHW - 1;
            float *restrict scratch_pointer = scratch + ncHW - 1;

            int should_flip_n = SHOULD_FLIP[nc/C];

            if (should_flip_n == 1){
                for (h = 0; h < H; h++){
                    for (w = 0; w < W; w++){
                            // scratch[ncHW + h*W + w] = INPUTS[ncHW + h*W + w];
                            *(++scratch_pointer) = *(++inputs_pointer);
                    }
                }

                inputs_pointer = INPUTS + ncHW - 1;
                for (h = 0; h < H; h++){
                    for (w = 0; w < W; w++){
                            // INPUTS[ncHW + h*W + w] = scratch[ncHW + h*W + ((W-1) - w)];
                        *(++inputs_pointer) = scratch[ncHW + h*W + ((W-1) - w)];
                    }
                }
            }

        }

    }
}

void vertical_reflection(int N, int C, int H, int W, float *restrict INPUTS, int *restrict SHOULD_FLIP, float *restrict scratch){
    
    #pragma offload target(mic:MIC_DEV) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(SHOULD_FLIP:length(0) REUSE) \
    in(scratch:length(0) REUSE)
    {
        int n, nc, h, w;

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nc, h, w) \
            shared(INPUTS, SHOULD_FLIP, scratch, N, C, H, W)
        for (nc = 0; nc < N*C; nc++){
            int ncHW = nc*H*W;
            float *restrict inputs_pointer = INPUTS + ncHW - 1;
            float *restrict scratch_pointer = scratch + ncHW - 1;

            int should_flip_n = SHOULD_FLIP[nc/C];

            if (should_flip_n == 1){
                for (h = 0; h < H; h++){
                    for (w = 0; w < W; w++){
                            // scratch[ncHW + h*W + w] = INPUTS[ncHW + h*W + w];
                            *(++scratch_pointer) = *(++inputs_pointer);
                    }
                }

                inputs_pointer = INPUTS + ncHW - 1;
                for (h = 0; h < H; h++){
                    for (w = 0; w < W; w++){
                            // INPUTS[ncHW + h*W + w] = scratch[ncHW + h*W + ((W-1) - w)];
                        // *(++inputs_pointer) = scratch[ncHW + h*W + ((W-1) - w)];
                        *(++inputs_pointer) = scratch[ncHW + ((H-1) - h)*W + w];
                    }
                }
            }

        }

    }
}

void crop(int N, int C, int H, int W, int output_W, int output_H, float *restrict INPUTS, int *restrict CROP_INFO, float *restrict OUTPUTS, int offloaded){
    // CROP_INFO is Nx2, and each row consists of left upper corner Y and X

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1)\ 
    in(INPUTS:length(0) REUSE) \ 
    in(CROP_INFO:length(0) REUSE) \
    in(OUTPUTS:length(0) REUSE)
    {
        int n, nc, h, w, h_start, w_start;

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(n, nc, h, w, h_start, w_start) \
            shared(INPUTS, CROP_INFO, OUTPUTS, N, C, H, W, output_H, output_W)
        for (nc = 0; nc < N*C; nc++){
            n = nc/C;
            h_start = CROP_INFO[2*n];
            w_start = CROP_INFO[2*n + 1];
            float *restrict outputs_pointer = OUTPUTS + nc*output_H*output_W - 1;
            float *restrict inputs_pointer = INPUTS + nc*H*W + h_start*W + w_start - 1;
            
            for (h = 0; h < output_H; h++){
                for (w = 0; w < output_W; w++){
                    *(++outputs_pointer) = *(++inputs_pointer);
                }
                inputs_pointer += -output_W + W;
            }
        }

    }
}

void rgb_to_hsv(int N, int H, int W, float *restrict INPUTS, float *restrict OUTPUTS, float *restrict SCRATCH, int offloaded){
    // CROP_INFO is Nx2, and each row consists of left upper corner Y and X

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1)\ 
    in(INPUTS:length(0) REUSE) \ 
    in(SCRATCH:length(0) REUSE) \
    in(OUTPUTS:length(0) REUSE)
    {
        int n, h, w;
        int C = 3;

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(n, h, w) \
            shared(INPUTS, SCRATCH, OUTPUTS, N, C, H, W)
        for (n = 0; n < N; n++){
            int L = H*W;
            float *restrict outputs_pointer = OUTPUTS + n*C*L;
            float *restrict inputs_pointer = INPUTS + n*C*L;

            float *restrict R = inputs_pointer;
            float *restrict G = inputs_pointer + L;
            float *restrict B = inputs_pointer + 2*L;

            float *restrict H = outputs_pointer;
            float *restrict S = outputs_pointer + L;
            float *restrict V = outputs_pointer + 2*L;

            float *restrict M = SCRATCH + n*C*L;
            float *restrict delta = SCRATCH + n*(C + 1)*L;

            V[0:L] = fmaxf(R[0:L], G[0:L]);
            V[0:L] = fmaxf(V[0:L], B[0:L]);
            

            M[0:L] = fminf(R[0:L], G[0:L]);
            M[0:L] = fminf(M[0:L], B[0:L]);
            // M[0:L] /= 255.0;
            
            delta[0:L] = V[0:L] - M[0:L];    
            S[0:L] = delta[0:L] / V[0:L];

            if (delta[0:L] == 0.f) H[0:L] = 0.f;
            else if (V[0:L] == R[0:L]) H[0:L] = (G[0:L] - B[0:L])/delta[0:L]; 
            else if (V[0:L] == G[0:L]) H[0:L] = 2.0 + 1.0 * (B[0:L] - R[0:L])/delta[0:L];
            else if (V[0:L] == B[0:L]) H[0:L] = 4.0 + 1.0 * (R[0:L] - G[0:L])/delta[0:L];            

            H[0:L] *= 60.0;
            if (H[0:L] < 0.0) H[0:L] += 360.0;
            
            if (V[0:L] == 0.f){
                S[0:L] = 0.f;
                H[0:L] = 0.f;
            }

            V[0:L] /= 255.0;

            // printf("H %f S %f V %f\n", __sec_reduce_add(H[0:L])/L, __sec_reduce_add(S[0:L])/L, __sec_reduce_add(V[0:L])/L);

                 
        }

    }
}

void hsv_to_rgb(int N, int H, int W, float *restrict INPUTS, float *restrict OUTPUTS, float *restrict SCRATCH, int offloaded){
    // CROP_INFO is Nx2, and each row consists of left upper corner Y and X

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1)\ 
    in(INPUTS:length(0) REUSE) \ 
    in(SCRATCH:length(0) REUSE) \
    in(OUTPUTS:length(0) REUSE)
    {
        int n, h, w;
        int C = 3;

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(n, h, w) \
            shared(INPUTS, SCRATCH, OUTPUTS, N, C, H, W)
        for (n = 0; n < N; n++){
            int L = H*W;
            float *restrict outputs_pointer = OUTPUTS + n*C*L;
            float *restrict inputs_pointer = INPUTS + n*C*L;

            float *restrict H = inputs_pointer;
            float *restrict S = inputs_pointer + L;
            float *restrict V = inputs_pointer + 2*L;

            float *restrict R = outputs_pointer;
            float *restrict G = outputs_pointer + L;
            float *restrict B = outputs_pointer + 2*L;

            float *restrict F = SCRATCH + ti3(n, 0, 0, C+1, L);
            float *restrict P = SCRATCH + ti3(n, 1, 0, C+1, L);
            float *restrict Q = SCRATCH + ti3(n, 2, 0, C+1, L);
            float *restrict T = SCRATCH + ti3(n, 3, 0, C+1, L);

            F[0:L] = H[0:L]/60.0 - floorf(H[0:L]/60.0);
            // problem is quite possibly with transition cases and F[]

            P[0:L] = V[0:L] * (1.0 - S[0:L]);
            Q[0:L] = V[0:L] * (1.0 - S[0:L]*F[0:L]);
            T[0:L] = V[0:L] * (1.0 - S[0:L]*(1.0 - F[0:L]));
            if (H[0:L] < 60.0){
                R[0:L] = V[0:L];
                G[0:L] = T[0:L];
                B[0:L] = P[0:L];
            }
            else if (H[0:L] >= 60.0 && H[0:L] < 120.0){
                R[0:L] = Q[0:L];
                G[0:L] = V[0:L];
                B[0:L] = P[0:L];
            }
            else if (H[0:L] >= 120.0 && H[0:L] < 180.0){
                R[0:L] = P[0:L];
                G[0:L] = V[0:L];
                B[0:L] = T[0:L];
            }
            else if (H[0:L] >= 180.0 && H[0:L] < 240.0){
                R[0:L] = P[0:L];
                G[0:L] = Q[0:L];
                B[0:L] = V[0:L];
            }
            else if (H[0:L] >= 240.0 && H[0:L] < 300.0){
                R[0:L] = T[0:L];
                G[0:L] = P[0:L];
                B[0:L] = V[0:L];
            }
            else if (H[0:L] >= 300.0){
                R[0:L] = V[0:L];
                G[0:L] = P[0:L];
                B[0:L] = Q[0:L];
            }

            if (S[0:L] == 0.f){
                R[0:L] = V[0:L];
                G[0:L] = V[0:L];
                B[0:L] = V[0:L];
            }

            R[0:L] *= 255.0;
            G[0:L] *= 255.0;
            B[0:L] *= 255.0;       
        }    
    }
}

void perturb_hue(int N, int C, int H, int W, float *restrict INPUTS, float *restrict PERTURBATIONS, int offloaded){
    #pragma offload target(mic:MIC_DEV) if(offloaded == 1)\ 
        in(INPUTS:length(0) REUSE) \ 
        in(PERTURBATIONS:length(0) REUSE)
    {
        int n;

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(n) \
            shared(INPUTS, PERTURBATIONS, N, C, H, W)
        for (n = 0; n < N; n++){
            int L = H*W;
            float *restrict inputs_pointer = INPUTS + n*C*L;

            float *restrict H = inputs_pointer;

            H[0:L] += PERTURBATIONS[n];
            if (H[0:L] > 360.0) H[0:L] -= 360.0;
            if (H[0:L] < 0.0) H[0:L] += 360.0;
        }
    }
}

void perturb_saturation(int N, int C, int H, int W, float *restrict INPUTS, float *restrict SCALE_PERTURBATIONS, float *restrict SHIFT_PERTURBATIONS, int offloaded){
    #pragma offload target(mic:MIC_DEV) if(offloaded == 1)\ 
        in(INPUTS:length(0) REUSE) \ 
        in(SCALE_PERTURBATIONS:length(0) REUSE) \
        in(SHIFT_PERTURBATIONS:length(0) REUSE)
    {
        int n;

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(n) \
            shared(INPUTS, SCALE_PERTURBATIONS, SHIFT_PERTURBATIONS, N, C, H, W)
        for (n = 0; n < N; n++){
            int L = H*W;
            float *restrict inputs_pointer = INPUTS + n*C*L;
            float *restrict S = inputs_pointer + L;

            S[0:L] *= SCALE_PERTURBATIONS[n];
            S[0:L] += SHIFT_PERTURBATIONS[n];
            if (S[0:L] > 1.0) S[0:L] = 1.0;
            if (S[0:L] < 0.0) S[0:L] = 0.0;
        }
    }
}

void perturb_value(int N, int C, int H, int W, float *restrict INPUTS, float *restrict SCALE_PERTURBATIONS, float *restrict SHIFT_PERTURBATIONS, int offloaded){
    #pragma offload target(mic:MIC_DEV) if(offloaded == 1)\ 
        in(INPUTS:length(0) REUSE) \ 
        in(SCALE_PERTURBATIONS:length(0) REUSE) \
        in(SHIFT_PERTURBATIONS:length(0) REUSE)
    {
        int n;

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(n) \
            shared(INPUTS, SCALE_PERTURBATIONS, SHIFT_PERTURBATIONS, N, C, H, W)
        for (n = 0; n < N; n++){
            int L = H*W;
            float *restrict inputs_pointer = INPUTS + n*C*L;
            float *restrict V = inputs_pointer + (C-1)*L;

            V[0:L] *= SCALE_PERTURBATIONS[n];
            V[0:L] += SHIFT_PERTURBATIONS[n];
            if (V[0:L] > 1.0) V[0:L] = 1.0;
            if (V[0:L] < 0.0) V[0:L] = 0.0;
        }
    }
}

// void perturb_contrast(int N, int H, int W, float *restrict INPUTS, float *restrict PERTURBATIONS, int offloaded){
//     #pragma offload target(mic:MIC_DEV) if(offloaded == 1)\ 
//         in(INPUTS:length(0) REUSE) \ 
//         in(PERTURBATIONS:length(0) REUSE)
//     {
//         int n;
//         int C = 3;

//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(n) \
//             shared(INPUTS, PERTURBATIONS, N, C, H, W)
//         for (n = 0; n < N; n++){
//             int L = H*W;
//             float *restrict inputs_pointer = INPUTS + n*C*L;
//             float *restrict V = inputs_pointer + 2*L;

//             V[0:L] += PERTURBATIONS[n];
//             if (V[0:L] > 1.0) V[0:L] = 1.0;
//             if (V[0:L] < 0.0) V[0:L] = 0.0;
//         }
//     }
// }

void response_normalization(int N, int K, int H, int W, float *restrict INPUTS, float *restrict OUTPUTS, float ALPHA, float BETA, int LOCAL_RADIUS){

    #pragma offload target(mic:MIC_DEV) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE)
    // in(GROUPS: length(0) REUSE)
    {
        int nk, n, k, h, w;
        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nk, n, k, h, w) \
            shared(N, K, H, W, INPUTS, OUTPUTS, ALPHA, BETA, LOCAL_RADIUS)
            // shared(N, K, H, W, INPUTS, OUTPUTS, GROUPS, ALPHA, BETA, LOCAL_RADIUS)
        for (nk = 0; nk < N*K; nk++){
                n = nk/K;
                k = md(nk, K);

                for (h = 0; h < H; h++){
                    for (w = 0; w < W; w++){
                        float sum = 0.f;

                        // if (LOCAL_RADIUS > 0){
                        for (int l = mx(0, k - LOCAL_RADIUS); l < mn(K, k + LOCAL_RADIUS); l++){
                            sum += pow(INPUTS[ti(n, l, h, w, K, H, W)], 2.f);
                        }
                        // }
                        // else {
                        //     for (int l = 0; l < K; l++){
                        //         if (GROUPS[n*K + l] == 1) sum += pow(INPUTS[ti(n, l, h, w, K, H, W)], 2.f);
                        //     }
                        // }

                        sum *= ALPHA/(2.f*((float) LOCAL_RADIUS) + 1.f);
                        sum += 1.f;
                        sum = pow(sum, -BETA);

                        OUTPUTS[ti(n, k, h, w, K, H, W)] = INPUTS[ti(n, k, h, w, K, H, W)] * sum;
                    }
                }
        }

    }

}

void response_normalization_gradient(int N, int K, int H, int W, float *restrict INPUTS, float *restrict OUTPUTS, float *restrict D_INPUTS, float *restrict D_OUTPUTS, float ALPHA, float BETA, int LOCAL_RADIUS){

    #pragma offload target(mic:MIC_DEV) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE) \
    in(D_INPUTS:length(0) REUSE) \ 
    in(D_OUTPUTS:length(0) REUSE)
    // in(GROUPS: length(0) REUSE)
    {
        int nk, n, k, h, w;
        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nk, n, k, h, w) \
            shared(N, K, H, W, INPUTS, OUTPUTS, D_INPUTS, D_OUTPUTS, ALPHA, BETA, LOCAL_RADIUS)
            // shared(N, K, H, W, INPUTS, OUTPUTS, GROUPS, ALPHA, BETA, LOCAL_RADIUS)
        for (nk = 0; nk < N*K; nk++){
                n = nk/K;
                k = md(nk, K);

                for (h = 0; h < H; h++){
                    for (w = 0; w < W; w++){
                        int nkhw = ti(n, k, h, w, K, H, W);

                        float sum = 0.f;
                        for (int l = mx(0, k - LOCAL_RADIUS); l < mn(K, k + LOCAL_RADIUS); l++){
                            sum += pow(INPUTS[ti(n, l, h, w, K, H, W)], 2.f);
                        }
                        sum *= ALPHA/(2.f*((float) LOCAL_RADIUS) + 1.f);
                        sum += 1.f;
                        sum = pow(sum, -BETA);
                        float value = D_OUTPUTS[ti(n, k, h, w, K, H, W)] * sum;


                        for (int l = mx(0, k - LOCAL_RADIUS); l < mn(K, k + LOCAL_RADIUS); l++){
                            int nlhw = ti(n, l, h, w, K, H, W);
                            float inputs_nlhw = INPUTS[nlhw];

                            if ((inputs_nlhw > 1.0e-5) || (inputs_nlhw < -1.0e-5))
                                value -= 2.f*ALPHA*BETA/(2.f*((float) LOCAL_RADIUS) + 1.f) 
                                        * pow(inputs_nlhw, 2.f)
                                        * pow(OUTPUTS[nlhw]/inputs_nlhw, (BETA + 1.f)/BETA)
                                        * D_OUTPUTS[nlhw];
                            // value -= 2.f*ALPHA*BETA/(2.f*((float) LOCAL_RADIUS) + 1.f) 
                            //         * pow(inputs_nlhw, 1.f - 1.f/BETA)       // * pow(inputs_nlhw, 2.f)
                            //         * pow(OUTPUTS[nlhw], (BETA + 1.f)/BETA)       // * pow(OUTPUTS[nlhw]/inputs_nlhw, (BETA + 1.f)/BETA)
                            //         * D_OUTPUTS[nlhw];

                            // float sum = 0.f;
                            // for (int i = mx(0, l - LOCAL_RADIUS); i < mn(K, l + LOCAL_RADIUS); i++){
                            //     sum += pow(INPUTS[ti(n, i, h, w, K, H, W)], 2.f);
                            // }
                            
                            // sum *= ALPHA/((float) N);
                            // sum += 1.f;

                            // // if (l == k) value += D_OUTPUTS[ti(n, l, h, w, K, H, W)] / pow(sum, BETA);

                            // value -= 2.f*ALPHA*BETA/((float) N) * D_OUTPUTS[ti(n, l, h, w, K, H, W)] * pow(INPUTS[ti(n, l, h, w, K, H, W)], 2.f) / pow(sum, BETA + 1.f);
                        }

                        D_INPUTS[nkhw] = value;
                    }
                }
        }

    }

}



// int *convolve_and_pool_layer1(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded){

//     assert(C == C_const);
//     assert(H == H_const);
//     assert(W == W_const);
//     assert(K == K_const);
//     assert(stride == stride_const);
//     assert(padding == padding_const);
//     assert(pooling_radius == pooling_radius_const);
//     assert(pooling_stride == pooling_stride_const);
//     assert(X == X_const);
//     assert(Y == Y_const);
//     assert(output_H_const == (H_const + 2*padding_const - Y_const + 1)/stride_const);
//     assert(output_W_const == (W_const + 2*padding_const - X_const + 1)/stride_const);
//     assert(pooled_H_const == (output_H_const - pooling_radius_const + 1)/pooling_stride_const);
//     assert(pooled_W_const == (output_W_const - pooling_radius_const + 1)/pooling_stride_const);

//     #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
//     in(INPUTS:length(0) REUSE) \ 
//     in(FILTERS:length(0) REUSE) \ 
//     in(OUTPUTS:length(0) REUSE) \
//     in(ARGMAXS:length(0) REUSE)
//     {
//         float convolution = 0.f;
//         int nkij, nkhw, n, k, i, j, h, w, c, y, x;
//         int nk, ij;

//         // computation of constants
//         int XW = -X_const + W_const,
//                     HYW = (H_const-Y_const)*W_const;

//         // #pragma omp parallel for \
//             // schedule(dynamic) \
//             // private(nkij)
//         // for (nkij = 0; nkij < N*K_const*pooled_H_const*pooled_W_const; nkij++) OUTPUTS[nkij] = -1.0e10;

//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(nkhw, nkij, nk, ij, n, k, h, w, c, y, x, i, j, convolution) \
//             shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, XW, HYW)
        
//         // for each example, for each filter, for each pooling region...
//         for (nk = 0; nk < N*K_const; nk++){
//             float *restrict outputs_pointer = OUTPUTS + nk*pooled_H_const*pooled_W_const - 1;
//             for (ij = 0; ij < pooled_H_const*pooled_W_const; ij++) *(++outputs_pointer) = -1.0e10;

//             n = nk / K_const;
//             k = md(nk, K_const);
            
//             int hw = 0,
//                 // nkhw = nk*output_H_const*output_W_const, // NOTE: not applicable for stride!
//                 kcyx_shift = k*C_const*Y_const*X_const - 1, // filters
//                 nchw_shift = (n*C_const*H_const)*W_const - 1; // inputs

//             for (h = 0; h < output_H_const; h+= stride_const){
//                 for (w = 0; w < output_W_const; w+= stride_const){
//                     convolution = 0.f;
//                     float *restrict filters_pointer = FILTERS + kcyx_shift;
//                     float *restrict inputs_pointer = INPUTS + nchw_shift + h*W_const + w;
                           
//                     // loop over convolution elements
//                     for (c = 0; c < C_const; ++c){
//                         for (y = 0; y < Y_const; ++y){
//                             for (x = 0; x < X_const; ++x){
//                                 convolution += (*(++inputs_pointer)) * (*(++filters_pointer));
//                             }

//                             inputs_pointer += XW;
//                         }

//                         inputs_pointer += HYW;
//                     }
                    
//                     // loop over pooled outputs that care about this particular (pre-pooled) output element
//                     // update max, argmax
//                     int i_start = (h + pooling_stride_const - pooling_radius_const)/pooling_stride_const; // ceil((h - pooling_radius + 1)/pooling_stride)
//                     i_start = (i_start < 0) ? 0 : i_start;
//                     int j_start = (w + pooling_stride_const - pooling_radius_const)/pooling_stride_const;
//                     j_start = (j_start < 0) ? 0 : j_start;
//                     int i_end = h/pooling_stride_const; // floor(h/pooling_stride_const)
//                     i_end = (i_end >= pooled_H_const) ? (pooled_H_const - 1) : i_end;
//                     int j_end = w/pooling_stride_const;
//                     j_end = (j_end >= pooled_W_const) ? (pooled_W_const - 1) : j_end;

//                     outputs_pointer = OUTPUTS + (nk*pooled_H_const + i_start)*pooled_W_const + j_start;
//                     int *restrict argmaxs_pointer = ARGMAXS + (nk*pooled_H_const + i_start)*pooled_W_const + j_start - 1;
//                     // printf("test %d, test %d, h %d w %d, start %d end %d start %d end %d and finally %d \n", ((int) -1)/((int) ), pooling_stride_const, h, w, i_start, i_end, j_start, j_end, (i_end-i_start)*(j_end-j_start));
//                     for (i = i_start; i <= i_end; ++i){
//                         for (j = j_start; j <= j_end; ++j){
//                             // nkij = (nk*pooled_H_const + i)*pooled_W_const + j;
//                             if (convolution > *outputs_pointer){
//                                 *(outputs_pointer++) = convolution;
//                                 *(++argmaxs_pointer) = hw;
//                             }
//                         } // j
//                         outputs_pointer += -(j_end - j_start) + pooled_W_const;
//                         argmaxs_pointer += -(j_end - j_start) + pooled_W_const;
//                     } // i

//                     hw++;
//                 } // w
//             } // h

//         } // nk
         
//     } // pragma_offload

//     return ARGMAXS;
// }

int *convolve_and_pool_layer1(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){

    assert(C == C_const);
    assert(H == H_const);
    assert(W == W_const);
    assert(K == K_const);
    assert(stride == stride_const);
    assert(padding == padding_const);
    assert(pooling_radius == pooling_radius_const);
    assert(pooling_stride == pooling_stride_const);
    assert(X == X_const);
    assert(Y == Y_const);
    assert(output_H_const == (H_const + 2*padding_const - Y_const + 1)/stride_const);
    assert(output_W_const == (W_const + 2*padding_const - X_const + 1)/stride_const);
    assert(pooled_H_const == (output_H_const - pooling_radius_const + 1)/pooling_stride_const);
    assert(pooled_W_const == (output_W_const - pooling_radius_const + 1)/pooling_stride_const);

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE) \
    in(ARGMAXS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {
        float convolution = 0.f;
        int nkij, nkhw, n, k, i, j, h, w, c, y, x;
        int nk, ij;

        // computation of constants
        int XW = -X_const + W_const,
                    HYW = (H_const-Y_const)*W_const;

        // pre-compute indices over channels, width and height of elements to be summed in convolution
        int *restrict indices = _mm_malloc(C_const*Y_const*X_const*sizeof(int), ALIGN);
        int *restrict indices_pointer = indices - 1;
        int index = -1;
        for (c = 0; c < C_const; ++c){
            for (y = 0; y < Y_const; ++y){                                
                for (x = 0; x < X_const; ++x){
                    *(++indices_pointer) = ++index;
                    } // x
                    index += XW;
            } // y
            index += HYW;
        } // c
        
        
        float *restrict ind_pointer = SCRATCH - 1;
        #pragma omp parallel for \
            schedule(dynamic, output_W_const) \
            private(h, w) \
            shared(SCRATCH, ind_pointer)
        for (h = 0; h < output_H_const; h++){
            for (w = 0; w < output_W_const; w++){
                // pointer goes i_start, i_end, j_start, j_end
                int i_start = (h + pooling_stride_const - pooling_radius_const)/pooling_stride_const; // ceil((h - pooling_radius + 1)/pooling_stride)
                *(++ind_pointer) = (float) (i_start < 0) ? 0 : i_start;

                int i_end = h/pooling_stride_const; // floor(h/pooling_stride_const)
                *(++ind_pointer) = (float) (i_end >= pooled_H_const) ? (pooled_H_const - 1) : i_end;

                int j_start = (w + pooling_stride_const - pooling_radius_const)/pooling_stride_const;
                *(++ind_pointer) = (float) (j_start < 0) ? 0 : j_start;               
                
                int j_end = w/pooling_stride_const;
                *(++ind_pointer) = (float) (j_end >= pooled_W_const) ? (pooled_W_const - 1) : j_end;
            }
        }

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nkhw, nkij, nk, ij, n, k, h, w, c, y, x, i, j, convolution) \
            shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XW, HYW, indices)
        
        // for each example, for each filter, for each pooling region...
        for (nk = 0; nk < N*K_const; nk++){
            float *restrict outputs_pointer = OUTPUTS + nk*pooled_H_const*pooled_W_const - 1;
            for (ij = 0; ij < pooled_H_const*pooled_W_const; ij++) *(++outputs_pointer) = -1.0e10;

            n = nk / K_const;
            k = md(nk, K_const);
            
            int hw = 0,
                // nkhw = nk*output_H_const*output_W_const, // NOTE: not applicable for stride!
                // kcyx_shift = k*C_const*Y_const*X_const - 1, // filters
                // nchw_shift = (n*C_const*H_const)*W_const - 1; // inputs
                kcyx_shift = k*C_const*Y_const*X_const, // filters
                nchw_shift = (n*C_const*H_const)*W_const; // inputs

            // float *outputs_tmp = SCRATCH + 2*omp_get_thread_num()*pooled_H_const*pooled_W_const - 1;
            // for (h = 0; h < pooled_H_const; h++){
            //     for (w = 0; w < pooled_W_const; w++){
            //         *(++outputs_tmp) = -1.0e10;
            //     } // w
            // } // h
            // outputs_tmp = SCRATCH + 2*omp_get_thread_num()*pooled_H_const*pooled_W_const;
            // float *argmaxs_tmp = SCRATCH + (240 + 2*omp_get_thread_num())*pooled_H_const*pooled_W_const;
            float *restrict ind_pointer = SCRATCH - 1;

            for (h = 0; h < output_H_const; h+= stride_const){
                for (w = 0; w < output_W_const; w+= stride_const){
                    convolution = 0.f;
                    float *restrict filters_pointer = FILTERS + kcyx_shift;
                   
                    // if we're not on boundary (i.e not affected by padding)
                    if (w - padding_const >= 0 &&
                        h - padding_const >= 0 &&
                        output_W_const - 1 - w >= padding_const  &&
                        output_H_const - 1 - h >= padding_const){

                        float *restrict inputs_pointer = INPUTS + nchw_shift + (h - padding_const)*W_const + (w - padding_const);
                        
                        convolution = __sec_reduce_add(inputs_pointer[indices[0:C_const*Y_const*X_const]] * filters_pointer[0:C_const*Y_const*X_const]);
                        filters_pointer += C_const*Y_const*X_const;

                        // The way I used to do this is with the below loop

                        // for (c = 0; c < C_const; ++c){
                        //     for (y = 0; y < Y_const; ++y){                                
                        //         for (x = 0; x < X_const; ++x){
                        //             convolution += (*(++inputs_pointer)) * (*(++filters_pointer));
                        //         } // x

                        //         inputs_pointer += XW;
                        //     } // y
                        //     inputs_pointer += HYW;
                        // } // c

                    }

                    else{
                        float *restrict inputs_pointer = INPUTS + nchw_shift + mx(mn(h-padding_const, H_const-1), 0)*W_const + mx(mn(w-padding_const, W_const-1), 0);
    
                        for (c = 0; c < C_const; ++c){
                            float *restrict inputs_pointer_ncYX = inputs_pointer;
                            
                            for (y = 0; y < Y_const; ++y){
                                
                                float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                                
                                if ((y + h - padding_const >= 0) && (y + h - padding_const < H_const)){
                                    for (x = 0; x < X_const; ++x){
                                        if ((x + w - padding_const >= 0) && (x + w - padding_const < W_const))
                                            convolution += (*(inputs_pointer++)) * (*filters_pointer);
                                        filters_pointer++;
                                    } // x
                                }

                                else{ // advance pointer without going into loop
                                    // inputs_pointer += X_const;
                                    filters_pointer += X_const;
                                } 

                                inputs_pointer = inputs_pointer_ncyX + W_const; 
                            } // y
    
                            inputs_pointer = inputs_pointer_ncYX + H_const*W_const; 
                        } // c
                    }
                    
                    // loop over pooled outputs that care about this particular (pre-pooled) output element
                    // update max, argmax
                    int i_start = (int) *(++ind_pointer);
                    int i_end = (int) *(++ind_pointer);
                    int j_start = (int) *(++ind_pointer);
                    int j_end = (int) *(++ind_pointer);
                    // int i_start = (h + pooling_stride_const - pooling_radius_const)/pooling_stride_const; // ceil((h - pooling_radius + 1)/pooling_stride)
                    // i_start = (i_start < 0) ? 0 : i_start;
                    // int j_start = (w + pooling_stride_const - pooling_radius_const)/pooling_stride_const;
                    // j_start = (j_start < 0) ? 0 : j_start;
                    // int i_end = h/pooling_stride_const; // floor(h/pooling_stride_const)
                    // i_end = (i_end >= pooled_H_const) ? (pooled_H_const - 1) : i_end;
                    // int j_end = w/pooling_stride_const;
                    // j_end = (j_end >= pooled_W_const) ? (pooled_W_const - 1) : j_end;

                    outputs_pointer = OUTPUTS + (nk*pooled_H_const + i_start)*pooled_W_const + j_start;
                    int *restrict argmaxs_pointer = ARGMAXS + (nk*pooled_H_const + i_start)*pooled_W_const + j_start - 1;
                    // float *restrict outputs_pointer = outputs_tmp + i_start*pooled_W_const + j_start;
                    // float *restrict argmaxs_pointer = argmaxs_tmp + i_start*pooled_W_const + j_start - 1;
                    for (i = i_start; i <= i_end; ++i){
                        for (j = j_start; j <= j_end; ++j){
                            // nkij = (nk*pooled_H_const + i)*pooled_W_const + j;
                            if (convolution > *outputs_pointer){
                                *(outputs_pointer++) = convolution;
                                *(++argmaxs_pointer) = hw;
                                // *(++argmaxs_pointer) = (float) hw;
                            }
                        } // j
                        outputs_pointer += -(j_end - j_start) + pooled_W_const;
                        argmaxs_pointer += -(j_end - j_start) + pooled_W_const;
                    } // i

                    hw++;
                } // w
            } // h

            // float *restrict outputs_pointer = OUTPUTS + nk*pooled_H_const*pooled_W_const - 1;
            // int *restrict argmaxs_pointer = ARGMAXS + nk*pooled_H_const*pooled_W_const - 1;
            // for (h = 0; h < pooled_H_const; h++){
            //     for (w = 0; w < pooled_W_const; w++){
            //         *(++outputs_pointer) = *(++outputs_tmp);
            //         *(++argmaxs_pointer) = (int) *(++argmaxs_tmp);
            //     } // w
            // } // h

        } // nk
         
    } // pragma_offload

    return ARGMAXS;
}

int *convolve_and_pool_shadow_layer1(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){

    assert(C == C_const);
    assert(H == H_const);
    assert(W == W_const);
    assert(K == K_const);
    assert(stride == stride_const);
    assert(padding == padding_const);
    assert(pooling_radius == pooling_radius_const);
    assert(pooling_stride == pooling_stride_const);
    assert(X == X_const);
    assert(Y == Y_const);
    assert(output_H_const == (H_const + 2*padding_const - Y_const + 1)/stride_const);
    assert(output_W_const == (W_const + 2*padding_const - X_const + 1)/stride_const);
    assert(pooled_H_const == (output_H_const - pooling_radius_const + 1)/pooling_stride_const);
    assert(pooled_W_const == (output_W_const - pooling_radius_const + 1)/pooling_stride_const);

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE) \
    in(ARGMAXS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {
        float convolution = 0.f;
        int nkij, nkhw, n, k, i, j, h, w, c, y, x;
        int nk, ij;

        // computation of constants
        int XW = -X_const + W_const,
                    HYW = (H_const-Y_const)*W_const;

        // pre-compute indices over channels, width and height of elements to be summed in convolution
        int *restrict indices = _mm_malloc(C_const*Y_const*X_const*sizeof(int), ALIGN);
        int *restrict indices_pointer = indices - 1;
        int index = -1;
        for (c = 0; c < C_const; ++c){
            for (y = 0; y < Y_const; ++y){                                
                for (x = 0; x < X_const; ++x){
                    *(++indices_pointer) = ++index;
                    } // x
                    index += XW;
            } // y
            index += HYW;
        } // c
        
        
        float *restrict ind_pointer = SCRATCH - 1;
        #pragma omp parallel for \
            schedule(dynamic, output_W_const) \
            private(h, w) \
            shared(SCRATCH, ind_pointer)
        for (h = 0; h < output_H_const; h++){
            for (w = 0; w < output_W_const; w++){
                // pointer goes i_start, i_end, j_start, j_end
                int i_start = (h + pooling_stride_const - pooling_radius_const)/pooling_stride_const; // ceil((h - pooling_radius + 1)/pooling_stride)
                *(++ind_pointer) = (float) (i_start < 0) ? 0 : i_start;

                int i_end = h/pooling_stride_const; // floor(h/pooling_stride_const)
                *(++ind_pointer) = (float) (i_end >= pooled_H_const) ? (pooled_H_const - 1) : i_end;

                int j_start = (w + pooling_stride_const - pooling_radius_const)/pooling_stride_const;
                *(++ind_pointer) = (float) (j_start < 0) ? 0 : j_start;               
                
                int j_end = w/pooling_stride_const;
                *(++ind_pointer) = (float) (j_end >= pooled_W_const) ? (pooled_W_const - 1) : j_end;
            }
        }

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nkhw, nkij, nk, ij, n, k, h, w, c, y, x, i, j, convolution) \
            shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XW, HYW, indices)
        
        // for each example, for each filter, for each pooling region...
        for (nk = 0; nk < N*K_const; nk++){
            float *restrict outputs_pointer = OUTPUTS + nk*pooled_H_const*pooled_W_const - 1;
            for (ij = 0; ij < pooled_H_const*pooled_W_const; ij++) *(++outputs_pointer) = -1.0e10;

            outputs_pointer = OUTPUTS + N*K_const*pooled_H_const*pooled_W_const + nk*pooled_H_const*pooled_W_const - 1;
            for (ij = 0; ij < pooled_H_const*pooled_W_const; ij++) *(++outputs_pointer) = -1.0e10;

            n = nk / K_const;
            k = md(nk, K_const);
            
            int hw = 0,
                // nkhw = nk*output_H_const*output_W_const, // NOTE: not applicable for stride!
                // kcyx_shift = k*C_const*Y_const*X_const - 1, // filters
                // nchw_shift = (n*C_const*H_const)*W_const - 1; // inputs
                kcyx_shift = k*C_const*Y_const*X_const, // filters
                nchw_shift = (n*C_const*H_const)*W_const; // inputs

            // float *outputs_tmp = SCRATCH + 2*omp_get_thread_num()*pooled_H_const*pooled_W_const - 1;
            // for (h = 0; h < pooled_H_const; h++){
            //     for (w = 0; w < pooled_W_const; w++){
            //         *(++outputs_tmp) = -1.0e10;
            //     } // w
            // } // h
            // outputs_tmp = SCRATCH + 2*omp_get_thread_num()*pooled_H_const*pooled_W_const;
            // float *argmaxs_tmp = SCRATCH + (240 + 2*omp_get_thread_num())*pooled_H_const*pooled_W_const;
            float *restrict ind_pointer = SCRATCH - 1;

            for (h = 0; h < output_H_const; h+= stride_const){
                for (w = 0; w < output_W_const; w+= stride_const){
                    convolution = 0.f;
                    float *restrict filters_pointer = FILTERS + kcyx_shift;
                   
                    // if we're not on boundary (i.e not affected by padding)
                    if (w - padding_const >= 0 &&
                        h - padding_const >= 0 &&
                        output_W_const - 1 - w >= padding_const  &&
                        output_H_const - 1 - h >= padding_const){

                        float *restrict inputs_pointer = INPUTS + nchw_shift + (h - padding_const)*W_const + (w - padding_const);
                        
                        convolution = __sec_reduce_add(inputs_pointer[indices[0:C_const*Y_const*X_const]] * filters_pointer[0:C_const*Y_const*X_const]);
                        filters_pointer += C_const*Y_const*X_const;

                    }

                    else{
                        float *restrict inputs_pointer = INPUTS + nchw_shift + mx(mn(h-padding_const, H_const-1), 0)*W_const + mx(mn(w-padding_const, W_const-1), 0);
    
                        for (c = 0; c < C_const; ++c){
                            float *restrict inputs_pointer_ncYX = inputs_pointer;
                            
                            for (y = 0; y < Y_const; ++y){
                                
                                float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                                
                                if ((y + h - padding_const >= 0) && (y + h - padding_const < H_const)){
                                    for (x = 0; x < X_const; ++x){
                                        if ((x + w - padding_const >= 0) && (x + w - padding_const < W_const))
                                            convolution += (*(inputs_pointer++)) * (*filters_pointer);
                                        filters_pointer++;
                                    } // x
                                }

                                else{ // advance pointer without going into loop
                                    // inputs_pointer += X_const;
                                    filters_pointer += X_const;
                                } 

                                inputs_pointer = inputs_pointer_ncyX + W_const; 
                            } // y
    
                            inputs_pointer = inputs_pointer_ncYX + H_const*W_const; 
                        } // c
                    }
                    
                    // loop over pooled outputs that care about this particular (pre-pooled) output element
                    // update max, argmax
                    int i_start = (int) *(++ind_pointer);
                    int i_end = (int) *(++ind_pointer);
                    int j_start = (int) *(++ind_pointer);
                    int j_end = (int) *(++ind_pointer);
                    // int i_start = (h + pooling_stride_const - pooling_radius_const)/pooling_stride_const; // ceil((h - pooling_radius + 1)/pooling_stride)
                    // i_start = (i_start < 0) ? 0 : i_start;
                    // int j_start = (w + pooling_stride_const - pooling_radius_const)/pooling_stride_const;
                    // j_start = (j_start < 0) ? 0 : j_start;
                    // int i_end = h/pooling_stride_const; // floor(h/pooling_stride_const)
                    // i_end = (i_end >= pooled_H_const) ? (pooled_H_const - 1) : i_end;
                    // int j_end = w/pooling_stride_const;
                    // j_end = (j_end >= pooled_W_const) ? (pooled_W_const - 1) : j_end;

                    outputs_pointer = OUTPUTS + (nk*pooled_H_const + i_start)*pooled_W_const + j_start;
                    int *restrict argmaxs_pointer = ARGMAXS + (nk*pooled_H_const + i_start)*pooled_W_const + j_start - 1;
                    // float *restrict outputs_pointer = outputs_tmp + i_start*pooled_W_const + j_start;
                    // float *restrict argmaxs_pointer = argmaxs_tmp + i_start*pooled_W_const + j_start - 1;
                    for (i = i_start; i <= i_end; ++i){
                        for (j = j_start; j <= j_end; ++j){
                            // nkij = (nk*pooled_H_const + i)*pooled_W_const + j;
                            if (convolution > *outputs_pointer){
                                *(outputs_pointer++) = convolution;
                                *(++argmaxs_pointer) = hw;
                                // *(++argmaxs_pointer) = (float) hw;
                            }
                        } // j
                        outputs_pointer += -(j_end - j_start) + pooled_W_const;
                        argmaxs_pointer += -(j_end - j_start) + pooled_W_const;
                    } // i

                    // shadow filters
                    convolution = -convolution;
                    outputs_pointer = OUTPUTS + N*K_const*pooled_H_const*pooled_W_const + (nk*pooled_H_const + i_start)*pooled_W_const + j_start;
                    argmaxs_pointer = ARGMAXS + N*K_const*pooled_H_const*pooled_W_const + (nk*pooled_H_const + i_start)*pooled_W_const + j_start - 1;
                    // float *restrict outputs_pointer = outputs_tmp + i_start*pooled_W_const + j_start;
                    // float *restrict argmaxs_pointer = argmaxs_tmp + i_start*pooled_W_const + j_start - 1;
                    for (i = i_start; i <= i_end; ++i){
                        for (j = j_start; j <= j_end; ++j){
                            // nkij = (nk*pooled_H_const + i)*pooled_W_const + j;
                            if (convolution > *outputs_pointer){
                                *(outputs_pointer++) = convolution;
                                *(++argmaxs_pointer) = hw;
                                // *(++argmaxs_pointer) = (float) hw;
                            }
                        } // j
                        outputs_pointer += -(j_end - j_start) + pooled_W_const;
                        argmaxs_pointer += -(j_end - j_start) + pooled_W_const;
                    } // i

                    hw++;
                } // w
            } // h

            // float *restrict outputs_pointer = OUTPUTS + nk*pooled_H_const*pooled_W_const - 1;
            // int *restrict argmaxs_pointer = ARGMAXS + nk*pooled_H_const*pooled_W_const - 1;
            // for (h = 0; h < pooled_H_const; h++){
            //     for (w = 0; w < pooled_W_const; w++){
            //         *(++outputs_pointer) = *(++outputs_tmp);
            //         *(++argmaxs_pointer) = (int) *(++argmaxs_tmp);
            //     } // w
            // } // h

        } // nk
    } // pragma_offload

    return ARGMAXS;
}

// int *convolve_and_pool_layer1(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){

//     assert(C == C_const);
//     assert(H == H_const);
//     assert(W == W_const);
//     assert(K == K_const);
//     assert(stride == stride_const);
//     assert(padding == padding_const);
//     assert(pooling_radius == pooling_radius_const);
//     assert(pooling_stride == pooling_stride_const);
//     assert(X == X_const);
//     assert(Y == Y_const);
//     assert(output_H_const == (H_const + 2*padding_const - Y_const + 1)/stride_const);
//     assert(output_W_const == (W_const + 2*padding_const - X_const + 1)/stride_const);
//     assert(pooled_H_const == (output_H_const - pooling_radius_const + 1)/pooling_stride_const);
//     assert(pooled_W_const == (output_W_const - pooling_radius_const + 1)/pooling_stride_const);

//     #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
//     in(INPUTS:length(0) REUSE) \ 
//     in(FILTERS:length(0) REUSE) \ 
//     in(OUTPUTS:length(0) REUSE) \
//     in(ARGMAXS:length(0) REUSE) \
//     in(SCRATCH:length(0) REUSE)
//     {
//         float convolution = 0.f;
//         int nkij, nkhw, n, k, i, j, h, w, c, y, x;
//         int nk, ij;

//         // computation of constants
//         int XW = -X_const + W_const,
//                     HYW = (H_const-Y_const)*W_const;

//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(nkhw, nkij, nk, ij, n, k, h, w, c, y, x, i, j, convolution) \
//             shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XW, HYW)
        
//         // for each example, for each filter
//         for (nk = 0; nk < N*K_const; nk++){
//             float *restrict outputs_pointer = OUTPUTS + nk*pooled_H_const*pooled_W_const - 1;
//             for (ij = 0; ij < pooled_H_const*pooled_W_const; ij++) *(++outputs_pointer) = -1.0e10;

//             n = nk / K_const;
//             k = md(nk, K_const);
            
//             int hw = 0,
//                 kcyx_shift = k*C_const*Y_const*X_const - 1, // filters
//                 nchw_shift = (n*C_const*H_const)*W_const - 1; // inputs

//             // scanning over a particular image, for a particular filter
//             for (h = 0; h < output_H_const; h+= stride_const){
//                 for (w = 0; w < output_W_const; w+= stride_const){

//                     // ------------------ CONVOLUTION ------------------ 
//                     convolution = 0.f;
//                     float *restrict filters_pointer = FILTERS + kcyx_shift;
                   
//                     // if we're not on boundary (i.e not affected by padding)
//                     if (w - padding_const >= 0 &&
//                         h - padding_const >= 0 &&
//                         output_W_const - 1 - w >= padding_const  &&
//                         output_H_const - 1 - h >= padding_const){

//                         float *restrict inputs_pointer = INPUTS + nchw_shift + (h - padding_const)*W_const + (w - padding_const);

//                         for (c = 0; c < C_const; ++c){
//                             for (y = 0; y < Y_const; ++y){                                
//                                 for (x = 0; x < X_const; ++x){
//                                     convolution += (*(++inputs_pointer)) * (*(++filters_pointer));
//                                 } // x
//                                 inputs_pointer += XW;
//                             } // y
//                             inputs_pointer += HYW;
//                         } // c
//                     }
//                     else { // if we're computing convolution in padding region
//                         float *restrict inputs_pointer = INPUTS + nchw_shift + mx(mn(h-padding_const, H_const-1), 0)*W_const + mx(mn(w-padding_const, W_const-1), 0);
    
//                         for (c = 0; c < C_const; ++c){
//                             float *restrict inputs_pointer_ncYX = inputs_pointer;
                            
//                             for (y = 0; y < Y_const; ++y){
//                                 float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                                
//                                 if ((y + h - padding_const >= 0) && (y + h - padding_const < H_const)){
//                                     for (x = 0; x < X_const; ++x){
//                                         if ((x + w - padding_const >= 0) && (x + w - padding_const < W_const))
//                                             filters_pointer++;
//                                             convolution += (*(inputs_pointer++)) * (*filters_pointer);
//                                     } // x
//                                 }
//                                 else filters_pointer += X_const; // advance pointer without going into loop

//                                 inputs_pointer = inputs_pointer_ncyX + W_const; 
//                             } // y
    
//                             inputs_pointer = inputs_pointer_ncYX + H_const*W_const; 
//                         } // c
//                     }
                    
//                     // ------------------ POOLING ------------------ 
//                     // compute boundaries of pooling windows that are affected by the particular convolution element considered
//                     int i_start = (h + pooling_stride_const - pooling_radius_const)/pooling_stride_const; // this is computing ceil((h - pooling_radius + 1)/pooling_stride)
//                     i_start = (i_start < 0) ? 0 : i_start;
//                     int j_start = (w + pooling_stride_const - pooling_radius_const)/pooling_stride_const;
//                     j_start = (j_start < 0) ? 0 : j_start;
//                     int i_end = h/pooling_stride_const; // this is computing floor(h/pooling_stride_const)
//                     i_end = (i_end >= pooled_H_const) ? (pooled_H_const - 1) : i_end;
//                     int j_end = w/pooling_stride_const;
//                     j_end = (j_end >= pooled_W_const) ? (pooled_W_const - 1) : j_end;

//                     outputs_pointer = OUTPUTS + (nk*pooled_H_const + i_start)*pooled_W_const + j_start;
//                     int *restrict argmaxs_pointer = ARGMAXS + (nk*pooled_H_const + i_start)*pooled_W_const + j_start - 1;
//                     for (i = i_start; i <= i_end; ++i){
//                         for (j = j_start; j <= j_end; ++j){
//                             if (convolution > *outputs_pointer){
//                                 *(outputs_pointer++) = convolution;
//                                 *(++argmaxs_pointer) = hw;
//                             } //i
//                         } // j
//                         outputs_pointer += -(j_end - j_start) + pooled_W_const;
//                         argmaxs_pointer += -(j_end - j_start) + pooled_W_const;
//                     } // i

//                     hw++;
//                 } // w
//             } // h

//         } // nk
         
//     } // pragma_offload

//     return ARGMAXS;
// }

void convolve_gradient_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){

  #pragma offload target(mic:MIC_DEV) \ 
  in(INPUTS:length(0) REUSE) \ 
  in(FILTERS:length(0) REUSE) \ 
  in(ARGMAXS:length(0) REUSE) \
  in(D_POOLED_OUTPUTS:length(0) REUSE) \
  in(D_FILTERS:length(0) REUSE) \
  in(SCRATCH:length(0) REUSE)
  {
        int nkc, ncHW, h, w, k, c, lin_index, y, x, h_arg, w_arg, n, tmp;
        int XW = -X_const + W_const,
                    HYW = (H_const-Y_const)*W_const;

        #pragma omp parallel for \
        default(none) \
        schedule(dynamic, C_const) \
        private(nkc, ncHW, h, w, k, c, lin_index, y, x, h_arg, w_arg, n, tmp) \
        shared(N, INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, SCRATCH, XW, HYW)

        for (nkc = 0; nkc < N*K_const*C_const; nkc++){
            n = nkc/(K_const*C_const);
            k = md(nkc, K_const*C_const)/C_const;
            c = md(nkc, C_const);

            // k = nkc/(N*C_const);
            // n = md(nkc, N*C_const)/C_const;
            // c = md(nkc, C_const);

            // n = nkc/(K_const*C_const);
            // c = md(nkc, K_const*C_const)/K_const;
            // k = md(nkc, K_const);

            // k = nkc/(N*C_const);
            // c = md(nkc, N*C_const)/N;
            // n = md(nkc, N);

            ncHW = (n*C_const + c)*H_const*W_const;

            float *restrict d_pooled_outputs_pointer = D_POOLED_OUTPUTS + (n*K_const + k)*pooled_H_const*pooled_W_const;
            int *restrict argmaxs_pointer = ARGMAXS + (n*K_const + k)*pooled_H_const*pooled_W_const - 1;

            float *d_filters_tmp = SCRATCH + 2*omp_get_thread_num()*Y_const*X_const - 1;
            for (y = 0; y < Y_const; y++){
                for (x = 0; x < X_const; x++){
                    *(++d_filters_tmp) = 0.f;
                } // x
            } // y
            d_filters_tmp = SCRATCH + 2*omp_get_thread_num()*Y_const*X_const - 1;

            int kcyx_shift = (k*C_const + c)*Y_const*X_const - 1;
            for (h = 0; h < pooled_H_const; h++){
                for (w = 0; w < pooled_W_const; w++){
                    
                    lin_index = *(++argmaxs_pointer);
                    h_arg = lin_index/output_W_const;
                    w_arg = lin_index - h_arg*output_W_const;

                    float d_pooled_outputs_value = *d_pooled_outputs_pointer;

                    // float *restrict d_filters_pointer = D_FILTERS + kcyx_shift;
                    float *restrict d_filters_pointer = d_filters_tmp;

                    if ((w_arg - padding_const >= 0) &&
                        (h_arg - padding_const >= 0) &&
                        (output_W_const - 1 - w_arg >= padding_const) &&
                        (output_H_const - 1 - h_arg >= padding_const)){
                        
                        float *restrict inputs_pointer = INPUTS + (w_arg-padding_const) + (h_arg-padding_const)*W_const + ncHW - 1;
                        // if ((w_arg-padding_const) < 0) {printf("\n\n W"); break; break; break; break;}
                        // if (((w_arg-padding_const) + (h_arg-padding_const)*W_const) < 0) {printf("\n\n JOINT"); break; break; break; break;}
                        float value = 0.f;
                        for (y = 0; y < Y_const; y++){
                            for (x = 0; x < X_const; x++){
                                *(++d_filters_pointer) += d_pooled_outputs_value * (*(++inputs_pointer)); // kcyx += nkhw * inputs_ind
                            } // x
                            inputs_pointer += XW;
                        } // y

                    }
                    else{
                        float *restrict inputs_pointer = INPUTS + ncHW - 1 + mx(mn(h_arg-padding_const, H_const-1), 0)*W_const + mx(mn(w_arg-padding_const, W_const-1), 0);

                        for (y = 0; y < Y_const; ++y){
                            float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                            if ((y + h_arg - padding_const >= 0) && (y + h_arg - padding_const < H_const)){
                                for (x = 0; x < X_const; ++x){
                                    d_filters_pointer++;
                                    if ((x + w_arg - padding_const >= 0) && (x + w_arg - padding_const < W_const))
                                        *d_filters_pointer += (*d_pooled_outputs_pointer) * (*(++inputs_pointer)); 
                                } // x
                            }

                            else d_filters_pointer += X_const; // advance pointer without going into loop
                            inputs_pointer = inputs_pointer_ncyX + W_const; 
                        } // y

                        // for (y = 0; y < Y_const; y++){
                        //     float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                        //     for (x = 0; x < X_const; x++){
                               
                        //         d_filters_pointer++;
                        //         if ((y + h_arg - padding_const >= 0) && (y + h_arg - padding_const < H_const) && (x + w_arg - padding_const >= 0) && (x + w_arg - padding_const < W_const))
                        //             *d_filters_pointer += (*d_pooled_outputs_pointer) * (*(++inputs_pointer)); // kcyx += nkhw * inputs_ind

                        //     } // x

                        //     inputs_pointer = inputs_pointer_ncyX + W_const;
                        // } // y
                    }
                    d_pooled_outputs_pointer++;
                } // w
            } // h

            float *restrict d_filters_pointer = D_FILTERS + kcyx_shift;
            for (y = 0; y < Y_const; y++){
                for (x = 0; x < X_const; x++){
                    *(++d_filters_pointer) += *(++d_filters_tmp); // kcyx += nkhw * inputs_ind
                } // x
            } // y
        } // nkc
  
  }
}

void convolve_gradient_shadow_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){

  #pragma offload target(mic:MIC_DEV) \ 
  in(INPUTS:length(0) REUSE) \ 
  in(FILTERS:length(0) REUSE) \ 
  in(ARGMAXS:length(0) REUSE) \
  in(D_POOLED_OUTPUTS:length(0) REUSE) \
  in(D_FILTERS:length(0) REUSE) \
  in(SCRATCH:length(0) REUSE)
  {
        int nkc, ncHW, h, w, k, c, lin_index, y, x, h_arg, w_arg, n, tmp;
        int XW = -X_const + W_const,
                    HYW = (H_const-Y_const)*W_const;

        #pragma omp parallel for \
        default(none) \
        schedule(dynamic, C_const) \
        private(nkc, ncHW, h, w, k, c, lin_index, y, x, h_arg, w_arg, n, tmp) \
        shared(N, INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, SCRATCH, XW, HYW)

        for (nkc = 0; nkc < N*K_const*C_const; nkc++){
            n = nkc/(K_const*C_const);
            k = md(nkc, K_const*C_const)/C_const;
            c = md(nkc, C_const);

            // k = nkc/(N*C_const);
            // n = md(nkc, N*C_const)/C_const;
            // c = md(nkc, C_const);

            // n = nkc/(K_const*C_const);
            // c = md(nkc, K_const*C_const)/K_const;
            // k = md(nkc, K_const);

            // k = nkc/(N*C_const);
            // c = md(nkc, N*C_const)/N;
            // n = md(nkc, N);

            ncHW = (n*C_const + c)*H_const*W_const;

            float *restrict d_pooled_outputs_pointer = D_POOLED_OUTPUTS + N*K_const*pooled_H_const*pooled_W_const + (n*K_const + k)*pooled_H_const*pooled_W_const;
            int *restrict argmaxs_pointer = ARGMAXS + N*K_const*pooled_H_const*pooled_W_const + (n*K_const + k)*pooled_H_const*pooled_W_const - 1;

            float *d_filters_tmp = SCRATCH + 2*omp_get_thread_num()*Y_const*X_const - 1;
            for (y = 0; y < Y_const; y++){
                for (x = 0; x < X_const; x++){
                    *(++d_filters_tmp) = 0.f;
                } // x
            } // y
            d_filters_tmp = SCRATCH + 2*omp_get_thread_num()*Y_const*X_const - 1;

            int kcyx_shift = (k*C_const + c)*Y_const*X_const - 1;
            for (h = 0; h < pooled_H_const; h++){
                for (w = 0; w < pooled_W_const; w++){
                    
                    lin_index = *(++argmaxs_pointer);
                    h_arg = lin_index/output_W_const;
                    w_arg = lin_index - h_arg*output_W_const;

                    float d_pooled_outputs_value = *d_pooled_outputs_pointer;

                    // float *restrict d_filters_pointer = D_FILTERS + kcyx_shift;
                    float *restrict d_filters_pointer = d_filters_tmp;

                    if ((w_arg - padding_const >= 0) &&
                        (h_arg - padding_const >= 0) &&
                        (output_W_const - 1 - w_arg >= padding_const) &&
                        (output_H_const - 1 - h_arg >= padding_const)){
                        
                        float *restrict inputs_pointer = INPUTS + (w_arg-padding_const) + (h_arg-padding_const)*W_const + ncHW - 1;
                        // if ((w_arg-padding_const) < 0) {printf("\n\n W"); break; break; break; break;}
                        // if (((w_arg-padding_const) + (h_arg-padding_const)*W_const) < 0) {printf("\n\n JOINT"); break; break; break; break;}
                        float value = 0.f;
                        for (y = 0; y < Y_const; y++){
                            for (x = 0; x < X_const; x++){
                                *(++d_filters_pointer) += d_pooled_outputs_value * (*(++inputs_pointer)); // kcyx += nkhw * inputs_ind
                            } // x
                            inputs_pointer += XW;
                        } // y

                    }
                    else{
                        float *restrict inputs_pointer = INPUTS + ncHW - 1 + mx(mn(h_arg-padding_const, H_const-1), 0)*W_const + mx(mn(w_arg-padding_const, W_const-1), 0);

                        for (y = 0; y < Y_const; ++y){
                            float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                            if ((y + h_arg - padding_const >= 0) && (y + h_arg - padding_const < H_const)){
                                for (x = 0; x < X_const; ++x){
                                    d_filters_pointer++;
                                    if ((x + w_arg - padding_const >= 0) && (x + w_arg - padding_const < W_const))
                                        *d_filters_pointer += (*d_pooled_outputs_pointer) * (*(++inputs_pointer)); 
                                } // x
                            }

                            else d_filters_pointer += X_const; // advance pointer without going into loop
                            inputs_pointer = inputs_pointer_ncyX + W_const; 
                        } // y

                        // for (y = 0; y < Y_const; y++){
                        //     float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                        //     for (x = 0; x < X_const; x++){
                               
                        //         d_filters_pointer++;
                        //         if ((y + h_arg - padding_const >= 0) && (y + h_arg - padding_const < H_const) && (x + w_arg - padding_const >= 0) && (x + w_arg - padding_const < W_const))
                        //             *d_filters_pointer += (*d_pooled_outputs_pointer) * (*(++inputs_pointer)); // kcyx += nkhw * inputs_ind

                        //     } // x

                        //     inputs_pointer = inputs_pointer_ncyX + W_const;
                        // } // y
                    }
                    d_pooled_outputs_pointer++;
                } // w
            } // h

            float *restrict d_filters_pointer = D_FILTERS + kcyx_shift;
            for (y = 0; y < Y_const; y++){
                for (x = 0; x < X_const; x++){
                    *(++d_filters_pointer) -= *(++d_filters_tmp); // kcyx += nkhw * inputs_ind
                } // x
            } // y
        } // nkc
  
  }
}


int *convolve_argmaxs_fixed_layer1(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded){

    assert(C == C_const);
    assert(H == H_const);
    assert(W == W_const);
    assert(K == K_const);
    assert(stride == stride_const);
    assert(padding == padding_const);
    assert(pooling_radius == pooling_radius_const);
    assert(pooling_stride == pooling_stride_const);
    assert(X == X_const);
    assert(Y == Y_const);
    assert(output_H_const == (H_const + 2*padding_const - Y_const + 1)/stride_const);
    assert(output_W_const == (W_const + 2*padding_const - X_const + 1)/stride_const);
    assert(pooled_H_const == (output_H_const - pooling_radius_const + 1)/pooling_stride_const);
    assert(pooled_W_const == (output_W_const - pooling_radius_const + 1)/pooling_stride_const);

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE) \
    in(ARGMAXS:length(0) REUSE)
    {
        float convolution = 0.f;
        int ij, nkhw, n, k, i, j, h, w, c, y, x;
        int nk, tmp, lin_index, h_arg, w_arg;

        // computation of constants
        int XW = -X_const + W_const,
            HYW = (H_const-Y_const)*W_const;

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nkhw, ij, lin_index, h_arg, w_arg, nk, n, k, h, w, c, y, x, i, j, convolution, tmp) \
            shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, XW, HYW)
        
        // for each example, for each filter, for each pooling region...
        for (nk = 0; nk < N*K_const; nk++){
            n = nk / K_const;
            k = md(nk, K_const);
            
            float *restrict pooled_outputs_pointer = OUTPUTS + nk*pooled_H_const*pooled_W_const - 1;

            int nCHW = n*C_const*H_const*W_const,
                nkhw_shift = nk*output_H_const*output_W_const,
                kcyx_shift = k*C_const*Y_const*X_const - 1; // filters

            for (ij = 0; ij < pooled_H_const*pooled_W_const; ij++){

                // get index (which is in input space)
                lin_index = ARGMAXS[nk*pooled_H_const*pooled_W_const + ij];
                h_arg = lin_index/output_W_const;
                w_arg = lin_index - h_arg*output_W_const;
                int nchw_shift = w_arg + h_arg*W_const + nCHW - 1;

                convolution = 0.f;
                float *restrict filters_pointer = FILTERS + kcyx_shift;
                // float *restrict inputs_pointer = INPUTS + nchw_shift;
                
                // if we're not on boundary (i.e not affected by padding)
                if (w_arg - padding_const >= 0 &&
                    h_arg - padding_const >= 0 &&
                    output_W_const - 1 - w_arg >= padding_const  &&
                    output_H_const - 1 - h_arg >= padding_const){

                    float *restrict inputs_pointer = INPUTS + nCHW + (h_arg-padding_const)*W_const + (w_arg-padding_const) - 1;
                    for (c = 0; c < C_const; ++c){
                        for (y = 0; y < Y_const; ++y){                                
                            for (x = 0; x < X_const; ++x){
                                convolution += (*(++inputs_pointer)) * (*(++filters_pointer));
                            } // x
                            inputs_pointer += XW;
                        } // y
                        inputs_pointer += HYW;
                    } // c

                }

                else{
                    float *restrict inputs_pointer = INPUTS + nCHW - 1 + mx(mn(h_arg-padding_const, H_const-1), 0)*W_const + mx(mn(w_arg-padding_const, W_const-1), 0);

                    for (c = 0; c < C_const; ++c){
                        float *restrict inputs_pointer_ncYX = inputs_pointer;
                        
                        for (y = 0; y < Y_const; ++y){
                            
                            float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                            
                            if ((y + h_arg - padding_const >= 0) && (y + h_arg - padding_const < H_const)){
                                for (x = 0; x < X_const; ++x){
                                    filters_pointer++;
                                    if ((x + w_arg - padding_const >= 0) && (x + w_arg - padding_const < W_const))
                                        convolution += (*(++inputs_pointer)) * (*filters_pointer);
                                } // x
                            }

                            else{ // advance pointer without going into loop
                                    // inputs_pointer += X_const;
                                    filters_pointer += X_const;
                                } 

                            inputs_pointer = inputs_pointer_ncyX + W_const; 
                        } // y

                        inputs_pointer = inputs_pointer_ncYX + H_const*W_const; 
                    } // c
                }

                // // loop over convolution elements
                // for (c = 0; c < C_const; ++c){
                //     for (y = 0; y < Y_const; ++y){
                //         for (x = 0; x < X_const; ++x){
                //             convolution += (*(++inputs_pointer)) * (*(++filters_pointer));
                //         }

                //         inputs_pointer += XW;
                //     }

                //     inputs_pointer += HYW;
                // }
                
                *(++pooled_outputs_pointer) = convolution;
            } // ij

        } // nk
         
    } // pragma_offload

    return ARGMAXS;
}

int *convolve_argmaxs_fixed_layer2(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded){

    assert(C == C_const2);
    assert(H == H_const2);
    assert(W == W_const2);
    assert(K == K_const2);
    assert(stride == stride_const2);
    assert(padding == padding_const2);
    assert(pooling_radius == pooling_radius_const2);
    assert(pooling_stride == pooling_stride_const2);
    assert(X == X_const2);
    assert(Y == Y_const2);
    assert(output_H_const2 == (H_const2 + 2*padding_const2 - Y_const2 + 1)/stride_const2);
    assert(output_W_const2 == (W_const2 + 2*padding_const2 - X_const2 + 1)/stride_const2);
    assert(pooled_H_const2 == (output_H_const2 - pooling_radius_const2 + 1)/pooling_stride_const2);
    assert(pooled_W_const2 == (output_W_const2 - pooling_radius_const2 + 1)/pooling_stride_const2);

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE) \
    in(ARGMAXS:length(0) REUSE)
    {
        float convolution = 0.f;
        int ij, nkhw, n, k, i, j, h, w, c, y, x;
        int nk, tmp, lin_index, h_arg, w_arg;

        // computation of const2ants
        int XW = -X_const2 + W_const2,
            HYW = (H_const2-Y_const2)*W_const2;

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nkhw, ij, lin_index, h_arg, w_arg, nk, n, k, h, w, c, y, x, i, j, convolution, tmp) \
            shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, XW, HYW)
        
        // for each example, for each filter, for each pooling region...
        for (nk = 0; nk < N*K_const2; nk++){
            n = nk / K_const2;
            k = md(nk, K_const2);
            
            float *restrict pooled_outputs_pointer = OUTPUTS + nk*pooled_H_const2*pooled_W_const2 - 1;

            int nCHW = n*C_const2*H_const2*W_const2,
                nkhw_shift = nk*output_H_const2*output_W_const2,
                kcyx_shift = k*C_const2*Y_const2*X_const2 - 1; // filters

            for (ij = 0; ij < pooled_H_const2*pooled_W_const2; ij++){

                // get index (which is in input space)
                lin_index = ARGMAXS[nk*pooled_H_const2*pooled_W_const2 + ij];
                h_arg = lin_index/output_W_const2;
                w_arg = lin_index - h_arg*output_W_const2;
                int nchw_shift = w_arg + h_arg*W_const2 + nCHW - 1;

                convolution = 0.f;
                float *restrict filters_pointer = FILTERS + kcyx_shift;
                // float *restrict inputs_pointer = INPUTS + nchw_shift;
                
                // if we're not on boundary (i.e not affected by padding)
                if (w_arg - padding_const2 >= 0 &&
                    h_arg - padding_const2 >= 0 &&
                    output_W_const2 - 1 - w_arg >= padding_const2  &&
                    output_H_const2 - 1 - h_arg >= padding_const2){

                    float *restrict inputs_pointer = INPUTS + nCHW + (h_arg-padding_const2)*W_const2 + (w_arg-padding_const2) - 1;
                    for (c = 0; c < C_const2; ++c){
                        for (y = 0; y < Y_const2; ++y){                                
                            for (x = 0; x < X_const2; ++x){
                                convolution += (*(++inputs_pointer)) * (*(++filters_pointer));
                            } // x
                            inputs_pointer += XW;
                        } // y
                        inputs_pointer += HYW;
                    } // c

                }

                else{
                    float *restrict inputs_pointer = INPUTS + nCHW - 1 + mx(mn(h_arg-padding_const2, H_const2-1), 0)*W_const2 + mx(mn(w_arg-padding_const2, W_const2-1), 0);

                    for (c = 0; c < C_const2; ++c){
                        float *restrict inputs_pointer_ncYX = inputs_pointer;
                        
                        for (y = 0; y < Y_const2; ++y){
                            
                            float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                            
                            if ((y + h_arg - padding_const2 >= 0) && (y + h_arg - padding_const2 < H_const2)){
                                for (x = 0; x < X_const2; ++x){
                                    filters_pointer++;
                                    if ((x + w_arg - padding_const2 >= 0) && (x + w_arg - padding_const2 < W_const2))
                                        convolution += (*(++inputs_pointer)) * (*filters_pointer);
                                } // x
                            }

                            else{ // advance pointer without going into loop
                                    // inputs_pointer += X_const2;
                                    filters_pointer += X_const2;
                            }  // advance pointer without going into loop

                            inputs_pointer = inputs_pointer_ncyX + W_const2; 
                        } // y

                        inputs_pointer = inputs_pointer_ncYX + H_const2*W_const2; 
                    } // c
                }

                // // loop over convolution elements
                // for (c = 0; c < C_const; ++c){
                //     for (y = 0; y < Y_const; ++y){
                //         for (x = 0; x < X_const; ++x){
                //             convolution += (*(++inputs_pointer)) * (*(++filters_pointer));
                //         }

                //         inputs_pointer += XW;
                //     }

                //     inputs_pointer += HYW;
                // }
                
                *(++pooled_outputs_pointer) = convolution;
            } // ij

        } // nk
         
    } // pragma_offload

    return ARGMAXS;
}

int *convolve_and_pool_layer2(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){

    assert(C == C_const2);
    assert(H == H_const2);
    assert(W == W_const2);
    assert(K == K_const2);
    assert(stride == stride_const2);
    assert(padding == padding_const2);
    assert(pooling_radius == pooling_radius_const2);
    assert(pooling_stride == pooling_stride_const2);
    assert(X == X_const2);
    assert(Y == Y_const2);
    assert(output_H_const2 == (H_const2 + 2*padding_const2 - Y_const2 + 1)/stride_const2);
    assert(output_W_const2 == (W_const2 + 2*padding_const2 - X_const2 + 1)/stride_const2);
    assert(pooled_H_const2 == (output_H_const2 - pooling_radius_const2 + 1)/pooling_stride_const2);
    assert(pooled_W_const2 == (output_W_const2 - pooling_radius_const2 + 1)/pooling_stride_const2);

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE) \
    in(ARGMAXS:length(0) REUSE)
    {
        float convolution = 0.f;
        int nkij, nkhw, n, k, i, j, h, w, c, y, x;
        int nk, ij;

        // computation of const2ants
        int XW = -X_const2 + W_const2,
                    HYW = (H_const2-Y_const2)*W_const2;

        int *restrict indices = _mm_malloc(C_const2*Y_const2*X_const2*sizeof(int), ALIGN);
        int *restrict indices_pointer = indices - 1;
        int index = -1;
        for (c = 0; c < C_const2; ++c){
            for (y = 0; y < Y_const2; ++y){                                
                for (x = 0; x < X_const2; ++x){
                    *(++indices_pointer) = ++index;
                    // printf("%d \n", *indices_pointer);
                    } // x
                    index += XW;
            } // y
            index += HYW;
        } // c

        // #pragma omp parallel for \
            // schedule(dynamic) \
            // private(nkij)
        // for (nkij = 0; nkij < N*K_const2*pooled_H_const2*pooled_W_const2; nkij++) OUTPUTS[nkij] = -1.0e10;

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nkhw, nkij, nk, ij, n, k, h, w, c, y, x, i, j, convolution) \
            shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, XW, HYW, indices)
        
        // for each example, for each filter, for each pooling region...
        for (nk = 0; nk < N*K_const2; nk++){
            float *restrict outputs_pointer = OUTPUTS + nk*pooled_H_const2*pooled_W_const2 - 1;
            for (ij = 0; ij < pooled_H_const2*pooled_W_const2; ij++) *(++outputs_pointer) = -1.0e10;

            n = nk / K_const2;
            k = md(nk, K_const2);
            
            int hw = 0,
                // nkhw = nk*output_H_const2*output_W_const2, // NOTE: not applicable for stride!
                // kcyx_shift = k*C_const2*Y_const2*X_const2 - 1, // filters
                // nchw_shift = (n*C_const2*H_const2)*W_const2 - 1; // inputs

            kcyx_shift = k*C_const2*Y_const2*X_const2, // filters
            nchw_shift = (n*C_const2*H_const2)*W_const2; // inputs

            for (h = 0; h < output_H_const2; h+= stride_const2){
                for (w = 0; w < output_W_const2; w+= stride_const2){
                    convolution = 0.f;
                    float *restrict filters_pointer = FILTERS + kcyx_shift;
                    
                    // if we're on boundary
                    if (w - padding_const2 >= 0 &&
                        h - padding_const2 >= 0 &&
                        output_W_const2 - 1 - w >= padding_const2  &&
                        output_H_const2 - 1 - h >= padding_const2){
                        float *restrict inputs_pointer = INPUTS + nchw_shift + (h - padding_const)*W_const2 + (w - padding_const);

                        convolution = __sec_reduce_add(inputs_pointer[indices[0:C_const2*Y_const2*X_const2]] * filters_pointer[0:C_const2*Y_const2*X_const2]);
                        filters_pointer += C_const2*Y_const2*X_const2;

                        // for (c = 0; c < C_const2; ++c){
                        //     for (y = 0; y < Y_const2; ++y){                                
                        //         for (x = 0; x < X_const2; ++x){
                        //             convolution += (*(++inputs_pointer)) * (*(++filters_pointer));
                        //         } // x

                        //         inputs_pointer += XW;
                        //     } // y
                        //     inputs_pointer += HYW;
                        // } // c

                    }

                    else{
                        float *restrict inputs_pointer = INPUTS + nchw_shift + mx(mn(h-padding_const2, H_const2-1), 0)*W_const2 + mx(mn(w-padding_const2, W_const2-1), 0);

                        for (c = 0; c < C_const2; ++c){
                            float *restrict inputs_pointer_ncYX = inputs_pointer;
                            
                            for (y = 0; y < Y_const2; ++y){
                                
                                float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                                
                                if ((y + h - padding_const2 >= 0) && (y + h - padding_const2 < H_const2)){
                                    for (x = 0; x < X_const2; ++x){
                                        if ((x + w - padding_const2 >= 0) && (x + w - padding_const2 < W_const2))
                                            convolution += (*(inputs_pointer++)) * (*filters_pointer);
                                        filters_pointer++;
                                    } // x
                                }

                                else{ // advance pointer without going into loop
                                    // inputs_pointer += X_const2;
                                    filters_pointer += X_const2;
                                } 
                                
                                inputs_pointer = inputs_pointer_ncyX + W_const2; 
                            } // y

                            inputs_pointer = inputs_pointer_ncYX + H_const2*W_const2; 
                            // inputs_pointer += HYW;
                        } // c
                    }
                    
                    // loop over pooled outputs that care about this particular (pre-pooled) output element
                    // update max, argmax
                    int i_start = (h + pooling_stride_const2 - pooling_radius_const2)/pooling_stride_const2; // ceil((h - pooling_radius + 1)/pooling_stride)
                    i_start = (i_start < 0) ? 0 : i_start;
                    int j_start = (w + pooling_stride_const2 - pooling_radius_const2)/pooling_stride_const2;
                    j_start = (j_start < 0) ? 0 : j_start;
                    int i_end = h/pooling_stride_const2; // floor(h/pooling_stride_const2)
                    i_end = (i_end >= pooled_H_const2) ? (pooled_H_const2 - 1) : i_end;
                    int j_end = w/pooling_stride_const2;
                    j_end = (j_end >= pooled_W_const2) ? (pooled_W_const2 - 1) : j_end;

                    outputs_pointer = OUTPUTS + (nk*pooled_H_const2 + i_start)*pooled_W_const2 + j_start;
                    int *restrict argmaxs_pointer = ARGMAXS + (nk*pooled_H_const2 + i_start)*pooled_W_const2 + j_start - 1;
                    // printf("test %d, test2 %d, h %d w %d, start %d end %d start %d end %d and finally %d \n", ((int) -1)/((int) 2), pooling_stride_const2, h, w, i_start, i_end, j_start, j_end, (i_end-i_start)*(j_end-j_start));
                    for (i = i_start; i <= i_end; ++i){
                        for (j = j_start; j <= j_end; ++j){
                            // nkij = (nk*pooled_H_const2 + i)*pooled_W_const2 + j;
                            if (convolution > *outputs_pointer){
                                *(outputs_pointer++) = convolution;
                                *(++argmaxs_pointer) = hw;
                            }
                        } // j
                        outputs_pointer += -(j_end - j_start) + pooled_W_const2;
                        argmaxs_pointer += -(j_end - j_start) + pooled_W_const2;
                    } // i

                    hw++;
                } // w
            } // h

        } // nk
         
    } // pragma_offload

    return ARGMAXS;
}


// int *convolve_and_pool_layer1(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int pooling_radius, int stride, int *restrict ARGMAXS, int argmaxs_fixed, int offloaded){

//     assert(C == C_const);
//     assert(H == H_const);
//     assert(W == W_const);
//     assert(K == K_const);
//     assert(pooling_radius == pooling_radius_const);
//     assert(stride == stride_const);
//     assert(X == X_const);
//     assert(Y == Y_const);
//     assert(output_H_const == H_const - Y_const + 1);
//     assert(output_W_const == W_const - X_const + 1);
//     assert(pooled_H_const == ceil(((float) output_H_const)/pooling_radius_const));
//     assert(pooled_W_const == ceil(((float) output_W_const)/pooling_radius_const));

//     #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
//     in(INPUTS:length(0) REUSE) \ 
//     in(FILTERS:length(0) REUSE) \ 
//     in(OUTPUTS:length(0) REUSE) \
//     in(ARGMAXS:length(0) REUSE)
//     {
//         float convolution = 0.f;
//         float max_convolution = -1.0e10;
//         int argmax_convolution, nkhwcyx, nkij, cyx, n, k, i, j, h, w, c, y, x, lin_index, h_arg, w_arg;
//         int hw, tmp, region_H, region_W;

//         // computation of constants
//         int XW = -X_const + W_const,
//                     HYW = (H_const-Y_const)*W_const;

//         #pragma omp parallel for \
//         schedule(dynamic, pooled_H_const*pooled_W_const) \
//         default(none) \
//         private(nkij, n, k, i, j, hw, region_H, region_W, h, w, cyx, c, y, x, argmax_convolution, convolution, max_convolution) \
//         shared(N, INPUTS, OUTPUTS, ARGMAXS, FILTERS, XW, HYW)
        
//         // for each example, for each filter, for each pooling region...
//         for (nkij = 0; nkij < N*K_const*pooled_H_const*pooled_W_const; nkij++){
//             n = nkij / (K_const*pooled_H_const*pooled_W_const);
//             k = md(nkij / (pooled_H_const*pooled_W_const), K_const);
//             i = md(nkij / pooled_W_const, pooled_H_const);
//             j = md(nkij, pooled_W_const);

//             // loop over elements in pooling region
//             region_H = (output_H_const - i*pooling_radius_const < pooling_radius_const) ? output_H_const - i*pooling_radius_const : pooling_radius_const;
//             region_W = (output_W_const - j*pooling_radius_const < pooling_radius_const) ? output_W_const - j*pooling_radius_const : pooling_radius_const;
//             max_convolution = -1.0e10;

//             // computation of constants
//             int kcyx_shift = k*C_const*Y_const*X_const - 1,
//                 ncij_shift = (n*C_const*H_const + i*pooling_radius_const)*W_const + j*pooling_radius_const - 1,
//                 nkij_shift = ((n*K_const + k)*output_H_const + i*pooling_radius_const)*output_W_const + j*pooling_radius_const;      

//             for (h = 0; h < region_H; h += stride_const){
//                 for (w = 0; w < region_W; w += stride_const){

//                     convolution = 0.f;
//                     float *restrict filters_pointer = FILTERS + kcyx_shift;
//                     float *restrict inputs_pointer = INPUTS + ncij_shift + h*W_const + w;
                    
//                     for (c = 0; c < C_const; ++c){
//                         for (y = 0; y < Y_const; ++y){
//                             for (x = 0; x < X_const; ++x){
//                                 convolution += (*(++inputs_pointer)) * (*(++filters_pointer));
//                             }

//                             inputs_pointer += XW;
//                         }

//                         inputs_pointer += HYW;
//                     }

//                     if (convolution > max_convolution){
//                         max_convolution = convolution;
//                         argmax_convolution = nkij_shift + h*output_W_const + w;
                         
//                     }
//                 }
//             }

//             OUTPUTS[nkij] = max_convolution;
//             ARGMAXS[nkij] = argmax_convolution;        

//         }
         
//     }

//   return ARGMAXS;
// }

// int *convolve_and_pool_layer2(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int pooling_radius, int stride, int *restrict ARGMAXS, int argmaxs_fixed, int offloaded){

//     assert(C == C_const2);
//     assert(H == H_const2);
//     assert(W == W_const2);
//     assert(K == K_const2);
//     assert(pooling_radius == pooling_radius_const2);
//     assert(stride == stride_const2);
//     assert(X == X_const2);
//     assert(Y == Y_const2);
//     assert(output_H_const2 == H_const2 - Y_const2 + 1);
//     assert(output_W_const2 == W_const2 - X_const2 + 1);
//     assert(pooled_H_const2 == ceil(((float) output_H_const2)/pooling_radius_const2));
//     assert(pooled_W_const2 == ceil(((float) output_W_const2)/pooling_radius_const2));

//     #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
//     in(INPUTS:length(0) REUSE) \ 
//     in(FILTERS:length(0) REUSE) \ 
//     in(OUTPUTS:length(0) REUSE) \
//     in(ARGMAXS:length(0) REUSE)
//     {
//         float convolution = 0.f;
//         float max_convolution = -1.0e10;
//         int argmax_convolution, nkhwcyx, nkij, cyx, n, k, i, j, h, w, c, y, x, lin_index, h_arg, w_arg;
//         int hw, tmp, region_H, region_W;

//         // computation of constants
//         int XW = -X_const2 + W_const2,
//                     HYW = (H_const2-Y_const2)*W_const2;

//         #pragma omp parallel for \
//         schedule(dynamic, pooled_H_const2*pooled_W_const2) \
//         default(none) \
//         private(nkij, n, k, i, j, hw, region_H, region_W, h, w, cyx, c, y, x, argmax_convolution, convolution, max_convolution) \
//         shared(N, INPUTS, OUTPUTS, ARGMAXS, FILTERS, XW, HYW)
        
//         // for each example, for each filter, for each pooling region...
//         for (nkij = 0; nkij < N*K_const2*pooled_H_const2*pooled_W_const2; nkij++){
//             n = nkij / (K_const2*pooled_H_const2*pooled_W_const2);
//             k = md(nkij / (pooled_H_const2*pooled_W_const2), K_const2);
//             i = md(nkij / pooled_W_const2, pooled_H_const2);
//             j = md(nkij, pooled_W_const2);

//             // loop over elements in pooling region
//             region_H = (output_H_const2 - i*pooling_radius_const2 < pooling_radius_const2) ? output_H_const2 - i*pooling_radius_const2 : pooling_radius_const2;
//             region_W = (output_W_const2 - j*pooling_radius_const2 < pooling_radius_const2) ? output_W_const2 - j*pooling_radius_const2 : pooling_radius_const2;
//             max_convolution = -1.0e10;

//             // computation of constants
//             int kcyx_shift = k*C_const2*Y_const2*X_const2 - 1,
//                 ncij_shift = (n*C_const2*H_const2 + i*pooling_radius_const2)*W_const2 + j*pooling_radius_const2 - 1,
//                 nkij_shift = ((n*K_const2 + k)*output_H_const2 + i*pooling_radius_const2)*output_W_const2 + j*pooling_radius_const2;      

//             for (h = 0; h < region_H; h += stride_const2){
//                 for (w = 0; w < region_W; w += stride_const2){

//                     convolution = 0.f;
//                     float *restrict filters_pointer = FILTERS + kcyx_shift;
//                     float *restrict inputs_pointer = INPUTS + ncij_shift + h*W_const2 + w;
                    
//                     for (c = 0; c < C_const2; ++c){
//                         for (y = 0; y < Y_const2; ++y){
//                             for (x = 0; x < X_const2; ++x){
//                                 convolution += (*(++inputs_pointer)) * (*(++filters_pointer));
//                             }

//                             inputs_pointer += XW;
//                         }

//                         inputs_pointer += HYW;
//                     }

//                     if (convolution > max_convolution){
//                         max_convolution = convolution;
//                         argmax_convolution = nkij_shift + h*output_W_const2 + w;
                         
//                     }
//                 }
//             }

//             OUTPUTS[nkij] = max_convolution;
//             ARGMAXS[nkij] = argmax_convolution;        

//         }
         
//     }

//   return ARGMAXS;
// }


// int *convolve_and_pool_layer1(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int pooling_radius, int stride, int *restrict ARGMAXS, int argmaxs_fixed, int offloaded){

//     // assert(N == N_const);
//     assert(C == C_const);
//     assert(H == H_const);
//     assert(W == W_const);
//     assert(K == K_const);
//     assert(pooling_radius == pooling_radius_const);
//     assert(stride == stride_const);
//     assert(X == X_const);
//     assert(Y == Y_const);
//     assert(output_H_const == H_const - Y_const + 1);
//     assert(output_W_const == W_const - X_const + 1);
//     assert(pooled_H_const == ceil(((float) output_H_const)/pooling_radius_const));
//     assert(pooled_W_const == ceil(((float) output_W_const)/pooling_radius_const));

//     #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
//     in(INPUTS:length(0) REUSE) \ 
//     in(FILTERS:length(0) REUSE) \ 
//     in(OUTPUTS:length(0) REUSE) \
//     in(ARGMAXS:length(0) REUSE)
//     {
//         float convolution = 0.f;
//         float max_convolution = -1.0e10;
//         int argmax_convolution, nkhwcyx, nkij, cyx, n, k, i, j, h, w, c, y, x, lin_index, h_arg, w_arg;
//         int hw, tmp, region_H, region_W;

//         tmp = Y_const*X_const;

//         #pragma omp parallel for \
//         schedule(dynamic, pooled_H_const*pooled_W_const) \
//         default(none) \
//         private(nkij, n, k, i, j, hw, region_H, region_W, h, w, cyx, c, y, x, argmax_convolution, convolution, max_convolution) \
//         firstprivate(tmp) \
//         shared(N, INPUTS, OUTPUTS, ARGMAXS, FILTERS)
//         // compute each output element separately
//         for (nkij = 0; nkij < N*K_const*pooled_H_const*pooled_W_const; nkij++){
//             n = nkij / (K_const*pooled_H_const*pooled_W_const);
//             k = md(nkij / (pooled_H_const*pooled_W_const), K_const);
//             i = md(nkij / pooled_W_const, pooled_H_const);
//             j = md(nkij, pooled_W_const);

//             // loop over elements in pooling region
//             region_H = (output_H_const - i*pooling_radius_const < pooling_radius_const) ? output_H_const - i*pooling_radius_const : pooling_radius_const;
//             region_W = (output_W_const - j*pooling_radius_const < pooling_radius_const) ? output_W_const - j*pooling_radius_const : pooling_radius_const;
//             max_convolution = -1.0e10;

//             for (h = 0; h < region_H; h += stride_const){
//               for (w = 0; w < region_W; w += stride_const){
//                 // compute convolution for that particular element
//                 convolution = 0.f;
//                 float *restrict filters_pointer = FILTERS + k*C_const*X_const*Y_const - 1;
//                 float *restrict inputs_pointer = INPUTS + (n*C_const*H_const + i*pooling_radius_const + h)*W_const + j*pooling_radius_const + w - 1;
//                 // int ncHW = (n*C_const*H_const + i*pooling_radius_const + h)*W_const + j*pooling_radius_const + w;
//                 for (c = 0; c < C_const; ++c){
//                     for (y = 0; y < Y_const; ++y){
//                         for (x = 0; x < X_const; ++x){
//                             // convolution += INPUTS[ti(n, c, i*pooling_radius_const + h + y, j*pooling_radius_const + w + x, C_const, H_const, W_const)]
//                             // convolution += INPUTS[ncHW + y*W_const + x]
//                             convolution += (*(++inputs_pointer))
//                                 * (*(++filters_pointer));
//                             }

//                             inputs_pointer += -X_const + W_const;
//                         }

//                     inputs_pointer += (H_const-Y_const)*W_const;    
//                     // ncHW += c*H_const*W_const;
//                     }

//                 if (convolution > max_convolution){
//                     max_convolution = convolution;
//                     argmax_convolution = ti(n, k, i*pooling_radius_const + h, j*pooling_radius_const + w, K_const, output_H_const, output_W_const);
//                 }

//             }
//           }

//             OUTPUTS[nkij] = max_convolution;
//             ARGMAXS[nkij] = argmax_convolution;        

//         }
         
//   }

//   return ARGMAXS;
// }


// int *convolve_and_pool_layer1(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int pooling_radius, int stride, int *restrict ARGMAXS, int argmaxs_fixed, int offloaded){
//     assert(C == C_const);
//     assert(H == H_const);
//     assert(W == W_const);
//     assert(K == K_const);
//     assert(pooling_radius == pooling_radius_const);
//     assert(stride == stride_const);
//     assert(X == X_const);
//     assert(Y == Y_const);
//     assert(output_H_const == H_const - Y_const + 1);
//     assert(output_W_const == W_const - X_const + 1);
//     assert(pooled_H_const == ceil(((float) output_H_const)/pooling_radius_const));
//     assert(pooled_W_const == ceil(((float) output_W_const)/pooling_radius_const));

//     #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
//     in(INPUTS:length(0) REUSE) \ 
//     in(FILTERS:length(0) REUSE) \ 
//     in(OUTPUTS:length(0) REUSE) \
//     in(ARGMAXS:length(0) REUSE)
//     {
//         float convolution = 0.f;
//         int nkhw, n, k, h, w, pooled_h, pooled_w, c, y, x, nkhw;    

//         #pragma omp parallel for \
//         schedule(dynamic, W_const) \
//         default(none) \
//         private(convolution, nkhw, n, k, h, w, pooled_h, pooled_w, c, y, x, nkhw) \ 
//         shared(N, INPUTS, OUTPUTS, ARGMAXS, FILTERS)
//         for (nkhw = 0; nkhw < N*K_const*H_const*W_const; nkhw++){
//             n = nkhw/(K_const*H_const*W_const);
//             k = md(nkhw/(H_const*W_const), K_const);
//             h = md(nkhw/W_const, H_const);
//             w = md(nkhw, W_const);
//         // for (n = 0; n < N; n++){
//         //     for (k = 0; k < K_const; k++){
//         //         for (h = 0; h < H_const; h++){
//         //             for (w = 0; w < W_const; w++){

//                         pooled_h = h/pooling_radius_const;
//                         pooled_w = w/pooling_radius_const;

//                         convolution = 0.f;

//                         for (c = 0; c < C_const; c++){
//                             for (y = 0; y < Y_const; y++){
//                                 for (x = 0; x < X_const; x++){
//                                     convolution += INPUTS[ti(n, c, h + y, w + x, C_const, H_const, W_const)]
//                                                  * FILTERS[ti(k, c, y, x, C_const, Y_const, X_const)];
//                                 }
//                             }
//                         }

//                         nkhw = ti(n, k, pooled_h, pooled_w, K_const, pooled_H_const, pooled_W_const); 
//                         if (convolution > OUTPUTS[nkhw]){
//                             OUTPUTS[nkhw] = convolution;
//                             ARGMAXS[nkhw] = ti(n, k, h, w, K_const, output_H_const, output_W_const);   
//                         }
                        

//             //         }
//             //     }
//             // }
//         }
//     }

//   return ARGMAXS;
// }

// void convolve_gradient_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, int pooling_radius, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){

//   // const int output_H_const = H_const - Y_const + 1;
//   // const int output_W_const = W_const - X_const + 1;
//   // const int pooled_H_const = ceil(((float) output_H_const)/pooling_radius);
//   // const int pooled_W_const = ceil(((float) output_W_const)/pooling_radius);

//   #pragma offload target(mic:MIC_DEV) \ 
//   in(INPUTS:length(0) REUSE) \ 
//   in(FILTERS:length(0) REUSE) \ 
//   in(ARGMAXS:length(0) REUSE) \
//   in(D_INPUTS:length(0) REUSE) \
//   in(D_POOLED_OUTPUTS:length(0) REUSE) \
//   in(D_FILTERS:length(0) REUSE)
//   {
//         int nkc, nkhw, ncHW, h, w, k, c, kcyx, lin_index, inputs_ind, y, x, h_arg, w_arg, n, tmp;
//         #pragma omp parallel for \
//         default(none) \
//         private(nkc, nkhw, kcyx, ncHW, h, w, k, c, lin_index, inputs_ind, y, x, h_arg, w_arg, n, tmp) \
//         shared(N, INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_INPUTS, D_FILTERS)
//       // loop over elements of filter
//         for (nkc = 0; nkc < N*K_const*C_const; nkc++){
//             n = nkc/(K_const*C_const);
//             k = md(nkc, K_const*C_const)/C_const;
//             c = md(nkc, C_const);
//             ncHW = (n*C_const + c)*H_const*W_const;

//             float *restrict d_pooled_outputs_pointer = D_POOLED_OUTPUTS + (n*K_const + k)*pooled_H_const*pooled_W_const;
//             int *restrict argmaxs_pointer = ARGMAXS + (n*K_const + k)*pooled_H_const*pooled_W_const - 1;

//             for (h = 0; h < pooled_H_const; h++){
//                 for (w = 0; w < pooled_W_const; w++){
//                     float *restrict d_filters_pointer = D_FILTERS + (k*C_const + c)*Y_const*X_const - 1;
//                     float *restrict filters_pointer = FILTERS + (k*C_const + c)*Y_const*X_const - 1;

//                     lin_index = (*(++argmaxs_pointer));
//                     tmp = output_H_const*output_W_const; tmp = md(lin_index, tmp);
//                     h_arg = tmp/output_W_const;
//                     w_arg = tmp - h_arg*output_W_const;

//                     float *restrict inputs_pointer = INPUTS + ncHW + h_arg*W_const + w_arg - 1;
//                     float *restrict d_inputs_pointer = INPUTS + ncHW + h_arg*W_const + w_arg - 1;


//                     for (y = 0; y < Y_const; ++y){
//                         for (x = 0; x < X_const; ++x){
//                             (*(++d_filters_pointer)) += (*d_pooled_outputs_pointer) * (*(++inputs_pointer));
//                             (*(++d_inputs_pointer)) += (*d_pooled_outputs_pointer) * (*(++filters_pointer));
//                         }

//                         inputs_pointer += W_const - X_const;
//                         d_inputs_pointer += W_const - X_const;
//                     }

//                     ++d_pooled_outputs_pointer;
//                 }
//             }

//         }
  
//   }
// }

// void convolve_gradient_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){

//     #pragma offload target(mic:MIC_DEV) \ 
//     in(INPUTS:length(0) REUSE) \ 
//     in(FILTERS:length(0) REUSE) \ 
//     in(ARGMAXS:length(0) REUSE) \
//     in(D_INPUTS:length(0) REUSE) \
//     in(D_POOLED_OUTPUTS:length(0) REUSE) \
//     in(D_FILTERS:length(0) REUSE)
//     {
//         float convolution = 0.f;
//         int ij, nkhw, n, k, i, j, h, w, c, y, x;
//         int nk, tmp, lin_index, h_arg, w_arg;

//         // computation of constants
//         int XW = -X_const + W_const,
//                     HYW = (H_const-Y_const)*W_const;

//         #pragma omp parallel for \
//             schedule(dynamic, K_const) \
//             default(none) \
//             private(nkhw, ij, lin_index, h_arg, w_arg, nk, n, k, h, w, c, y, x, i, j, convolution, tmp) \
//             shared(N, INPUTS, FILTERS, ARGMAXS, XW, HYW, D_POOLED_OUTPUTS, D_INPUTS, D_FILTERS)
        
//         // for each example, for each filter, for each pooling region...
//         for (nk = 0; nk < N*K_const; nk++){
//             n = nk / K_const;
//             k = md(nk, K_const);
            
//             float *restrict d_pooled_outputs_pointer = D_POOLED_OUTPUTS + nk*pooled_H_const*pooled_W_const;
//             int *restrict argmaxs_pointer = ARGMAXS + nk*pooled_H_const*pooled_W_const;

//             int nCHW = n*C_const*H_const*W_const,
//                 kcyx_shift = k*C_const*Y_const*X_const - 1; // filters

//             for (ij = 0; ij < pooled_H_const*pooled_W_const; ij++){

//                 // get index (which is in input space)
//                 lin_index = *(argmaxs_pointer++);
//                 tmp = output_H_const*output_W_const; tmp = md(lin_index, tmp);
//                 h_arg = tmp/output_W_const;
//                 w_arg = tmp - h_arg*output_W_const;
//                 int nchw_shift = w_arg + h_arg*W_const + nCHW - 1;

//                 float *restrict filters_pointer = FILTERS + kcyx_shift;
//                 float *restrict d_filters_pointer = D_FILTERS + kcyx_shift;
//                 float *restrict inputs_pointer = INPUTS + nchw_shift;
//                 float *restrict d_inputs_pointer = D_INPUTS + nchw_shift;
                       
//                 // loop over convolution elements
//                 for (c = 0; c < C_const; ++c){
//                     for (y = 0; y < Y_const; ++y){
//                         for (x = 0; x < X_const; ++x){
//                             *(++d_filters_pointer) += (*d_pooled_outputs_pointer) * (*(++inputs_pointer));
//                             *(++d_inputs_pointer) += (*d_pooled_outputs_pointer) * (*(++filters_pointer));

//                             // TRY TO CONVERT EVEYTHING FROM POINTERS TO INDICES, LIKE FORMULATED ORINALLY?

//                             // D_FILTERS[kcyx] += D_POOLED_OUTPUTS[nkhw] * INPUTS[inputs_ind];
//                             // D_INPUTS[inputs_ind] += D_POOLED_OUTPUTS[nkhw] * FILTERS[kcyx];
//                         }

//                         inputs_pointer += XW;
//                         d_inputs_pointer += XW;
//                     }

//                     inputs_pointer += HYW;
//                     d_inputs_pointer += HYW;
//                 }

//                 d_pooled_outputs_pointer++;
//             } // ij

//         } // nk
         
//     } // pragma_offload
// }



// void convolve_gradient_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){

//   #pragma offload target(mic:MIC_DEV) \ 
//   in(INPUTS:length(0) REUSE) \ 
//   in(FILTERS:length(0) REUSE) \ 
//   in(ARGMAXS:length(0) REUSE) \
//   in(D_INPUTS:length(0) REUSE) \
//   in(D_POOLED_OUTPUTS:length(0) REUSE) \
//   in(D_FILTERS:length(0) REUSE)
//   {
//         int nkc, nkhw, ncHW, h, w, k, c, kcyx, lin_index, inputs_ind_shift, inputs_ind, y, x, h_arg, w_arg, n, tmp;
//         #pragma omp parallel for \
//         default(none) \
//         private(nkc, nkhw, kcyx, ncHW, h, w, k, c, lin_index, inputs_ind, inputs_ind_shift, y, x, h_arg, w_arg, n, tmp) \
//         shared(N, INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_INPUTS, D_FILTERS)
//       // loop over elements of filter
//         for (nkc = 0; nkc < N*K_const*C_const; nkc++){
//             n = nkc/(K_const*C_const);
//             k = md(nkc, K_const*C_const)/C_const;
//             c = md(nkc, C_const);
//             nkhw = (n*K_const + k)*pooled_H_const*pooled_W_const;
//             ncHW = (n*C_const + c)*H_const*W_const;

//             for (h = 0; h < pooled_H_const; h++){
//                 for (w = 0; w < pooled_W_const; w++){
//                     kcyx = (k*C_const + c)*Y_const*X_const;
                    
//                     lin_index = ARGMAXS[nkhw];
//                     tmp = output_H_const*output_W_const; tmp = md(lin_index, tmp);
//                     h_arg = tmp/output_W_const;
//                     w_arg = tmp - h_arg*output_W_const;
//                     inputs_ind_shift = w_arg + h_arg*W_const + ncHW;

//                     // float *restrict filters_pointer = FILTERS + kcyx_shift;
// //                 float *restrict d_filters_pointer = D_FILTERS + kcyx_shift;
// //                 float *restrict inputs_pointer = INPUTS + nchw_shift;
// //                 float *restrict d_inputs_pointer = D_INPUTS + nchw_shift;

//                     for (y = 0; y < Y_const; y++){
//                         for (x = 0; x < X_const; x++){
//                             inputs_ind = x + y*W_const + inputs_ind_shift;

//                             // *(++d_filters_pointer) += (*d_pooled_outputs_pointer) * (*(++inputs_pointer));
//                             // *(++d_inputs_pointer) += (*d_pooled_outputs_pointer) * (*(++filters_pointer));
                            
//                             D_FILTERS[kcyx] += D_POOLED_OUTPUTS[nkhw] * INPUTS[inputs_ind];
//                             D_INPUTS[inputs_ind] += D_POOLED_OUTPUTS[nkhw] * FILTERS[kcyx];

//                             kcyx++;
//                         }
//                     }

//                     nkhw++;
//                 }
//             }

//         }
  
//   }
// }


// void convolve_gradient_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){

//   #pragma offload target(mic:MIC_DEV) \ 
//   in(INPUTS:length(0) REUSE) \ 
//   in(FILTERS:length(0) REUSE) \ 
//   in(ARGMAXS:length(0) REUSE) \
//   in(D_POOLED_OUTPUTS:length(0) REUSE) \
//   in(D_FILTERS:length(0) REUSE)
//   {
//         int nkc, ncHW, h, w, k, c, lin_index, y, x, h_arg, w_arg, n, tmp;
//         int XW = -X_const + W_const,
//                     HYW = (H_const-Y_const)*W_const;

//         #pragma omp parallel for \
//         default(none) \
//         schedule(dynamic, N) \
//         private(nkc, ncHW, h, w, k, c, lin_index, y, x, h_arg, w_arg, n, tmp) \
//         shared(N, INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, XW, HYW)

//         for (nkc = 0; nkc < N*K_const*C_const; nkc++){
//             // n = nkc/(K_const*C_const);
//             // k = md(nkc, K_const*C_const)/C_const;
//             // c = md(nkc, C_const);

//             // k = nkc/(N*C_const);
//             // n = md(nkc, N*C_const)/C_const;
//             // c = md(nkc, C_const);

//             // n = nkc/(K_const*C_const);
//             // c = md(nkc, K_const*C_const)/K_const;
//             // k = md(nkc, K_const);

//             k = nkc/(N*C_const);
//             c = md(nkc, N*C_const)/N;
//             n = md(nkc, N);

//             ncHW = (n*C_const + c)*H_const*W_const;

//             float *restrict d_pooled_outputs_pointer = D_POOLED_OUTPUTS + (n*K_const + k)*pooled_H_const*pooled_W_const;
//             int *restrict argmaxs_pointer = ARGMAXS + (n*K_const + k)*pooled_H_const*pooled_W_const - 1;

//             int kcyx_shift = (k*C_const + c)*Y_const*X_const - 1;

//             for (h = 0; h < pooled_H_const; h++){
//                 for (w = 0; w < pooled_W_const; w++){
                    
//                     lin_index = *(++argmaxs_pointer);
//                     // h_arg = lin_index/output_W_const;
//                     // w_arg = lin_index - h_arg*output_W_const;

//                     h_arg = lin_index/output_W_const;
//                     w_arg = lin_index - h_arg*output_W_const;

//                     float *restrict d_filters_pointer = D_FILTERS + kcyx_shift;
//                     float *restrict inputs_pointer = INPUTS + w_arg + h_arg*W_const + ncHW - 1;
//                     // float *restrict inputs_pointer = INPUTS + mx(0, w_arg - padding_const) + mx(0, h_arg - padding_const)*W_const + ncHW - 1;
                    
//                     // int X_padding = mn(X_const, w - padding_const + X_const);
//                     // X_padding = mn(X_padding, )
//                     // int Y_padding = mn(Y_const, h - padding_const + Y_const);

//                     for (y = 0; y < Y_const; ++y){
//                         for (x = 0; x < X_const; ++x){
//                             *(++d_filters_pointer) += (*d_pooled_outputs_pointer) * (*(++inputs_pointer)); // kcyx += nkhw * inputs_ind
//                         } // x

//                         inputs_pointer += XW;
//                     } // y

//                     d_pooled_outputs_pointer++;
//                 } // w
//             } // h
//         } // nkc
  
//   }
// }



// void convolve_gradient_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_FILTERS, float *SCRATCH){

//   #pragma offload target(mic:MIC_DEV) \ 
//   in(INPUTS:length(0) REUSE) \ 
//   in(FILTERS:length(0) REUSE) \ 
//   in(ARGMAXS:length(0) REUSE) \
//   in(D_POOLED_OUTPUTS:length(0) REUSE) \
//   in(D_FILTERS:length(0) REUSE)
//   {
//         int nkc, ncHW, h, w, k, c, lin_index, y, x, h_arg, w_arg, n, tmp;
//         int XW = -X_const + W_const,
//                     HYW = (H_const-Y_const)*W_const;

//         #pragma omp parallel for \
//         default(none) \
//         schedule(dynamic, N) \ 
//         private(nkc, ncHW, h, w, k, c, lin_index, y, x, h_arg, w_arg, n, tmp) \
//         shared(N, INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, XW, HYW)

//         for (nkc = 0; nkc < N*K_const*C_const; nkc++){
            
//             k = nkc/(N*C_const);
//             c = md(nkc, N*C_const)/N;
//             n = md(nkc, N);

//             ncHW = (n*C_const + c)*H_const*W_const;

//             float *restrict d_pooled_outputs_pointer = D_POOLED_OUTPUTS + (n*K_const + k)*pooled_H_const*pooled_W_const;
//             int *restrict argmaxs_pointer = ARGMAXS + (n*K_const + k)*pooled_H_const*pooled_W_const - 1;

//             int kcyx_shift = (k*C_const + c)*Y_const*X_const - 1;

//             for (h = 0; h < pooled_H_const; h++){
//                 for (w = 0; w < pooled_W_const; w++){
                    
//                     lin_index = *(++argmaxs_pointer);
//                     // tmp = output_H_const*output_W_const; tmp = md(lin_index, tmp);
//                     h_arg = lin_index/output_W_const;
//                     w_arg = lin_index - h_arg*output_W_const;

//                     float *restrict d_filters_pointer = D_FILTERS + kcyx_shift;
//                     float *restrict inputs_pointer = INPUTS + w_arg + h_arg*W_const + ncHW - 1;

//                     for (y = 0; y < Y_const; y++){
//                         for (x = 0; x < X_const; x++){
//                             *(++d_filters_pointer) += (*d_pooled_outputs_pointer) * (*(++inputs_pointer)); // kcyx += nkhw * inputs_ind
//                         } // x

//                         inputs_pointer += XW;
//                     } // y

//                     d_pooled_outputs_pointer++;
//                 } // w
//             } // h
//         } // nkc
  
//   }
// }

// void convolve_gradient_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){

//   #pragma offload target(mic:MIC_DEV) \ 
//   in(INPUTS:length(0) REUSE) \ 
//   in(FILTERS:length(0) REUSE) \ 
//   in(ARGMAXS:length(0) REUSE) \
//   in(D_POOLED_OUTPUTS:length(0) REUSE) \
//   in(D_FILTERS:length(0) REUSE)
//   {
//         int kcyx, ncHW, h, w, k, c, lin_index, y, x, h_arg, w_arg, n, tmp;
//         int XW = -X_const + W_const,
//                     HYW = (H_const-Y_const)*W_const;

//         #pragma omp parallel for \
//         default(none) \
//         schedule(dynamic, X_const*Y_const) \
//         private(kcyx, ncHW, h, w, k, c, lin_index, y, x, h_arg, w_arg, n, tmp) \
//         shared(N, INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, XW, HYW)

//         for (kcyx = 0; kcyx < K_const*C_const*Y_const*X_const; kcyx++){
//             float *restrict d_filters_pointer = D_FILTERS + kcyx;
//             float value = 0.f;
//             k = kcyx / (C_const*Y_const*X_const);
//             c = md(kcyx, C_const*Y_const*X_const) / (Y_const*X_const);

//             for (n = 0; n < N; n++){
//                 // float *restrict inputs_pointer = INPUTS + (n*C_const + c)*H_const*W_const;
//                 float *restrict d_pooled_outputs_pointer = D_POOLED_OUTPUTS + (n*K_const + k)*pooled_H_const*pooled_W_const - 1;
//                 int *restrict argmaxs_pointer = ARGMAXS + (n*K_const + k)*pooled_H_const*pooled_W_const - 1;
//                 float *restrict inputs_pointer = INPUTS + (n*C_const + c)*H_const*W_const;

//                 for (h = 0; h < pooled_H_const; h++){
//                     for (w = 0; w < pooled_W_const; w++){
//                         lin_index = *(++argmaxs_pointer);
//                         h_arg = lin_index/output_W_const;
//                         w_arg = lin_index - h_arg*output_W_const;

//                         value += (*(++d_pooled_outputs_pointer)) * (*(inputs_pointer + w_arg + h_arg*W_const));
//                     }
//                 }
//             }
//             *d_filters_pointer = value;

//         }
  
//     }
// }


// void convolve_gradient_layer2(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){

//   // const int output_H_const = H_const2 - Y_const2 + 1;
//   // const2 int output_W_const2 = W_const2 - X_const2 + 1;
//   // const2 int pooled_H_const2 = ceil(((float) output_H_const2)/pooling_radius);
//   // const2 int pooled_W_const2 = ceil(((float) output_W_const2)/pooling_radius);

//   #pragma offload target(mic:MIC_DEV) \ 
//   in(INPUTS:length(0) REUSE) \ 
//   in(FILTERS:length(0) REUSE) \ 
//   in(ARGMAXS:length(0) REUSE) \
//   in(D_INPUTS:length(0) REUSE) \
//   in(D_POOLED_OUTPUTS:length(0) REUSE) \
//   in(D_FILTERS:length(0) REUSE)
//   {
//         int nkc, nkhw, ncHW, h, w, k, c, kcyx, lin_index, inputs_ind, inputs_ind_shift, y, x, h_arg, w_arg, n, tmp;
//         #pragma omp parallel for \
//         default(none) \
//         private(nkc, nkhw, kcyx, ncHW, h, w, k, c, lin_index, inputs_ind, inputs_ind_shift, y, x, h_arg, w_arg, n, tmp) \
//         shared(N, INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_INPUTS, D_FILTERS)
//       // loop over elements of filter
//         for (nkc = 0; nkc < N*K_const2*C_const2; nkc++){
//             n = nkc/(K_const2*C_const2);
//             k = md(nkc, K_const2*C_const2)/C_const2;
//             c = md(nkc, C_const2);
//             nkhw = (n*K_const2 + k)*pooled_H_const2*pooled_W_const2;
//             ncHW = (n*C_const2 + c)*H_const2*W_const2;

//             for (h = 0; h < pooled_H_const2; h++){
//                 for (w = 0; w < pooled_W_const2; w++){
//                     kcyx = (k*C_const2 + c)*Y_const2*X_const2;

//                     lin_index = ARGMAXS[nkhw];
//                     tmp = output_H_const2*output_W_const2; tmp = md(lin_index, tmp);
//                     h_arg = tmp/output_W_const2;
//                     w_arg = tmp - h_arg*output_W_const2;
//                     inputs_ind_shift = w_arg + h_arg*W_const2 + ncHW;

//                     for (y = 0; y < Y_const2; y++){
//                         for (x = 0; x < X_const2; x++){

//                             inputs_ind = x + y*W_const2 + inputs_ind_shift;
                            
//                             D_FILTERS[kcyx] += D_POOLED_OUTPUTS[nkhw] * INPUTS[inputs_ind];
//                             D_INPUTS[inputs_ind] += D_POOLED_OUTPUTS[nkhw] * FILTERS[kcyx];

//                             kcyx++;
//                         }
//                     }

//                     nkhw++;
//                 }
//             }

//         }
  
//   }
// }

void convolve_gradient_layer2(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){

  #pragma offload target(mic:MIC_DEV) \ 
  in(INPUTS:length(0) REUSE) \ 
  in(D_INPUTS:length(0) REUSE) \ 
  in(FILTERS:length(0) REUSE) \ 
  in(ARGMAXS:length(0) REUSE) \
  in(D_POOLED_OUTPUTS:length(0) REUSE) \
  in(D_FILTERS:length(0) REUSE) \
  in(SCRATCH:length(0) REUSE)
  {
        int nkc, ncHW, h, w, k, c, lin_index, y, x, h_arg, w_arg, n, tmp;
        int XW = -X_const2 + W_const2,
                    HYW = (H_const2-Y_const2)*W_const2;

        #pragma omp parallel for \
        default(none) \
        schedule(dynamic, C_const2) \ 
        private(nkc, ncHW, h, w, k, c, lin_index, y, x, h_arg, w_arg, n, tmp) \
        shared(N, INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, D_INPUTS, SCRATCH, XW, HYW)

        for (nkc = 0; nkc < N*K_const2*C_const2; nkc++){
            n = nkc/(K_const2*C_const2);
            k = md(nkc, K_const2*C_const2)/C_const2;
            c = md(nkc, C_const2);

            // k = nkc/(N*C_const2);
            // n = md(nkc, N*C_const2)/C_const2;
            // c = md(nkc, C_const2);

            // n = nkc/(K_const2*C_const2);
            // c = md(nkc, K_const2*C_const2)/K_const2;
            // k = md(nkc, K_const2);
            
            // k = nkc/(N*C_const2);
            // c = md(nkc, N*C_const2)/N;
            // n = md(nkc, N);

            ncHW = (n*C_const2 + c)*H_const2*W_const2;

            float *restrict d_pooled_outputs_pointer = D_POOLED_OUTPUTS + (n*K_const2 + k)*pooled_H_const2*pooled_W_const2;
            int *restrict argmaxs_pointer = ARGMAXS + (n*K_const2 + k)*pooled_H_const2*pooled_W_const2 - 1;

            float *d_filters_tmp = SCRATCH + 2*omp_get_thread_num()*Y_const2*X_const2 - 1;
            for (y = 0; y < Y_const2; y++){
                for (x = 0; x < X_const2; x++){
                    *(++d_filters_tmp) = 0.f;
                } // x
            } // y
            d_filters_tmp = SCRATCH + 2*omp_get_thread_num()*Y_const2*X_const2 - 1;
            
            float *d_inputs_tmp = SCRATCH + 240*2*Y_const2*X_const2 + 2*omp_get_thread_num()*H_const2*W_const2 - 1;
            for (h = 0; h < H_const2; h++){
                for (w = 0; w < W_const2; w++){
                    *(++d_inputs_tmp) = 0.f;
                } // w
            } // h
            d_inputs_tmp = SCRATCH + 240*2*Y_const2*X_const2 + 2*omp_get_thread_num()*H_const2*W_const2 - 1;


            int kcyx_shift = (k*C_const2 + c)*Y_const2*X_const2 - 1;

            for (h = 0; h < pooled_H_const2; h++){
                for (w = 0; w < pooled_W_const2; w++){
                    
                    lin_index = *(++argmaxs_pointer);
                    h_arg = lin_index/output_W_const2;
                    w_arg = lin_index - h_arg*output_W_const2;

                    float *restrict filters_pointer = FILTERS + kcyx_shift;
                    // float *restrict d_filters_pointer = D_FILTERS + kcyx_shift;
                    float *restrict d_filters_pointer = d_filters_tmp;

                    if (w_arg - padding_const2 >= 0 &&
                        h_arg - padding_const2 >= 0 &&
                        output_W_const2 - 1 - w_arg >= padding_const2  &&
                        output_H_const2 - 1 - h_arg >= padding_const2){
                        
                        float *restrict inputs_pointer = INPUTS + (w_arg-padding_const2) + (h_arg-padding_const2)*W_const2 + ncHW - 1;
                        // float *restrict d_inputs_pointer = D_INPUTS + (w_arg-padding_const2) + (h_arg-padding_const2)*W_const2 + ncHW - 1;
                        float *restrict d_inputs_pointer = d_inputs_tmp + (w_arg-padding_const2) + (h_arg-padding_const2)*W_const2 - 1;
                        
                        for (y = 0; y < Y_const2; y++){
                            for (x = 0; x < X_const2; x++){
                                *(++d_filters_pointer) += (*d_pooled_outputs_pointer) * (*(++inputs_pointer)); // kcyx += nkhw * inputs_ind
                                *(++d_inputs_pointer) += (*d_pooled_outputs_pointer) * (*(++filters_pointer));
                            } // x
                            inputs_pointer += XW;
                            d_inputs_pointer += XW;

                        } // y
                    }

                    else{
                        float *restrict inputs_pointer = INPUTS + ncHW - 1 + mx(mn(h_arg-padding_const2, H_const2-1), 0)*W_const2 + mx(mn(w_arg-padding_const2, W_const2-1), 0);
                        float *restrict d_inputs_pointer = d_inputs_tmp - 1 + mx(mn(h_arg-padding_const2, H_const2-1), 0)*W_const2 + mx(mn(w_arg-padding_const2, W_const2-1), 0);
                        // float *restrict d_inputs_pointer = D_INPUTS + ncHW - 1 + mx(mn(h_arg-padding_const2, H_const2-1), 0)*W_const2 + mx(mn(w_arg-padding_const2, W_const2-1), 0);

                        for (y = 0; y < Y_const2; ++y){
                            float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                            float *restrict d_inputs_pointer_ncyX = d_inputs_pointer;
                            if ((y + h_arg - padding_const2 >= 0) && (y + h_arg - padding_const2 < H_const2)){
                                for (x = 0; x < X_const2; ++x){
                                    filters_pointer++;
                                    d_filters_pointer++;
                                    if ((x + w_arg - padding_const2 >= 0) && (x + w_arg - padding_const2 < W_const2))
                                        *d_filters_pointer += (*d_pooled_outputs_pointer) * (*(++inputs_pointer));
                                        *(++d_inputs_pointer) += (*d_pooled_outputs_pointer) * (*filters_pointer); 
                                } // x
                            }

                            else { // advance pointer without going into loop
                                filters_pointer += X_const2;
                                d_filters_pointer += X_const2;
                            } 

                            inputs_pointer = inputs_pointer_ncyX + W_const2; 
                            d_inputs_pointer = d_inputs_pointer_ncyX + W_const2;
                        } // y

                        // for (y = 0; y < Y_const2; y++){
                        //     float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                        //     float *restrict d_inputs_pointer_ncyX = d_inputs_pointer;
                        //     for (x = 0; x < X_const2; x++){
                        //         filters_pointer++;
                        //         d_filters_pointer++;
                        //         if ((y + h_arg - padding_const2 >= 0) && (y + h_arg - padding_const2 < H_const2) && (x + w_arg - padding_const2 >= 0) && (x + w_arg - padding_const2 < W_const2)){
                        //             *d_filters_pointer += (*d_pooled_outputs_pointer) * (*(++inputs_pointer)); // kcyx += nkhw * inputs_ind
                        //             *(++d_inputs_pointer) += (*d_pooled_outputs_pointer) * (*filters_pointer);
                        //         }
                        //     } // x

                        //     inputs_pointer = inputs_pointer_ncyX + W_const2;
                        //     d_inputs_pointer = d_inputs_pointer_ncyX + W_const2;
                        // } // y
                    }
                    d_pooled_outputs_pointer++;
                } // w
            } // h

            float *restrict d_filters_pointer = D_FILTERS + kcyx_shift;
            for (y = 0; y < Y_const2; y++){
                for (x = 0; x < X_const2; x++){
                    *(++d_filters_pointer) += *(++d_filters_tmp);
                } // x
            } // y

            float *restrict d_inputs_pointer = D_INPUTS + ncHW - 1;
            for (h = 0; h < H_const2; h++){
                for (w = 0; w < W_const2; w++){
                    *(++d_inputs_pointer) += *(++d_inputs_tmp);
                } // w
            } // h
        } // nkc
  
    }
}

// #pragma offload:
#pragma offload_attribute(push,target(mic))
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
#include <cilk/cilk_api.h>
#pragma offload_attribute(pop)


// void local_layer1(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int stride, int padding, int offloaded, float *SCRATCH){
//     // __cilkrts_set_param("nworkers","236");

//     assert(C == C_constL);
//     assert(H == H_constL);
//     assert(W == W_constL);
//     assert(K == K_constL);
//     assert(stride == stride_constL);
//     assert(padding == padding_constL);
//     assert(X == X_constL);
//     assert(Y == Y_constL);
//     assert(output_H_constL == (H_constL + 2*padding_constL - Y_constL + 1)/stride_constL);
//     assert(output_W_constL == (W_constL + 2*padding_constL - X_constL + 1)/stride_constL);

//     #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
//     in(INPUTS:length(0) REUSE) \ 
//     in(FILTERS:length(0) REUSE) \ 
//     in(OUTPUTS:length(0) REUSE) \
//     in(SCRATCH:length(0) REUSE) 
//     {   
//     //  __cilkrts_end_cilk();
//     // if (0!= __cilkrts_set_param("nworkers", "236"))
//     // {
//     // printf("Failed to set worker count\n");
//     // exit;
//     //  }
    
//         float convolution;
//         int nk, n, k, h, w, c, y, x;
//         // __assume_aligned(INPUTS, ALIGN);
//         // __assume_aligned(FILTERS, ALIGN);
//         // __assume_aligned(OUTPUTS, ALIGN);

//         // computation of constLants
//         int XW = -X_constL + W_constL,
//                     HYW = (H_constL-Y_constL)*W_constL;

//         int *restrict indices = malloc(C_constL*Y_constL*X_constL*sizeof(int));
//         int *restrict indices_pointer = indices - 1;
//         int index = -1;
//         for (c = 0; c < C_constL; ++c){
//             for (y = 0; y < Y_constL; ++y){                                
//                 for (x = 0; x < X_constL; ++x){
//                     *(++indices_pointer) = ++index;
//                     // printf("%d \n", *indices_pointer);
//                     } // x
//                     index += XW;
//             } // y
//             index += HYW;
//         } // c
        
        
//         // for each example, for each filter, for each pooling region...
        
//         float (*restrict inputs_4D)[C_constL][H_constL][W_constL] = (float (*)[C_constL][H_constL][W_constL]) INPUTS;
//         float (*restrict outputs_4D)[K_constL][output_H_constL][output_W_constL] = (float (*)[K_constL][output_H_constL][output_W_constL]) OUTPUTS;
//         float (*restrict filters_6D)[output_H_constL][output_W_constL][C_constL][Y_constL][X_constL] = (float (*)[output_H_constL][output_W_constL][C_constL][Y_constL][X_constL]) FILTERS;

        
//         // __cilkrts_init();
//          // printf("%d, %d \n", __cilkrts_get_worker_number(), __cilkrts_get_nworkers());
//         // #pragma vector aligned
//         // #pragma ivdep
//         #pragma simd
//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(nk, n, k, h, w, c, y, x, convolution) \
//             shared(N, INPUTS, OUTPUTS, FILTERS, XW, HYW, indices, inputs_4D, outputs_4D, filters_6D)
         
//         // exit(0);
//             // cilk_for (int nk = 0; nk < N*K_constL; nk++){
//          // #pragma cilk grainsize = 1
//         for (nk = 0; nk < N*K_constL; nk++){
//             // n = nkhw / (K_constL*output_H_constL*output_W_constL);
//             // k = md(nkhw, K_constL*output_H_constL*output_W_constL) / (output_H_constL*output_W_constL);
//             // h = md(nkhw, output_H_constL*output_W_constL) / (output_W_constL);
//             // w = md(nkhw, output_W_constL);
//             // printf("%d\n", omp_get_thread_num());
//             // cilk_for (int k = 0; k < K_constL; k++){

//             n = nk / K_constL;
//             k = md(nk, K_constL);
            
//             // int khwcyx_shift = k*output_H_constL*output_W_constL*C_constL*Y_constL*X_constL - 1, // filters
//                 // nchw_shift = (n*C_constL*H_constL)*W_constL - 1; // inputs

//             // float *restrict outputs_pointer = OUTPUTS + nk*output_H_constL*output_W_constL - 1;
//             // int khwcyx_shift = k*output_H_constL*output_W_constL*C_constL*Y_constL*X_constL, // filters
//             //     nchw_shift = (n*C_constL*H_constL)*W_constL; // inputs

//             // float *restrict filters_pointer = FILTERS + khwcyx_shift;
            
                
//             for (int h = 0; h < output_H_constL; h++){
//                 for (int w = 0; w < output_W_constL; w++){
//                     // float *restrict inputs_pointer = INPUTS + nchw_shift + h*W_constL + w;
                       
//                     if (w - padding_constL >= 0 &&
//                         h - padding_constL >= 0 &&
//                         output_W_constL - 1 - w >= padding_constL  &&
//                         output_H_constL - 1 - h >= padding_constL){
//                         outputs_4D[n][k][h][w] = __sec_reduce_add(filters_6D[n][k][h][:][:][:] * inputs_4D[n][:][(h - padding_constL):(h - padding_constL)+Y_constL][(w - padding_constL):(w - padding_constL)+X_constL]);

//                         // float *restrict inputs_pointer = INPUTS + nchw_shift + (h - padding_constL)*W_constL + (w - padding_constL);

//                         // convolution = __sec_reduce_add(inputs_pointer[indices[0:C_constL*Y_constL*X_constL]] * filters_pointer[0:C_constL*Y_constL*X_constL]);
//                         // filters_pointer += C_constL*Y_constL*X_constL;

//                     }
//                     else{
//                         int h_start = mx(h - padding_constL, 0);
//                         int w_start = mx(w - padding_constL, 0);
//                         int h_end = mn(h - padding_constL + Y_constL, H_const);
//                         int w_end = mn(w - padding_constL + X_constL, W_const);
//                         int y_start = mx(padding_constL - h, 0);
//                         int x_start = mx(padding_constL - w, 0);
//                         int y_end = y_start + (h_end - h_start);
//                         int x_end = x_start + (w_end - w_start);
//                         // printf("%d, %d, %d, %d, \n %d, %d, %d, %d, \n", h, w, h - padding_constL, h - padding_constL + Y_constL, h_start, h_end, w_start, w_end);
//                         // exit(0);
//                         outputs_4D[n][k][h][w] = __sec_reduce_add(filters_6D[n][k][h][:][x_start:x_end][y_start:y_end] * inputs_4D[n][:][h_start:h_end][w_start:w_end]);                        
//                     }

//                     // else{
//                     //     float *restrict inputs_pointer = INPUTS + nchw_shift + mx(mn(h-padding_constL, H_constL-1), 0)*W_constL + mx(mn(w-padding_constL, W_constL-1), 0);
    
//                     //     for (c = 0; c < C_constL; ++c){
//                     //         float *restrict inputs_pointer_ncYX = inputs_pointer;
                            
//                     //         for (y = 0; y < Y_constL; ++y){
                                
//                     //             float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                                
//                     //             if ((y + h - padding_constL >= 0) && (y + h - padding_constL < H_constL)){
//                     //                 for (x = 0; x < X_constL; ++x){
//                     //                     filters_pointer++;
//                     //                     if ((x + w - padding_constL >= 0) && (x + w - padding_constL < W_constL))
//                     //                         convolution += (*(++inputs_pointer)) * (*filters_pointer);
//                     //                 } // x
//                     //             }

//                     //             else{ // advance pointer without going into loop
//                     //                 // inputs_pointer += X_constL;
//                     //                 filters_pointer += X_constL;
//                     //             } 
    
//                     //             inputs_pointer = inputs_pointer_ncyX + W_constL; 
//                     //         } // y
    
//                     //         inputs_pointer = inputs_pointer_ncYX + H_constL*W_constL; 
//                     //     } // c
//                     // }
//                     // OUTPUTS[n][k][h][w] = convolution;
//                     // *(++outputs_pointer) = convolution;
//                 } // w
//             } // h
//         // }
//         } // nk
         
//     } // pragma_offload
// }

void local_layer1(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int stride, int padding, int offloaded, float *SCRATCH){

    assert(C == C_constL);
    assert(H == H_constL);
    assert(W == W_constL);
    assert(K == K_constL);
    assert(stride == stride_constL);
    assert(padding == padding_constL);
    assert(X == X_constL);
    assert(Y == Y_constL);
    assert(output_H_constL == (H_constL + 2*padding_constL - Y_constL + 1)/stride_constL);
    assert(output_W_constL == (W_constL + 2*padding_constL - X_constL + 1)/stride_constL);

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE) 
    {
        float convolution;
        int nk, n, k, h, w, c, y, x;
        // __assume_aligned(INPUTS, ALIGN);
        // __assume_aligned(FILTERS, ALIGN);
        // __assume_aligned(OUTPUTS, ALIGN);

        // computation of constLants
        int XW = -X_constL + W_constL,
                    HYW = (H_constL-Y_constL)*W_constL;

        int *restrict indices = _mm_malloc(C_constL*Y_constL*X_constL*sizeof(int), ALIGN);
        int *restrict indices_pointer = indices - 1;
        int index = -1;
        for (c = 0; c < C_constL; ++c){
            for (y = 0; y < Y_constL; ++y){                                
                for (x = 0; x < X_constL; ++x){
                    *(++indices_pointer) = ++index;
                    // printf("%d \n", *indices_pointer);
                    } // x
                    index += XW;
            } // y
            index += HYW;
        } // c
        
        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nk, n, k, h, w, c, y, x, convolution) \
            shared(N, INPUTS, OUTPUTS, FILTERS, XW, HYW, indices)
        
        // for each example, for each filter, for each pooling region...
        
        for (nk = 0; nk < N*K_constL; nk++){
            float *restrict outputs_pointer = OUTPUTS + nk*output_H_constL*output_W_constL - 1;

            n = nk / K_constL;
            k = md(nk, K_constL);
            
            // int khwcyx_shift = k*output_H_constL*output_W_constL*C_constL*Y_constL*X_constL - 1, // filters
            //     nchw_shift = (n*C_constL*H_constL)*W_constL - 1; // inputs

            int khwcyx_shift = k*output_H_constL*output_W_constL*C_constL*Y_constL*X_constL, // filters
                nchw_shift = (n*C_constL*H_constL)*W_constL; // inputs

            float *restrict filters_pointer = FILTERS + khwcyx_shift;
                
            for (h = 0; h < output_H_constL; h+= stride_constL){
                for (w = 0; w < output_W_constL; w+= stride_constL){
                    convolution = 0.f;
                    // float *restrict inputs_pointer = INPUTS + nchw_shift + h*W_constL + w;
                       
                    if (w - padding_constL >= 0 &&
                        h - padding_constL >= 0 &&
                        output_W_constL - 1 - w >= padding_constL  &&
                        output_H_constL - 1 - h >= padding_constL){

                        float *restrict inputs_pointer = INPUTS + nchw_shift + (h - padding_constL)*W_constL + (w - padding_constL);
                        // for (int cyx = 0; cyx < C_constL*Y_constL*X_constL; ++cyx){
                        //     convolution += (*(++inputs_pointer)) * (*(++filters_pointer));
                        // } // c

                        convolution = __sec_reduce_add(inputs_pointer[indices[0:C_constL*Y_constL*X_constL]] * filters_pointer[0:C_constL*Y_constL*X_constL]);
                        filters_pointer += C_constL*Y_constL*X_constL;
                        // for (c = 0; c < C_constL; ++c){
                        //     for (y = 0; y < Y_constL; ++y){                                
                        //         // #pragma vector aligned
                        //         // #pragma ivdep
                        //         for (x = 0; x < X_constL; ++x){
                        //             convolution += (*(++inputs_pointer)) * (*++filters_pointer);
                        //             // convolution += inputs_pointer[x] * filters_pointer[x];
                        //         } // x

                        //         inputs_pointer += XW;
                        //     } // y
                        //     inputs_pointer += HYW;
                        // } // c

                    }

                    else{
                        float *restrict inputs_pointer = INPUTS + nchw_shift + mx(mn(h-padding_constL, H_constL-1), 0)*W_constL + mx(mn(w-padding_constL, W_constL-1), 0);
    
                        for (c = 0; c < C_constL; ++c){
                            float *restrict inputs_pointer_ncYX = inputs_pointer;
                            
                            for (y = 0; y < Y_constL; ++y){
                                
                                float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                                
                                if ((y + h - padding_constL >= 0) && (y + h - padding_constL < H_constL)){
                                    for (x = 0; x < X_constL; ++x){
                                        if ((x + w - padding_constL >= 0) && (x + w - padding_constL < W_constL))
                                            convolution += (*(inputs_pointer++)) * (*filters_pointer);
                                        filters_pointer++;
                                    } // x
                                }

                                else{ // advance pointer without going into loop
                                    // inputs_pointer += X_constL;
                                    filters_pointer += X_constL;
                                } 
    
                                inputs_pointer = inputs_pointer_ncyX + W_constL; 
                            } // y
    
                            inputs_pointer = inputs_pointer_ncYX + H_constL*W_constL; 
                        } // c
                    }
                    // loop over convolution elements
                    // for (c = 0; c < C_constL; ++c){
                    //     for (y = 0; y < Y_constL; ++y){
                    //         for (x = 0; x < X_constL; ++x){
                    //             convolution += (*(++inputs_pointer)) * (*(++filters_pointer));
                    //         }

                    //         inputs_pointer += XW;
                    //     }

                    //     inputs_pointer += HYW;
                    // }

                    *(++outputs_pointer) = convolution;
                } // w
            } // h

        } // nk
         
    } // pragma_offload
}

// void local_layer1(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int stride, int padding, int offloaded, float *SCRATCH){

//     assert(C == C_constL);
//     assert(H == H_constL);
//     assert(W == W_constL);
//     assert(K == K_constL);
//     assert(stride == stride_constL);
//     assert(padding == padding_constL);
//     assert(X == X_constL);
//     assert(Y == Y_constL);
//     assert(output_H_constL == (H_constL + 2*padding_constL - Y_constL + 1)/stride_constL);
//     assert(output_W_constL == (W_constL + 2*padding_constL - X_constL + 1)/stride_constL);

//     #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
//     in(INPUTS:length(0) REUSE) \ 
//     in(FILTERS:length(0) REUSE) \ 
//     in(OUTPUTS:length(0) REUSE)
//     {
//         float convolution;
//         int nkhw, n, k, h, w, c, y, x;

//         // computation of constLants
//         int XW = -X_constL + W_constL,
//                     HYW = (H_constL-Y_constL)*W_constL;


//         #pragma vector aligned
//         #pragma ivdep
//         #pragma simd
//         #pragma omp parallel for \
//             schedule(dynamic, output_W_const) \
//             default(none) \
//             private(nkhw, n, k, h, w, c, y, x, convolution) \
//             shared(N, INPUTS, OUTPUTS, FILTERS, XW, HYW)
        
//         // for each example, for each filter, for each pooling region...
//         for (nkhw = 0; nkhw < N*K_constL*output_H_const*output_W_const; nkhw++){
//             float *restrict outputs_pointer = OUTPUTS + nkhw;

//             n = nkhw / (K_constL*output_H_const*output_W_const);
//             k = md(nkhw, K_constL*output_H_const*output_W_const) / (output_H_const*output_W_const);
//             h = md(nkhw, output_H_const*output_W_const) / output_W_const;
//             h = md(nkhw, output_W_const);
            
//             int khwcyx_shift = md(nkhw, K_constL*output_H_const*output_W_const) *C_constL*Y_constL*X_constL - 1, // filters
//                 nchw_shift = (n*C_constL*H_constL)*W_constL - 1; // inputs

//             float *restrict filters_pointer = FILTERS + khwcyx_shift;
                
//             convolution = 0.f;
//             // float *restrict inputs_pointer = INPUTS + nchw_shift + h*W_constL + w;
               
//             if (w - padding_constL >= 0 &&
//                 h - padding_constL >= 0 &&
//                 output_W_constL - 1 - w >= padding_constL  &&
//                 output_H_constL - 1 - h >= padding_constL){

//                 float *restrict inputs_pointer = INPUTS + nchw_shift + (h - padding_constL)*W_constL + (w - padding_constL);
//                 for (c = 0; c < C_constL; ++c){
//                     for (y = 0; y < Y_constL; ++y){                                
//                         for (x = 0; x < X_constL; ++x){
//                             convolution += (*(++inputs_pointer)) * (*(++filters_pointer));
//                         } // x

//                         inputs_pointer += XW;
//                     } // y
//                     inputs_pointer += HYW;
//                 } // c

//             }

//             else{
//                 float *restrict inputs_pointer = INPUTS + nchw_shift + mx(mn(h-padding_constL, H_constL-1), 0)*W_constL + mx(mn(w-padding_constL, W_constL-1), 0);

//                 for (c = 0; c < C_constL; ++c){
//                     float *restrict inputs_pointer_ncYX = inputs_pointer;
                    
//                     for (y = 0; y < Y_constL; ++y){
                        
//                         float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                        
//                         if ((y + h - padding_constL >= 0) && (y + h - padding_constL < H_constL)){
//                             for (x = 0; x < X_constL; ++x){
//                                 filters_pointer++;
//                                 if ((x + w - padding_constL >= 0) && (x + w - padding_constL < W_constL))
//                                     convolution += (*(++inputs_pointer)) * (*filters_pointer);
//                             } // x
//                         }

//                         else{ // advance pointer without going into loop
//                             // inputs_pointer += X_constL;
//                             filters_pointer += X_constL;
//                         } 

//                         inputs_pointer = inputs_pointer_ncyX + W_constL; 
//                     } // y

//                     inputs_pointer = inputs_pointer_ncYX + H_constL*W_constL; 
//                 } // c
//             }
//             // loop over convolution elements
//             // for (c = 0; c < C_constL; ++c){
//             //     for (y = 0; y < Y_constL; ++y){
//             //         for (x = 0; x < X_constL; ++x){
//             //             convolution += (*(++inputs_pointer)) * (*(++filters_pointer));
//             //         }

//             //         inputs_pointer += XW;
//             //     }

//             //     inputs_pointer += HYW;
//             // }

//             *(outputs_pointer) = convolution;

//         } // nkhw
         
//     } // pragma_offload
// }

void local_gradient_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, float *D_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){

  #pragma offload target(mic:MIC_DEV) \ 
  in(INPUTS:length(0) REUSE) \ 
  in(D_INPUTS:length(0) REUSE) \ 
  in(FILTERS:length(0) REUSE) \ 
  in(D_OUTPUTS:length(0) REUSE) \
  in(D_FILTERS:length(0) REUSE) \
  in(SCRATCH:length(0) REUSE)
  {
        int nkc, ncHW, h, w, k, c, y, x, n;
        int XW = -X_constL + W_constL,
                    HYW = (H_constL-Y_constL)*W_constL;

        #pragma omp parallel for \
        default(none) \
        schedule(dynamic, N) \ 
        private(nkc, ncHW, h, w, k, c, y, x, n) \
        shared(N, INPUTS, FILTERS, D_OUTPUTS, D_FILTERS, D_INPUTS, XW, HYW, SCRATCH)

        for (nkc = 0; nkc < N*K_constL*C_constL; nkc++){
            // n = nkc/(K_constL*C_constL);
            // k = md(nkc, K_constL*C_constL)/C_constL;
            // c = md(nkc, C_constL);

            // k = nkc/(N*C_constL);
            // n = md(nkc, N*C_constL)/C_constL;
            // c = md(nkc, C_constL);

            // n = nkc/(K_constL*C_constL);
            // c = md(nkc, K_constL*C_constL)/K_constL;
            // k = md(nkc, K_constL);
            
            k = nkc/(N*C_constL);
            c = md(nkc, N*C_constL)/N;
            n = md(nkc, N);

            ncHW = (n*C_constL + c)*H_constL*W_constL;
            
            // float *d_inputs_tmp = SCRATCH + 2*omp_get_thread_num()*H_constL*W_constL - 1;
            // for (h = 0; h < H_constL; h++){
            //     for (w = 0; w < W_constL; w++){
            //         *(++d_inputs_tmp) = 0.f;
            //     } // w
            // } // h
            // d_inputs_tmp = SCRATCH + 2*omp_get_thread_num()*H_constL*W_constL - 1;


            float *restrict d_outputs_pointer = D_OUTPUTS + (n*K_constL + k)*output_H_constL*output_W_constL;

            int khwcyx_shift = (k*output_H_constL*output_W_constL*C_constL + c)*Y_constL*X_constL - 1;


            for (h = 0; h < output_H_constL; h++){
                for (w = 0; w < output_W_constL; w++){

                    float *restrict filters_pointer = FILTERS + khwcyx_shift + (h*output_W_constL + w)*C_constL*Y_constL*X_constL;
                    float *restrict d_filters_pointer = D_FILTERS + khwcyx_shift + (h*output_W_constL + w)*C_constL*Y_constL*X_constL;

                    if (w - padding_constL >= 0 &&
                        h - padding_constL >= 0 &&
                        output_W_constL - 1 - w >= padding_constL  &&
                        output_H_constL - 1 - h >= padding_constL){
                        
                        float *restrict inputs_pointer = INPUTS + (w-padding_constL) + (h-padding_constL)*W_constL + ncHW - 1;
                        float *restrict d_inputs_pointer = D_INPUTS + (w-padding_constL) + (h-padding_constL)*W_constL + ncHW - 1;
                        // float *restrict d_inputs_pointer = d_inputs_tmp + (w-padding_constL) + (h-padding_constL)*W_constL - 1;
                        
                        for (y = 0; y < Y_constL; y++){
                            for (x = 0; x < X_constL; x++){
                                *(++d_filters_pointer) += (*d_outputs_pointer) * (*(++inputs_pointer)); // kcyx += nkhw * inputs_ind
                                *(++d_inputs_pointer) += (*d_outputs_pointer) * (*(++filters_pointer));
                            } // x
                            inputs_pointer += XW;
                            d_inputs_pointer += XW;

                        } // y
                    }

                    else{
                        float *restrict inputs_pointer = INPUTS + ncHW - 1 + mx(mn(h-padding_constL, H_constL-1), 0)*W_constL + mx(mn(w-padding_constL, W_constL-1), 0);
                        // float *restrict d_inputs_pointer = d_inputs_tmp - 1 + mx(mn(h-padding_constL, H_constL-1), 0)*W_constL + mx(mn(w-padding_constL, W_constL-1), 0);
                        float *restrict d_inputs_pointer = D_INPUTS + ncHW - 1 + mx(mn(h-padding_constL, H_constL-1), 0)*W_constL + mx(mn(w-padding_constL, W_constL-1), 0);

                        for (y = 0; y < Y_constL; ++y){
                            float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                            float *restrict d_inputs_pointer_ncyX = d_inputs_pointer;
                            if ((y + h - padding_constL >= 0) && (y + h - padding_constL < H_constL)){
                                for (x = 0; x < X_constL; ++x){
                                    filters_pointer++;
                                    d_filters_pointer++;
                                    if ((x + w - padding_constL >= 0) && (x + w - padding_constL < W_constL)){
                                        *d_filters_pointer += (*d_outputs_pointer) * (*(++inputs_pointer));
                                        *(++d_inputs_pointer) += (*d_outputs_pointer) * (*filters_pointer); 
                                    }
                                } // x
                            }

                            else { // advance pointer without going into loop
                                filters_pointer += X_constL;
                                d_filters_pointer += X_constL;
                            } 
                            inputs_pointer = inputs_pointer_ncyX + W_constL; 
                            d_inputs_pointer = d_inputs_pointer_ncyX + W_constL;
                        } // y

                        // for (y = 0; y < Y_constL; y++){
                        //     float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                        //     float *restrict d_inputs_pointer_ncyX = d_inputs_pointer;
                        //     for (x = 0; x < X_constL; x++){
                        //         filters_pointer++;
                        //         d_filters_pointer++;
                        //         if ((y + h - padding_constL >= 0) && (y + h - padding_constL < H_constL) && (x + w - padding_constL >= 0) && (x + w - padding_constL < W_constL)){
                        //             *d_filters_pointer += (*d_outputs_pointer) * (*(++inputs_pointer)); // kcyx += nkhw * inputs_ind
                        //             *(++d_inputs_pointer) += (*d_outputs_pointer) * (*filters_pointer);
                        //         }
                        //     } // x

                        //     inputs_pointer = inputs_pointer_ncyX + W_constL;
                        //     d_inputs_pointer = d_inputs_pointer_ncyX + W_constL;
                        // } // y
                    }
                    // float *restrict inputs_pointer = INPUTS + w + h*W_constL + ncHW - 1;
                    // float *restrict d_inputs_pointer = D_INPUTS + w + h*W_constL + ncHW - 1;

                    // for (y = 0; y < Y_constL; y++){
                    //     for (x = 0; x < X_constL; x++){
                    //         *(++d_filters_pointer) += (*d_outputs_pointer) * (*(++inputs_pointer)); // kcyx += nkhw * inputs_ind
                    //         *(++d_inputs_pointer) += (*d_outputs_pointer) * (*(++filters_pointer));
                    //     } // x

                    //     inputs_pointer += XW;
                    //     d_inputs_pointer += XW;
                    // } // y

                    d_outputs_pointer++;
                } // w
            } // h

            // float *restrict d_inputs_pointer = D_INPUTS + ncHW - 1;
            // for (h = 0; h < H_constL; h++){
            //     for (w = 0; w < W_constL; w++){
            //         *(++d_inputs_pointer) += *(++d_inputs_tmp);
            //     } // w
            // } // h
        } // nkc
  
  }
}

void local_layer2(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int stride, int padding, int offloaded, float *SCRATCH){

    assert(C == C_constL2);
    assert(H == H_constL2);
    assert(W == W_constL2);
    assert(K == K_constL2);
    assert(stride == stride_constL2);
    assert(padding == padding_constL2);
    assert(X == X_constL2);
    assert(Y == Y_constL2);
    assert(output_H_constL2 == (H_constL2 + 2*padding_constL2 - Y_constL2 + 1)/stride_constL2);
    assert(output_W_constL2 == (W_constL2 + 2*padding_constL2 - X_constL2 + 1)/stride_constL2);

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE)
    {
        float convolution = 0.f;
        int nk, n, k, h, w, c, y, x;

        // computation of constL2ants
        int XW = -X_constL2 + W_constL2,
                    HYW = (H_constL2-Y_constL2)*W_constL2;

        int *restrict indices = _mm_malloc(C_constL2*Y_constL2*X_constL2*sizeof(int), ALIGN);
        int *restrict indices_pointer = indices - 1;
        int index = -1;
        for (c = 0; c < C_constL2; ++c){
            for (y = 0; y < Y_constL2; ++y){                                
                for (x = 0; x < X_constL2; ++x){
                    *(++indices_pointer) = ++index;
                    // printf("%d \n", *indices_pointer);
                    } // x
                    index += XW;
            } // y
            index += HYW;
        } // c

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nk, n, k, h, w, c, y, x, convolution) \
            shared(N, INPUTS, OUTPUTS, FILTERS, XW, HYW, indices)
        
        // for each example, for each filter, for each pooling region...
        for (nk = 0; nk < N*K_constL2; nk++){
            float *restrict outputs_pointer = OUTPUTS + nk*output_H_constL2*output_W_constL2 - 1;

            n = nk / K_constL2;
            k = md(nk, K_constL2);
            
            // int khwcyx_shift = k*output_H_constL2*output_W_constL2*C_constL2*Y_constL2*X_constL2 - 1, // filters
            //     nchw_shift = (n*C_constL2*H_constL2)*W_constL2 - 1; // inputs

            int khwcyx_shift = k*output_H_constL2*output_W_constL2*C_constL2*Y_constL2*X_constL2, // filters
                nchw_shift = (n*C_constL2*H_constL2)*W_constL2; // inputs

            float *restrict filters_pointer = FILTERS + khwcyx_shift;
                
            for (h = 0; h < output_H_constL2; h+= stride_constL2){
                for (w = 0; w < output_W_constL2; w+= stride_constL2){
                    convolution = 0.f;
                    // float *restrict inputs_pointer = INPUTS + nchw_shift + h*W_constL2 + w;
                       
                    if (w - padding_constL2 >= 0 &&
                        h - padding_constL2 >= 0 &&
                        output_W_constL2 - 1 - w >= padding_constL2  &&
                        output_H_constL2 - 1 - h >= padding_constL2){

                        float *restrict inputs_pointer = INPUTS + nchw_shift + (h - padding_constL2)*W_constL2 + (w - padding_constL2);

                        convolution = __sec_reduce_add(inputs_pointer[indices[0:C_constL2*Y_constL2*X_constL2]] * filters_pointer[0:C_constL2*Y_constL2*X_constL2]);
                        filters_pointer += C_constL2*Y_constL2*X_constL2;
                        // for (c = 0; c < C_constL2; ++c){
                        //     for (y = 0; y < Y_constL2; ++y){                                
                        //         for (x = 0; x < X_constL2; ++x){
                        //             convolution += (*(++inputs_pointer)) * (*(++filters_pointer));
                        //         } // x

                        //         inputs_pointer += XW;
                        //     } // y
                        //     inputs_pointer += HYW;
                        // } // c

                    }

                    else{
                        float *restrict inputs_pointer = INPUTS + nchw_shift + mx(mn(h-padding_constL2, H_constL2-1), 0)*W_constL2 + mx(mn(w-padding_constL2, W_constL2-1), 0);
    
                        for (c = 0; c < C_constL2; ++c){
                            float *restrict inputs_pointer_ncYX = inputs_pointer;
                            
                            for (y = 0; y < Y_constL2; ++y){
                                
                                float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                                
                                if ((y + h - padding_constL2 >= 0) && (y + h - padding_constL2 < H_constL2)){
                                    for (x = 0; x < X_constL2; ++x){
                                        if ((x + w - padding_constL2 >= 0) && (x + w - padding_constL2 < W_constL2))
                                            convolution += (*(inputs_pointer++)) * (*filters_pointer);

                                        filters_pointer++;
                                    } // x
                                }

                                else{ // advance pointer without going into loop
                                    // inputs_pointer += X_constL2;
                                    filters_pointer += X_constL2;
                                } 
    
                                inputs_pointer = inputs_pointer_ncyX + W_constL2; 
                            } // y
    
                            inputs_pointer = inputs_pointer_ncYX + H_constL2*W_constL2; 
                        } // c
                    }
                    // loop over convolution elements
                    // for (c = 0; c < C_constL2; ++c){
                    //     for (y = 0; y < Y_constL2; ++y){
                    //         for (x = 0; x < X_constL2; ++x){
                    //             convolution += (*(++inputs_pointer)) * (*(++filters_pointer));
                    //         }

                    //         inputs_pointer += XW;
                    //     }

                    //     inputs_pointer += HYW;
                    // }

                    *(++outputs_pointer) = convolution;
                } // w
            } // h

        } // nk
         
    } // pragma_offload
}

void local_gradient_layer2(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, float *D_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){

  #pragma offload target(mic:MIC_DEV) \ 
  in(INPUTS:length(0) REUSE) \ 
  in(D_INPUTS:length(0) REUSE) \ 
  in(FILTERS:length(0) REUSE) \ 
  in(D_OUTPUTS:length(0) REUSE) \
  in(D_FILTERS:length(0) REUSE) \
  in(SCRATCH:length(0) REUSE)
  {
        int nkc, ncHW, h, w, k, c, y, x, n;
        int XW = -X_constL2 + W_constL2,
                    HYW = (H_constL2-Y_constL2)*W_constL2;

        #pragma omp parallel for \
        default(none) \
        schedule(dynamic, N) \ 
        private(nkc, ncHW, h, w, k, c, y, x, n) \
        shared(N, INPUTS, FILTERS, D_OUTPUTS, D_FILTERS, D_INPUTS, XW, HYW, SCRATCH)

        for (nkc = 0; nkc < N*K_constL2*C_constL2; nkc++){
            // n = nkc/(K_constL2*C_constL2);
            // k = md(nkc, K_constL2*C_constL2)/C_constL2;
            // c = md(nkc, C_constL2);

            // k = nkc/(N*C_constL2);
            // n = md(nkc, N*C_constL2)/C_constL2;
            // c = md(nkc, C_constL2);

            // n = nkc/(K_constL2*C_constL2);
            // c = md(nkc, K_constL2*C_constL2)/K_constL2;
            // k = md(nkc, K_constL2);
            
            k = nkc/(N*C_constL2);
            c = md(nkc, N*C_constL2)/N;
            n = md(nkc, N);

            ncHW = (n*C_constL2 + c)*H_constL2*W_constL2;
            
            // float *d_inputs_tmp = SCRATCH + 2*omp_get_thread_num()*H_constL2*W_constL2 - 1;
            // for (h = 0; h < H_constL2; h++){
            //     for (w = 0; w < W_constL2; w++){
            //         *(++d_inputs_tmp) = 0.f;
            //     } // w
            // } // h
            // d_inputs_tmp = SCRATCH + 2*omp_get_thread_num()*H_constL2*W_constL2 - 1;


            float *restrict d_outputs_pointer = D_OUTPUTS + (n*K_constL2 + k)*output_H_constL2*output_W_constL2;

            int khwcyx_shift = (k*output_H_constL2*output_W_constL2*C_constL2 + c)*Y_constL2*X_constL2 - 1;


            for (h = 0; h < output_H_constL2; h++){
                for (w = 0; w < output_W_constL2; w++){

                    float *restrict filters_pointer = FILTERS + khwcyx_shift + (h*output_W_constL2 + w)*C_constL2*Y_constL2*X_constL2;
                    float *restrict d_filters_pointer = D_FILTERS + khwcyx_shift + (h*output_W_constL2 + w)*C_constL2*Y_constL2*X_constL2;

                    if (w - padding_constL2 >= 0 &&
                        h - padding_constL2 >= 0 &&
                        output_W_constL2 - 1 - w >= padding_constL2  &&
                        output_H_constL2 - 1 - h >= padding_constL2){
                        
                        float *restrict inputs_pointer = INPUTS + (w-padding_constL2) + (h-padding_constL2)*W_constL2 + ncHW - 1;
                        float *restrict d_inputs_pointer = D_INPUTS + (w-padding_constL2) + (h-padding_constL2)*W_constL2 + ncHW - 1;
                        // float *restrict d_inputs_pointer = d_inputs_tmp + (w-padding_constL2) + (h-padding_constL2)*W_constL2 - 1;
                        
                        for (y = 0; y < Y_constL2; y++){
                            for (x = 0; x < X_constL2; x++){
                                *(++d_filters_pointer) += (*d_outputs_pointer) * (*(++inputs_pointer)); // kcyx += nkhw * inputs_ind
                                *(++d_inputs_pointer) += (*d_outputs_pointer) * (*(++filters_pointer));
                            } // x
                            inputs_pointer += XW;
                            d_inputs_pointer += XW;

                        } // y
                    }

                    else{
                        float *restrict inputs_pointer = INPUTS + ncHW - 1 + mx(mn(h-padding_constL2, H_constL2-1), 0)*W_constL2 + mx(mn(w-padding_constL2, W_constL2-1), 0);
                        // float *restrict d_inputs_pointer = d_inputs_tmp - 1 + mx(mn(h-padding_constL2, H_constL2-1), 0)*W_constL2 + mx(mn(w-padding_constL2, W_constL2-1), 0);
                        float *restrict d_inputs_pointer = D_INPUTS + ncHW - 1 + mx(mn(h-padding_constL2, H_constL2-1), 0)*W_constL2 + mx(mn(w-padding_constL2, W_constL2-1), 0);

                        for (y = 0; y < Y_constL2; ++y){
                            float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                            float *restrict d_inputs_pointer_ncyX = d_inputs_pointer;
                            if ((y + h - padding_constL2 >= 0) && (y + h - padding_constL2 < H_constL2)){
                                for (x = 0; x < X_constL2; ++x){
                                    filters_pointer++;
                                    d_filters_pointer++;
                                    if ((x + w - padding_constL2 >= 0) && (x + w - padding_constL2 < W_constL2)){
                                        *d_filters_pointer += (*d_outputs_pointer) * (*(++inputs_pointer));
                                        *(++d_inputs_pointer) += (*d_outputs_pointer) * (*filters_pointer); 
                                    }
                                } // x
                            }

                            else { // advance pointer without going into loop
                                filters_pointer += X_constL2;
                                d_filters_pointer += X_constL2;
                            } 
                            inputs_pointer = inputs_pointer_ncyX + W_constL2; 
                            d_inputs_pointer = d_inputs_pointer_ncyX + W_constL2;
                        } // y

                        // for (y = 0; y < Y_constL2; y++){
                        //     float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
                        //     float *restrict d_inputs_pointer_ncyX = d_inputs_pointer;
                        //     for (x = 0; x < X_constL2; x++){
                        //         filters_pointer++;
                        //         d_filters_pointer++;
                        //         if ((y + h - padding_constL2 >= 0) && (y + h - padding_constL2 < H_constL2) && (x + w - padding_constL2 >= 0) && (x + w - padding_constL2 < W_constL2)){
                        //             *d_filters_pointer += (*d_outputs_pointer) * (*(++inputs_pointer)); // kcyx += nkhw * inputs_ind
                        //             *(++d_inputs_pointer) += (*d_outputs_pointer) * (*filters_pointer);
                        //         }
                        //     } // x

                        //     inputs_pointer = inputs_pointer_ncyX + W_constL2;
                        //     d_inputs_pointer = d_inputs_pointer_ncyX + W_constL2;
                        // } // y
                    }
                    // float *restrict inputs_pointer = INPUTS + w + h*W_constL2 + ncHW - 1;
                    // float *restrict d_inputs_pointer = D_INPUTS + w + h*W_constL2 + ncHW - 1;

                    // for (y = 0; y < Y_constL2; y++){
                    //     for (x = 0; x < X_constL2; x++){
                    //         *(++d_filters_pointer) += (*d_outputs_pointer) * (*(++inputs_pointer)); // kcyx += nkhw * inputs_ind
                    //         *(++d_inputs_pointer) += (*d_outputs_pointer) * (*(++filters_pointer));
                    //     } // x

                    //     inputs_pointer += XW;
                    //     d_inputs_pointer += XW;
                    // } // y

                    d_outputs_pointer++;
                } // w
            } // h

            // float *restrict d_inputs_pointer = D_INPUTS + ncHW - 1;
            // for (h = 0; h < H_constL2; h++){
            //     for (w = 0; w < W_constL2; w++){
            //         *(++d_inputs_pointer) += *(++d_inputs_tmp);
            //     } // w
            // } // h
        } // nkc
  
  }
}

void convolve(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int tight){
  // C channels per input, per filter

  #pragma offload target(mic:MIC_DEV) \ 
  in(INPUTS:length(0) REUSE) \ 
  in(FILTERS:length(0) REUSE) \ 
  in(OUTPUTS:length(0) REUSE)
  {
      float *output_scratch = (float *) _mm_malloc((H + Y - 1)*(W + X - 1)*sizeof(float), ALIGN);

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

    // free(output_scratch);    
    _mm_free(output_scratch);  
  }

}

void gram_schmidt(int N, int C, float *restrict A, float *restrict SCRATCH){

    #pragma offload target(mic:MIC_DEV) \ 
    in(A:length(0) REUSE) \ 
    in(SCRATCH:length(0) REUSE)
    {

        SCRATCH[0 : N*C] = A[0 : N*C];
        float *projection = SCRATCH + N*C;

        // orthogonalize vectors
        for (int n = 1; n < N; n++){
            float *current_u = A + n*C;
            float *current_v = SCRATCH + n*C;

            for (int nn = 0; nn < n; nn++){
                // compute projection
                float *previous_u = A + nn*C;
                float projection_coeff = __sec_reduce_add(previous_u[0 : C] * current_v[0 : C]) / 
                                         __sec_reduce_add(previous_u[0 : C] * previous_u[0 : C]);
                projection[0 : C] =  projection_coeff * previous_u[0 : C];

                // subtract projection
                current_u[0 : C] -= projection[0 : C];
            }
        }

        // normalize vectors
        for (int n = 0; n < N; n++){
            float *current_u = A + n*C;

            float norm2_u = pow(__sec_reduce_add(current_u[0 : C] * current_u[0 : C]), 0.5);
            current_u[0 : C] /= norm2_u;
        }
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
