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
#include <mkl_dfti.h>
#include <complex.h>

#include <generated_macros.h>
#include <macros.h>


#if ((N_BLOCK*K_BLOCK) != 16) && ((N_BLOCK*K_BLOCK) != 32) && ((N_BLOCK*K_BLOCK) != 64)
    #error "N_BLOCK*K_BLOCK should be 16,32 or 64"
#endif


 
#if ( (N_BLOCK * K_BLOCK) == 16) 
#define LOAD_OUTPUTS \
            __m512 res_1 = _mm512_extload_ps(convolutions, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
#define STORE_OUTPUTS \
            _mm512_extstore_ps((float *)(convolutions), res_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
#elif ( (N_BLOCK * K_BLOCK) == 32)
#define LOAD_OUTPUTS \
            __m512 res_1 = _mm512_extload_ps(convolutions, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); \
            __m512 res_2 = _mm512_extload_ps(convolutions + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
#define STORE_OUTPUTS \
            _mm512_extstore_ps((float *)(convolutions), res_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE); \
            _mm512_extstore_ps((float *)(convolutions + 16), res_2, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
#elif ( (N_BLOCK * K_BLOCK) == 64)
#define LOAD_OUTPUTS \
            __m512 res_1 = _mm512_extload_ps(convolutions, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); \
            __m512 res_2 = _mm512_extload_ps(convolutions + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); \
            __m512 res_3 = _mm512_extload_ps(convolutions + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); \
            __m512 res_4 = _mm512_extload_ps(convolutions + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
#define STORE_OUTPUTS \
            _mm512_extstore_ps((float *)(convolutions), res_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE); \
            _mm512_extstore_ps((float *)(convolutions + 16), res_2, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE); \
            _mm512_extstore_ps((float *)(convolutions + 32), res_3, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE); \
            _mm512_extstore_ps((float *)(convolutions + 48), res_4, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
#endif

#if (N_BLOCK == 1)

#if (K_BLOCK == 16) 
#define COMPUTE_OUTPUTS \
                    res_1 = _mm512_fmadd_ps(v_filters_0, v_inputs_0, res_1); 
#elif (K_BLOCK == 32)
#define COMPUTE_OUTPUTS \
                    res_1 = _mm512_fmadd_ps(v_filters_0, v_inputs_0, res_1); \
                    res_2 = _mm512_fmadd_ps(v_filters_1, v_inputs_0, res_2); 
#elif (K_BLOCK == 64)
#define COMPUTE_OUTPUTS \
                    res_1 = _mm512_fmadd_ps(v_filters_0, v_inputs_0, res_1); \
                    res_2 = _mm512_fmadd_ps(v_filters_1, v_inputs_0, res_2); \
                    res_3 = _mm512_fmadd_ps(v_filters_2, v_inputs_0, res_3); \
                    res_4 = _mm512_fmadd_ps(v_filters_3, v_inputs_0, res_4); 
#endif

#elif (N_BLOCK == 4)

#if (K_BLOCK == 16) 
#define COMPUTE_OUTPUTS \
                    res_1 = _mm512_fmadd_ps(v_filters_0, _mm512_swizzle_ps(v_inputs_0, _MM_SWIZ_REG_AAAA), res_1); \
                    res_2 = _mm512_fmadd_ps(v_filters_0, _mm512_swizzle_ps(v_inputs_0, _MM_SWIZ_REG_BBBB), res_2); \
                    res_3 = _mm512_fmadd_ps(v_filters_0, _mm512_swizzle_ps(v_inputs_0, _MM_SWIZ_REG_CCCC), res_3); \
                    res_4 = _mm512_fmadd_ps(v_filters_0, _mm512_swizzle_ps(v_inputs_0, _MM_SWIZ_REG_DDDD), res_4); 
#endif

#elif (N_BLOCK == 16)

#if (K_BLOCK == 1)
#define COMPUTE_OUTPUTS \
                    res_1 = _mm512_fmadd_ps(v_inputs_0, v_filters_0, res_1); 
#elif (K_BLOCK == 4)
#define COMPUTE_OUTPUTS \
                    res_1 = _mm512_fmadd_ps(v_inputs_0, _mm512_swizzle_ps(v_filters_0, _MM_SWIZ_REG_AAAA), res_1); \
                    res_2 = _mm512_fmadd_ps(v_inputs_0, _mm512_swizzle_ps(v_filters_0, _MM_SWIZ_REG_BBBB), res_2); \
                    res_3 = _mm512_fmadd_ps(v_inputs_0, _mm512_swizzle_ps(v_filters_0, _MM_SWIZ_REG_CCCC), res_3); \
                    res_4 = _mm512_fmadd_ps(v_inputs_0, _mm512_swizzle_ps(v_filters_0, _MM_SWIZ_REG_DDDD), res_4); 
#endif

#elif (N_BLOCK == 32)

#if (K_BLOCK == 1)
#define COMPUTE_OUTPUTS \
                    res_1 = _mm512_fmadd_ps(v_inputs_0, v_filters_0, res_1); \
                    res_2 = _mm512_fmadd_ps(v_inputs_1, v_filters_0, res_2); 
#endif

#elif (N_BLOCK == 64)

#if (K_BLOCK == 1)
#define COMPUTE_OUTPUTS \
                    res_1 = _mm512_fmadd_ps(v_inputs_0, v_filters_0, res_1); \
                    res_2 = _mm512_fmadd_ps(v_inputs_1, v_filters_0, res_2); \
                    res_3 = _mm512_fmadd_ps(v_inputs_2, v_filters_0, res_3); \
                    res_4 = _mm512_fmadd_ps(v_inputs_3, v_filters_0, res_4); 
#endif
#endif


#if 0 
                    res_1 = _mm512_fmadd_ps(v_inputs_0, v_filters_0, res_1); \
                    res_2 = _mm512_fmadd_ps(v_inputs_0, v_filters_1, res_2); \
                    res_3 = _mm512_fmadd_ps(v_inputs_0, v_filters_2, res_3); \
                    res_4 = _mm512_fmadd_ps(v_inputs_0, v_filters_3, res_4); 
#endif 


// PREFETCH_FILTERS
#if (K_BLOCK == 1)
#define PREFETCH_FILTERS ;
#elif (K_BLOCK == 4)
#define PREFETCH_FILTERS ;
#elif (K_BLOCK == 16) 
#define PREFETCH_FILTERS \
                    _mm_prefetch((char *)(filters_pointer + X_const*K_BLOCK), _MM_HINT_T0);
#elif (K_BLOCK == 32)
#define PREFETCH_FILTERS \
                    _mm_prefetch((char *)(filters_pointer + X_const*K_BLOCK), _MM_HINT_T0); \
                    _mm_prefetch((char *)(filters_pointer + X_const*K_BLOCK + 16), _MM_HINT_T0);
#elif (K_BLOCK == 64)
#define PREFETCH_FILTERS \
                    _mm_prefetch((char *)(filters_pointer + X_const*K_BLOCK), _MM_HINT_T0); \
                    _mm_prefetch((char *)(filters_pointer + X_const*K_BLOCK + 16), _MM_HINT_T0); \
                    _mm_prefetch((char *)(filters_pointer + X_const*K_BLOCK + 32), _MM_HINT_T0); \
                    _mm_prefetch((char *)(filters_pointer + X_const*K_BLOCK + 48), _MM_HINT_T0);
#endif

// PREFETCH_INPUTS
#if (N_BLOCK == 1) 
#define PREFETCH_INPUTS ;
#elif (N_BLOCK == 4) 
#define PREFETCH_INPUTS ;
#elif (N_BLOCK == 16) 
#define PREFETCH_INPUTS \
                    _mm_prefetch((char *)(inputs_pointer + W_const*N_BLOCK), _MM_HINT_T0);
#elif (N_BLOCK == 32)
#define PREFETCH_INPUTS \
                    _mm_prefetch((char *)(inputs_pointer + W_const*N_BLOCK), _MM_HINT_T0); \
                    _mm_prefetch((char *)(inputs_pointer + W_const*N_BLOCK + 16), _MM_HINT_T0);
#elif (N_BLOCK == 64)
#define PREFETCH_INPUTS \
                    _mm_prefetch((char *)(inputs_pointer + W_const*N_BLOCK), _MM_HINT_T0); \
                    _mm_prefetch((char *)(inputs_pointer + W_const*N_BLOCK + 16), _MM_HINT_T0); \
                    _mm_prefetch((char *)(inputs_pointer + W_const*N_BLOCK + 32), _MM_HINT_T0); \
                    _mm_prefetch((char *)(inputs_pointer + W_const*N_BLOCK + 48), _MM_HINT_T0);
#endif

//LOAD_INPUTS
#if N_BLOCK==1 
#define LOAD_INPUTS \
                    __m512 v_inputs_0 = _mm512_set1_ps(*inputs_pointer);
#elif N_BLOCK==4 
#define LOAD_INPUTS \
                    __m512 v_inputs_0 = _mm512_extload_ps(inputs_pointer +  0, _MM_UPCONV_PS_NONE, _MM_BROADCAST_4X16, _MM_HINT_NONE); 
#elif N_BLOCK == 16 
#define LOAD_INPUTS \
                    __m512 v_inputs_0 = _mm512_extload_ps(inputs_pointer +  0, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
#elif N_BLOCK == 32 
#define LOAD_INPUTS \
                    __m512 v_inputs_0 = _mm512_extload_ps(inputs_pointer +  0, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); \
                    __m512 v_inputs_1 = _mm512_extload_ps(inputs_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
#elif N_BLOCK == 64 
#define LOAD_INPUTS \
                    __m512 v_inputs_0 = _mm512_extload_ps(inputs_pointer +  0, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);\
                    __m512 v_inputs_1 = _mm512_extload_ps(inputs_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);\
                    __m512 v_inputs_2 = _mm512_extload_ps(inputs_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);\
                    __m512 v_inputs_3 = _mm512_extload_ps(inputs_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
#endif  


//LOAD_FILTERS
#if K_BLOCK==1 
#define LOAD_FILTERS \
                    __m512 v_filters_0 = _mm512_set1_ps(*filters_pointer);
#elif K_BLOCK==4 
#define LOAD_FILTERS \
                    __m512 v_filters_0 = _mm512_extload_ps(filters_pointer +  0, _MM_UPCONV_PS_NONE, _MM_BROADCAST_4X16, _MM_HINT_NONE); 
#elif K_BLOCK == 16 
#define LOAD_FILTERS \
                    __m512 v_filters_0 = _mm512_extload_ps(filters_pointer +  0, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
#elif K_BLOCK == 32 
#define LOAD_FILTERS \
                    __m512 v_filters_0 = _mm512_extload_ps(filters_pointer +  0, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); \
                    __m512 v_filters_1 = _mm512_extload_ps(filters_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
#elif K_BLOCK == 64 
#define LOAD_FILTERS \
                    __m512 v_filters_0 = _mm512_extload_ps(filters_pointer +  0, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);\
                    __m512 v_filters_1 = _mm512_extload_ps(filters_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);\
                    __m512 v_filters_2 = _mm512_extload_ps(filters_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);\
                    __m512 v_filters_3 = _mm512_extload_ps(filters_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
#endif  


// convolution after interleaving N and K and blocking C, after intrinsics
int *convolution_layer1(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){
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
    assert(pooled_H_const == ceil((output_H_const - pooling_radius_const + 1.f)/pooling_stride_const));
    assert(pooled_W_const == ceil((output_W_const - pooling_radius_const + 1.f)/pooling_stride_const));

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE) \
    in(ARGMAXS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {
        int n_block, n, k_block, k, i, j, h, w, c, c_block, y, x;
        int nk, hw, ij, nkhw;

        // computation of constants
        int XWN = (-X_const + W_const)*N,
            HYWN = (H_const-Y_const)*W_const*N;        

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nk, hw, ij, n_block, n, k, k_block, h, w, c, c_block, y, x, i, j) \
            shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XWN, HYWN)
        
        // ~=~=~=~=~=~=~=~= CONVOLUTION ~=~=~=~=~=~=~=~= 
        for (nk = 0; nk < N/N_BLOCK*K_const/K_BLOCK; nk++){
           
#if K_BLOCK > N_BLOCK
            k_block = nk / (N/N_BLOCK);
            k = k_block*K_BLOCK;
            n_block = md(nk, N/N_BLOCK);
            n = n_block*N_BLOCK;
#else 
            n_block = nk / (K_const/K_BLOCK);
            n = n_block*N_BLOCK;
            k_block = md(nk, K_const/K_BLOCK);
            k = k_block*K_BLOCK;
#endif

            SCRATCH[omp_get_thread_num()*output_H_const*output_W_const*N_BLOCK*K_BLOCK : output_H_const*output_W_const*N_BLOCK*K_BLOCK] = 0.f;

            for (c_block = 0; c_block < C_const/C_BLOCK; c_block++){
                c = c_block*C_BLOCK;

                for (h = 0; h < output_H_const; h++){
                    for (w = 0; w < output_W_const; w++){

#if K_BLOCK > N_BLOCK
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const, output_W_const, N_BLOCK, K_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const, output_W_const, N_BLOCK, K_BLOCK);
#else
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const, output_W_const, K_BLOCK, N_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const, output_W_const, K_BLOCK, N_BLOCK);
#endif
                __assume_aligned(convolutions, 64);

#pragma unroll
            for(int i = 0; i < (N_BLOCK*K_BLOCK); i+= 16)
            {
                _mm_prefetch((char *)(convolutions_next + i), _MM_HINT_ET0);
            }

                        // if we're not on boundary (i.e not affected by padding)
                        if (w - padding_const >= 0 &&
                            h - padding_const >= 0 &&
                            output_W_const - 1 - w >= padding_const  &&
                            output_H_const - 1 - h >= padding_const){

#if 1 && defined __MIC__
                // The following loads the a N_BLOCK*K_BLOCK region of SCRATCH space into SIMD registers
                LOAD_OUTPUTS;
#endif

#pragma unroll (C_BLOCK)
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, h - padding_const, w - padding_const, 0, C_const, H_const, W_const, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const, Y_const, X_const, K_BLOCK);
                             
                // The idea is to ensure that the working space of INPUTS = [N_BLOCK * C_BLOCK * W * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // The idea is to ensure that the working space of FILTERS = [K_BLOCK * C_BLOCK * X * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // TODO: introduce L2 prefetch for INPUTS + ti5(n_block, c, h - padding_const + Y_const, w - padding_const + X_const + 2 that may help the above problem if it exists
                            for (y = 0; y < Y_const; ++y){                            
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const*N_BLOCK), _MM_HINT_T0);
                                for (x = 0; x < X_const; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                                } // x
                                inputs_pointer += (-X_const + W_const)*N_BLOCK;
                            } // y
                } // cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        }

                        else{
#if 1 && defined __MIC__
                LOAD_OUTPUTS;
#endif
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, mx(mn(h-padding_const, H_const-1), 0), mx(mn(w-padding_const, W_const-1), 0), 0, C_const, H_const, W_const, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const, Y_const, X_const, K_BLOCK);
                            
                int min_x = mx(0, (padding_const - w)); 
                int max_x = mn(X_const, (W_const + padding_const - w)); 
                int min_y = mx(0, (padding_const - h)); 
                int max_y = mn(Y_const, (H_const + padding_const - h)); 

                filters_pointer += min_y*X_const*K_BLOCK;
                //TODO: I am fairly sure more prefetches are required for FILTERS here...
                            for (y = min_y; y < max_y; ++y){
                                float *restrict inputs_pointer_y = inputs_pointer; // start-of-line pointer
                filters_pointer += min_x*K_BLOCK;
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const*N_BLOCK), _MM_HINT_T0);
#pragma unroll (X_const-padding_const)
#pragma noprefetch
                                for (x = min_x; x < max_x; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                } // x
                filters_pointer += (X_const - max_x)*K_BLOCK;
                                inputs_pointer = inputs_pointer_y + W_const*N_BLOCK; 
                            } //y 
                filters_pointer += (Y_const - max_y)*X_const*K_BLOCK;
                } //cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        } // if-else
                    } // w
                } // h
            } // c_block


            // ~=~=~=~=~=~=~=~= POOLING ~=~=~=~=~=~=~=~= 
            for (h = 0; h < pooled_H_const; h++){
                for (w = 0; w < pooled_W_const; w++){

                    int h_output = h*pooling_stride_const;
                    int w_output = w*pooling_stride_const;

                    int window_width = pooling_radius_const - mx(w_output + pooling_radius_const - output_W_const, 0);
                    int window_height = pooling_radius_const - mx(h_output + pooling_radius_const - output_H_const, 0);
                    
                    for (int kk = 0; kk < K_BLOCK; kk++){
#if K_BLOCK > N_BLOCK 
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, 0, kk, output_H_const, output_W_const, N_BLOCK, K_BLOCK);
#else
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, kk, 0, output_H_const, output_W_const, K_BLOCK, N_BLOCK);
#endif
                        int *restrict argmaxs_pointer = ARGMAXS + ti5(n_block, k + kk, h, w, 0, K_const, pooled_H_const, pooled_W_const, N_BLOCK);
                        float *restrict pooled_outputs_pointer = OUTPUTS + ti5(n_block, k + kk, h, w, 0, K_const, pooled_H_const, pooled_W_const, N_BLOCK);
                        pooled_outputs_pointer[0 : N_BLOCK] = -1.0e6;

                        int outputs_index = h_output*output_W_const + w_output;
                        for (y = 0; y < window_height; y++){
                            for (x = 0; x < window_width; x++){
#if K_BLOCK > N_BLOCK
                                if (outputs_pointer[0 : N_BLOCK : K_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK : K_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#else
                                if (outputs_pointer[0 : N_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#endif
                                outputs_index++;
                                outputs_pointer += K_BLOCK*N_BLOCK;
                            }
                            outputs_index += output_W_const - window_width;
                            outputs_pointer += (output_W_const - window_width)*K_BLOCK*N_BLOCK;
                        }
                    }

                }
            }
        } //nk

    } // pragma_offload
}

int *convolution_layer2(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){
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
    assert(pooled_H_const2 == ceil((output_H_const2 - pooling_radius_const2 + 1.f)/pooling_stride_const2));
    assert(pooled_W_const2 == ceil((output_W_const2 - pooling_radius_const2 + 1.f)/pooling_stride_const2));

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE) \
    in(ARGMAXS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {
        int n_block, n, k_block, k, i, j, h, w, c, c_block, y, x;
        int nk, hw, ij, nkhw;

        // computation of const2ants
        int XWN = (-X_const2 + W_const2)*N,
            HYWN = (H_const2-Y_const2)*W_const2*N;        

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nk, hw, ij, n_block, n, k, k_block, h, w, c, c_block, y, x, i, j) \
            shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XWN, HYWN)
        
        // ~=~=~=~=~=~=~=~= CONVOLUTION ~=~=~=~=~=~=~=~= 
        for (nk = 0; nk < N/N_BLOCK*K_const2/K_BLOCK; nk++){
           
#if K_BLOCK > N_BLOCK
            k_block = nk / (N/N_BLOCK);
            k = k_block*K_BLOCK;
            n_block = md(nk, N/N_BLOCK);
            n = n_block*N_BLOCK;
#else 
            n_block = nk / (K_const2/K_BLOCK);
            n = n_block*N_BLOCK;
            k_block = md(nk, K_const2/K_BLOCK);
            k = k_block*K_BLOCK;
#endif

            SCRATCH[omp_get_thread_num()*output_H_const2*output_W_const2*N_BLOCK*K_BLOCK : output_H_const2*output_W_const2*N_BLOCK*K_BLOCK] = 0.f;

            for (c_block = 0; c_block < C_const2/C_BLOCK; c_block++){
                c = c_block*C_BLOCK;

                for (h = 0; h < output_H_const2; h++){
                    for (w = 0; w < output_W_const2; w++){

#if K_BLOCK > N_BLOCK
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const2, output_W_const2, N_BLOCK, K_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const2, output_W_const2, N_BLOCK, K_BLOCK);
#else
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const2, output_W_const2, K_BLOCK, N_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const2, output_W_const2, K_BLOCK, N_BLOCK);
#endif
                __assume_aligned(convolutions, 64);

#pragma unroll
            for(int i = 0; i < (N_BLOCK*K_BLOCK); i+= 16)
            {
                _mm_prefetch((char *)(convolutions_next + i), _MM_HINT_ET0);
            }

                        // if we're not on boundary (i.e not affected by padding)
                        if (w - padding_const2 >= 0 &&
                            h - padding_const2 >= 0 &&
                            output_W_const2 - 1 - w >= padding_const2  &&
                            output_H_const2 - 1 - h >= padding_const2){

#if 1 && defined __MIC__
                // The following loads the a N_BLOCK*K_BLOCK region of SCRATCH space into SIMD registers
                LOAD_OUTPUTS;
#endif

#pragma unroll (C_BLOCK)
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, h - padding_const2, w - padding_const2, 0, C_const2, H_const2, W_const2, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const2, Y_const2, X_const2, K_BLOCK);
                             
                // The idea is to ensure that the working space of INPUTS = [N_BLOCK * C_BLOCK * W * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // The idea is to ensure that the working space of FILTERS = [K_BLOCK * C_BLOCK * X * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // TODO: introduce L2 prefetch for INPUTS + ti5(n_block, c, h - padding_const2 + Y_const2, w - padding_const2 + X_const2 + 2 that may help the above problem if it exists
                            for (y = 0; y < Y_const2; ++y){                            
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const2*N_BLOCK), _MM_HINT_T0);
                                for (x = 0; x < X_const2; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                                } // x
                                inputs_pointer += (-X_const2 + W_const2)*N_BLOCK;
                            } // y
                } // cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        }

                        else{
#if 1 && defined __MIC__
                LOAD_OUTPUTS;
#endif
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, mx(mn(h-padding_const2, H_const2-1), 0), mx(mn(w-padding_const2, W_const2-1), 0), 0, C_const2, H_const2, W_const2, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const2, Y_const2, X_const2, K_BLOCK);
                            
                int min_x = mx(0, (padding_const2 - w)); 
                int max_x = mn(X_const2, (W_const2 + padding_const2 - w)); 
                int min_y = mx(0, (padding_const2 - h)); 
                int max_y = mn(Y_const2, (H_const2 + padding_const2 - h)); 

                filters_pointer += min_y*X_const2*K_BLOCK;
                //TODO: I am fairly sure more prefetches are required for FILTERS here...
                            for (y = min_y; y < max_y; ++y){
                                float *restrict inputs_pointer_y = inputs_pointer; // start-of-line pointer
                filters_pointer += min_x*K_BLOCK;
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const2*N_BLOCK), _MM_HINT_T0);
#pragma unroll (X_const2-padding_const2)
#pragma noprefetch
                                for (x = min_x; x < max_x; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                } // x
                filters_pointer += (X_const2 - max_x)*K_BLOCK;
                                inputs_pointer = inputs_pointer_y + W_const2*N_BLOCK; 
                            } //y 
                filters_pointer += (Y_const2 - max_y)*X_const2*K_BLOCK;
                } //cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        } // if-else
                    } // w
                } // h
            } // c_block


            // ~=~=~=~=~=~=~=~= POOLING ~=~=~=~=~=~=~=~= 
            for (h = 0; h < pooled_H_const2; h++){
                for (w = 0; w < pooled_W_const2; w++){

                    int h_output = h*pooling_stride_const2;
                    int w_output = w*pooling_stride_const2;

                    int window_width = pooling_radius_const2 - mx(w_output + pooling_radius_const2 - output_W_const2, 0);
                    int window_height = pooling_radius_const2 - mx(h_output + pooling_radius_const2 - output_H_const2, 0);
                    
                    for (int kk = 0; kk < K_BLOCK; kk++){
#if K_BLOCK > N_BLOCK 
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, 0, kk, output_H_const2, output_W_const2, N_BLOCK, K_BLOCK);
#else
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, kk, 0, output_H_const2, output_W_const2, K_BLOCK, N_BLOCK);
#endif
                        int *restrict argmaxs_pointer = ARGMAXS + ti5(n_block, k + kk, h, w, 0, K_const2, pooled_H_const2, pooled_W_const2, N_BLOCK);
                        float *restrict pooled_outputs_pointer = OUTPUTS + ti5(n_block, k + kk, h, w, 0, K_const2, pooled_H_const2, pooled_W_const2, N_BLOCK);
                        pooled_outputs_pointer[0 : N_BLOCK] = -1.0e6;

                        int outputs_index = h_output*output_W_const2 + w_output;
                        for (y = 0; y < window_height; y++){
                            for (x = 0; x < window_width; x++){
#if K_BLOCK > N_BLOCK
                                if (outputs_pointer[0 : N_BLOCK : K_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK : K_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#else
                                if (outputs_pointer[0 : N_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#endif
                                outputs_index++;
                                outputs_pointer += K_BLOCK*N_BLOCK;
                            }
                            outputs_index += output_W_const2 - window_width;
                            outputs_pointer += (output_W_const2 - window_width)*K_BLOCK*N_BLOCK;
                        }
                    }

                }
            }
        } //nk

    } // pragma_offload
}

int *convolution_layer3(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){
    assert(C == C_const3);
    assert(H == H_const3);
    assert(W == W_const3);
    assert(K == K_const3);
    assert(stride == stride_const3);
    assert(padding == padding_const3);
    assert(pooling_radius == pooling_radius_const3);
    assert(pooling_stride == pooling_stride_const3);
    assert(X == X_const3);
    assert(Y == Y_const3);
    assert(output_H_const3 == (H_const3 + 2*padding_const3 - Y_const3 + 1)/stride_const3);
    assert(output_W_const3 == (W_const3 + 2*padding_const3 - X_const3 + 1)/stride_const3);
    assert(pooled_H_const3 == ceil((output_H_const3 - pooling_radius_const3 + 1.f)/pooling_stride_const3));
    assert(pooled_W_const3 == ceil((output_W_const3 - pooling_radius_const3 + 1.f)/pooling_stride_const3));

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE) \
    in(ARGMAXS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {
        int n_block, n, k_block, k, i, j, h, w, c, c_block, y, x;
        int nk, hw, ij, nkhw;

        // computation of const2ants
        int XWN = (-X_const3 + W_const3)*N,
            HYWN = (H_const3-Y_const3)*W_const3*N;        

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nk, hw, ij, n_block, n, k, k_block, h, w, c, c_block, y, x, i, j) \
            shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XWN, HYWN)
        
        // ~=~=~=~=~=~=~=~= CONVOLUTION ~=~=~=~=~=~=~=~= 
        for (nk = 0; nk < N/N_BLOCK*K_const3/K_BLOCK; nk++){
           
#if K_BLOCK > N_BLOCK
            k_block = nk / (N/N_BLOCK);
            k = k_block*K_BLOCK;
            n_block = md(nk, N/N_BLOCK);
            n = n_block*N_BLOCK;
#else 
            n_block = nk / (K_const3/K_BLOCK);
            n = n_block*N_BLOCK;
            k_block = md(nk, K_const3/K_BLOCK);
            k = k_block*K_BLOCK;
#endif

            SCRATCH[omp_get_thread_num()*output_H_const3*output_W_const3*N_BLOCK*K_BLOCK : output_H_const3*output_W_const3*N_BLOCK*K_BLOCK] = 0.f;

            for (c_block = 0; c_block < C_const3/C_BLOCK; c_block++){
                c = c_block*C_BLOCK;

                for (h = 0; h < output_H_const3; h++){
                    for (w = 0; w < output_W_const3; w++){

#if K_BLOCK > N_BLOCK
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const3, output_W_const3, N_BLOCK, K_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const3, output_W_const3, N_BLOCK, K_BLOCK);
#else
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const3, output_W_const3, K_BLOCK, N_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const3, output_W_const3, K_BLOCK, N_BLOCK);
#endif
                __assume_aligned(convolutions, 64);

#pragma unroll
            for(int i = 0; i < (N_BLOCK*K_BLOCK); i+= 16)
            {
                _mm_prefetch((char *)(convolutions_next + i), _MM_HINT_ET0);
            }

                        // if we're not on boundary (i.e not affected by padding)
                        if (w - padding_const3 >= 0 &&
                            h - padding_const3 >= 0 &&
                            output_W_const3 - 1 - w >= padding_const3  &&
                            output_H_const3 - 1 - h >= padding_const3){

#if 1 && defined __MIC__
                // The following loads the a N_BLOCK*K_BLOCK region of SCRATCH space into SIMD registers
                LOAD_OUTPUTS;
#endif

#pragma unroll (C_BLOCK)
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, h - padding_const3, w - padding_const3, 0, C_const3, H_const3, W_const3, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const3, Y_const3, X_const3, K_BLOCK);
                             
                // The idea is to ensure that the working space of INPUTS = [N_BLOCK * C_BLOCK * W * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // The idea is to ensure that the working space of FILTERS = [K_BLOCK * C_BLOCK * X * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // TODO: introduce L2 prefetch for INPUTS + ti5(n_block, c, h - padding_const3 + Y_const3, w - padding_const3 + X_const3 + 2 that may help the above problem if it exists
                            for (y = 0; y < Y_const3; ++y){                            
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const3*N_BLOCK), _MM_HINT_T0);
                                for (x = 0; x < X_const3; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                                } // x
                                inputs_pointer += (-X_const3 + W_const3)*N_BLOCK;
                            } // y
                } // cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        }

                        else{
#if 1 && defined __MIC__
                LOAD_OUTPUTS;
#endif
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, mx(mn(h-padding_const3, H_const3-1), 0), mx(mn(w-padding_const3, W_const3-1), 0), 0, C_const3, H_const3, W_const3, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const3, Y_const3, X_const3, K_BLOCK);
                            
                int min_x = mx(0, (padding_const3 - w)); 
                int max_x = mn(X_const3, (W_const3 + padding_const3 - w)); 
                int min_y = mx(0, (padding_const3 - h)); 
                int max_y = mn(Y_const3, (H_const3 + padding_const3 - h)); 

                filters_pointer += min_y*X_const3*K_BLOCK;
                //TODO: I am fairly sure more prefetches are required for FILTERS here...
                            for (y = min_y; y < max_y; ++y){
                                float *restrict inputs_pointer_y = inputs_pointer; // start-of-line pointer
                filters_pointer += min_x*K_BLOCK;
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const3*N_BLOCK), _MM_HINT_T0);
#pragma unroll (X_const3-padding_const3)
#pragma noprefetch
                                for (x = min_x; x < max_x; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                } // x
                filters_pointer += (X_const3 - max_x)*K_BLOCK;
                                inputs_pointer = inputs_pointer_y + W_const3*N_BLOCK; 
                            } //y 
                filters_pointer += (Y_const3 - max_y)*X_const3*K_BLOCK;
                } //cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        } // if-else
                    } // w
                } // h
            } // c_block


            // ~=~=~=~=~=~=~=~= POOLING ~=~=~=~=~=~=~=~= 
            for (h = 0; h < pooled_H_const3; h++){
                for (w = 0; w < pooled_W_const3; w++){

                    int h_output = h*pooling_stride_const3;
                    int w_output = w*pooling_stride_const3;

                    int window_width = pooling_radius_const3 - mx(w_output + pooling_radius_const3 - output_W_const3, 0);
                    int window_height = pooling_radius_const3 - mx(h_output + pooling_radius_const3 - output_H_const3, 0);
                    
                    for (int kk = 0; kk < K_BLOCK; kk++){
#if K_BLOCK > N_BLOCK 
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, 0, kk, output_H_const3, output_W_const3, N_BLOCK, K_BLOCK);
#else
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, kk, 0, output_H_const3, output_W_const3, K_BLOCK, N_BLOCK);
#endif
                        int *restrict argmaxs_pointer = ARGMAXS + ti5(n_block, k + kk, h, w, 0, K_const3, pooled_H_const3, pooled_W_const3, N_BLOCK);
                        float *restrict pooled_outputs_pointer = OUTPUTS + ti5(n_block, k + kk, h, w, 0, K_const3, pooled_H_const3, pooled_W_const3, N_BLOCK);
                        pooled_outputs_pointer[0 : N_BLOCK] = -1.0e6;

                        int outputs_index = h_output*output_W_const3 + w_output;
                        for (y = 0; y < window_height; y++){
                            for (x = 0; x < window_width; x++){
#if K_BLOCK > N_BLOCK
                                if (outputs_pointer[0 : N_BLOCK : K_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK : K_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#else
                                if (outputs_pointer[0 : N_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#endif
                                outputs_index++;
                                outputs_pointer += K_BLOCK*N_BLOCK;
                            }
                            outputs_index += output_W_const3 - window_width;
                            outputs_pointer += (output_W_const3 - window_width)*K_BLOCK*N_BLOCK;
                        }
                    }

                }
            }
        } //nk

    } // pragma_offload
}

int *convolution_layer4(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){
    assert(C == C_const4);
    assert(H == H_const4);
    assert(W == W_const4);
    assert(K == K_const4);
    assert(stride == stride_const4);
    assert(padding == padding_const4);
    assert(pooling_radius == pooling_radius_const4);
    assert(pooling_stride == pooling_stride_const4);
    assert(X == X_const4);
    assert(Y == Y_const4);
    assert(output_H_const4 == (H_const4 + 2*padding_const4 - Y_const4 + 1)/stride_const4);
    assert(output_W_const4 == (W_const4 + 2*padding_const4 - X_const4 + 1)/stride_const4);
    assert(pooled_H_const4 == ceil((output_H_const4 - pooling_radius_const4 + 1.f)/pooling_stride_const4));
    assert(pooled_W_const4 == ceil((output_W_const4 - pooling_radius_const4 + 1.f)/pooling_stride_const4));

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE) \
    in(ARGMAXS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {
        int n_block, n, k_block, k, i, j, h, w, c, c_block, y, x;
        int nk, hw, ij, nkhw;

        // computation of const2ants
        int XWN = (-X_const4 + W_const4)*N,
            HYWN = (H_const4-Y_const4)*W_const4*N;        

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nk, hw, ij, n_block, n, k, k_block, h, w, c, c_block, y, x, i, j) \
            shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XWN, HYWN)
        
        // ~=~=~=~=~=~=~=~= CONVOLUTION ~=~=~=~=~=~=~=~= 
        for (nk = 0; nk < N/N_BLOCK*K_const4/K_BLOCK; nk++){
           
#if K_BLOCK > N_BLOCK
            k_block = nk / (N/N_BLOCK);
            k = k_block*K_BLOCK;
            n_block = md(nk, N/N_BLOCK);
            n = n_block*N_BLOCK;
#else 
            n_block = nk / (K_const4/K_BLOCK);
            n = n_block*N_BLOCK;
            k_block = md(nk, K_const4/K_BLOCK);
            k = k_block*K_BLOCK;
#endif

            SCRATCH[omp_get_thread_num()*output_H_const4*output_W_const4*N_BLOCK*K_BLOCK : output_H_const4*output_W_const4*N_BLOCK*K_BLOCK] = 0.f;

            for (c_block = 0; c_block < C_const4/C_BLOCK; c_block++){
                c = c_block*C_BLOCK;

                for (h = 0; h < output_H_const4; h++){
                    for (w = 0; w < output_W_const4; w++){

#if K_BLOCK > N_BLOCK
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const4, output_W_const4, N_BLOCK, K_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const4, output_W_const4, N_BLOCK, K_BLOCK);
#else
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const4, output_W_const4, K_BLOCK, N_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const4, output_W_const4, K_BLOCK, N_BLOCK);
#endif
                __assume_aligned(convolutions, 64);

#pragma unroll
            for(int i = 0; i < (N_BLOCK*K_BLOCK); i+= 16)
            {
                _mm_prefetch((char *)(convolutions_next + i), _MM_HINT_ET0);
            }

                        // if we're not on boundary (i.e not affected by padding)
                        if (w - padding_const4 >= 0 &&
                            h - padding_const4 >= 0 &&
                            output_W_const4 - 1 - w >= padding_const4  &&
                            output_H_const4 - 1 - h >= padding_const4){

#if 1 && defined __MIC__
                // The following loads the a N_BLOCK*K_BLOCK region of SCRATCH space into SIMD registers
                LOAD_OUTPUTS;
#endif

#pragma unroll (C_BLOCK)
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, h - padding_const4, w - padding_const4, 0, C_const4, H_const4, W_const4, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const4, Y_const4, X_const4, K_BLOCK);
                             
                // The idea is to ensure that the working space of INPUTS = [N_BLOCK * C_BLOCK * W * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // The idea is to ensure that the working space of FILTERS = [K_BLOCK * C_BLOCK * X * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // TODO: introduce L2 prefetch for INPUTS + ti5(n_block, c, h - padding_const4 + Y_const4, w - padding_const4 + X_const4 + 2 that may help the above problem if it exists
                            for (y = 0; y < Y_const4; ++y){                            
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const4*N_BLOCK), _MM_HINT_T0);
                                for (x = 0; x < X_const4; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                                } // x
                                inputs_pointer += (-X_const4 + W_const4)*N_BLOCK;
                            } // y
                } // cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        }

                        else{
#if 1 && defined __MIC__
                LOAD_OUTPUTS;
#endif
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, mx(mn(h-padding_const4, H_const4-1), 0), mx(mn(w-padding_const4, W_const4-1), 0), 0, C_const4, H_const4, W_const4, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const4, Y_const4, X_const4, K_BLOCK);
                            
                int min_x = mx(0, (padding_const4 - w)); 
                int max_x = mn(X_const4, (W_const4 + padding_const4 - w)); 
                int min_y = mx(0, (padding_const4 - h)); 
                int max_y = mn(Y_const4, (H_const4 + padding_const4 - h)); 

                filters_pointer += min_y*X_const4*K_BLOCK;
                //TODO: I am fairly sure more prefetches are required for FILTERS here...
                            for (y = min_y; y < max_y; ++y){
                                float *restrict inputs_pointer_y = inputs_pointer; // start-of-line pointer
                filters_pointer += min_x*K_BLOCK;
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const4*N_BLOCK), _MM_HINT_T0);
#pragma unroll (X_const4-padding_const4)
#pragma noprefetch
                                for (x = min_x; x < max_x; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                } // x
                filters_pointer += (X_const4 - max_x)*K_BLOCK;
                                inputs_pointer = inputs_pointer_y + W_const4*N_BLOCK; 
                            } //y 
                filters_pointer += (Y_const4 - max_y)*X_const4*K_BLOCK;
                } //cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        } // if-else
                    } // w
                } // h
            } // c_block


            // ~=~=~=~=~=~=~=~= POOLING ~=~=~=~=~=~=~=~= 
            for (h = 0; h < pooled_H_const4; h++){
                for (w = 0; w < pooled_W_const4; w++){

                    int h_output = h*pooling_stride_const4;
                    int w_output = w*pooling_stride_const4;

                    int window_width = pooling_radius_const4 - mx(w_output + pooling_radius_const4 - output_W_const4, 0);
                    int window_height = pooling_radius_const4 - mx(h_output + pooling_radius_const4 - output_H_const4, 0);
                    
                    for (int kk = 0; kk < K_BLOCK; kk++){
#if K_BLOCK > N_BLOCK 
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, 0, kk, output_H_const4, output_W_const4, N_BLOCK, K_BLOCK);
#else
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, kk, 0, output_H_const4, output_W_const4, K_BLOCK, N_BLOCK);
#endif
                        int *restrict argmaxs_pointer = ARGMAXS + ti5(n_block, k + kk, h, w, 0, K_const4, pooled_H_const4, pooled_W_const4, N_BLOCK);
                        float *restrict pooled_outputs_pointer = OUTPUTS + ti5(n_block, k + kk, h, w, 0, K_const4, pooled_H_const4, pooled_W_const4, N_BLOCK);
                        pooled_outputs_pointer[0 : N_BLOCK] = -1.0e6;

                        int outputs_index = h_output*output_W_const4 + w_output;
                        for (y = 0; y < window_height; y++){
                            for (x = 0; x < window_width; x++){
#if K_BLOCK > N_BLOCK
                                if (outputs_pointer[0 : N_BLOCK : K_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK : K_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#else
                                if (outputs_pointer[0 : N_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#endif
                                outputs_index++;
                                outputs_pointer += K_BLOCK*N_BLOCK;
                            }
                            outputs_index += output_W_const4 - window_width;
                            outputs_pointer += (output_W_const4 - window_width)*K_BLOCK*N_BLOCK;
                        }
                    }

                }
            }
        } //nk

    } // pragma_offload
}

int *convolution_layer5(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){
    assert(C == C_const5);
    assert(H == H_const5);
    assert(W == W_const5);
    assert(K == K_const5);
    assert(stride == stride_const5);
    assert(padding == padding_const5);
    assert(pooling_radius == pooling_radius_const5);
    assert(pooling_stride == pooling_stride_const5);
    assert(X == X_const5);
    assert(Y == Y_const5);
    assert(output_H_const5 == (H_const5 + 2*padding_const5 - Y_const5 + 1)/stride_const5);
    assert(output_W_const5 == (W_const5 + 2*padding_const5 - X_const5 + 1)/stride_const5);
    assert(pooled_H_const5 == ceil((output_H_const5 - pooling_radius_const5 + 1.f)/pooling_stride_const5));
    assert(pooled_W_const5 == ceil((output_W_const5 - pooling_radius_const5 + 1.f)/pooling_stride_const5));

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE) \
    in(ARGMAXS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {
        int n_block, n, k_block, k, i, j, h, w, c, c_block, y, x;
        int nk, hw, ij, nkhw;

        // computation of const2ants
        int XWN = (-X_const5 + W_const5)*N,
            HYWN = (H_const5-Y_const5)*W_const5*N;        

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nk, hw, ij, n_block, n, k, k_block, h, w, c, c_block, y, x, i, j) \
            shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XWN, HYWN)
        
        // ~=~=~=~=~=~=~=~= CONVOLUTION ~=~=~=~=~=~=~=~= 
        for (nk = 0; nk < N/N_BLOCK*K_const5/K_BLOCK; nk++){
           
#if K_BLOCK > N_BLOCK
            k_block = nk / (N/N_BLOCK);
            k = k_block*K_BLOCK;
            n_block = md(nk, N/N_BLOCK);
            n = n_block*N_BLOCK;
#else 
            n_block = nk / (K_const5/K_BLOCK);
            n = n_block*N_BLOCK;
            k_block = md(nk, K_const5/K_BLOCK);
            k = k_block*K_BLOCK;
#endif

            SCRATCH[omp_get_thread_num()*output_H_const5*output_W_const5*N_BLOCK*K_BLOCK : output_H_const5*output_W_const5*N_BLOCK*K_BLOCK] = 0.f;

            for (c_block = 0; c_block < C_const5/C_BLOCK; c_block++){
                c = c_block*C_BLOCK;

                for (h = 0; h < output_H_const5; h++){
                    for (w = 0; w < output_W_const5; w++){

#if K_BLOCK > N_BLOCK
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const5, output_W_const5, N_BLOCK, K_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const5, output_W_const5, N_BLOCK, K_BLOCK);
#else
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const5, output_W_const5, K_BLOCK, N_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const5, output_W_const5, K_BLOCK, N_BLOCK);
#endif
                __assume_aligned(convolutions, 64);

#pragma unroll
            for(int i = 0; i < (N_BLOCK*K_BLOCK); i+= 16)
            {
                _mm_prefetch((char *)(convolutions_next + i), _MM_HINT_ET0);
            }

                        // if we're not on boundary (i.e not affected by padding)
                        if (w - padding_const5 >= 0 &&
                            h - padding_const5 >= 0 &&
                            output_W_const5 - 1 - w >= padding_const5  &&
                            output_H_const5 - 1 - h >= padding_const5){

#if 1 && defined __MIC__
                // The following loads the a N_BLOCK*K_BLOCK region of SCRATCH space into SIMD registers
                LOAD_OUTPUTS;
#endif

#pragma unroll (C_BLOCK)
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, h - padding_const5, w - padding_const5, 0, C_const5, H_const5, W_const5, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const5, Y_const5, X_const5, K_BLOCK);
                             
                // The idea is to ensure that the working space of INPUTS = [N_BLOCK * C_BLOCK * W * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // The idea is to ensure that the working space of FILTERS = [K_BLOCK * C_BLOCK * X * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // TODO: introduce L2 prefetch for INPUTS + ti5(n_block, c, h - padding_const5 + Y_const5, w - padding_const5 + X_const5 + 2 that may help the above problem if it exists
                            for (y = 0; y < Y_const5; ++y){                            
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const5*N_BLOCK), _MM_HINT_T0);
                                for (x = 0; x < X_const5; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                                } // x
                                inputs_pointer += (-X_const5 + W_const5)*N_BLOCK;
                            } // y
                } // cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        }

                        else{
#if 1 && defined __MIC__
                LOAD_OUTPUTS;
#endif
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, mx(mn(h-padding_const5, H_const5-1), 0), mx(mn(w-padding_const5, W_const5-1), 0), 0, C_const5, H_const5, W_const5, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const5, Y_const5, X_const5, K_BLOCK);
                            
                int min_x = mx(0, (padding_const5 - w)); 
                int max_x = mn(X_const5, (W_const5 + padding_const5 - w)); 
                int min_y = mx(0, (padding_const5 - h)); 
                int max_y = mn(Y_const5, (H_const5 + padding_const5 - h)); 

                filters_pointer += min_y*X_const5*K_BLOCK;
                //TODO: I am fairly sure more prefetches are required for FILTERS here...
                            for (y = min_y; y < max_y; ++y){
                                float *restrict inputs_pointer_y = inputs_pointer; // start-of-line pointer
                filters_pointer += min_x*K_BLOCK;
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const5*N_BLOCK), _MM_HINT_T0);
#pragma unroll (X_const5-padding_const5)
#pragma noprefetch
                                for (x = min_x; x < max_x; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                } // x
                filters_pointer += (X_const5 - max_x)*K_BLOCK;
                                inputs_pointer = inputs_pointer_y + W_const5*N_BLOCK; 
                            } //y 
                filters_pointer += (Y_const5 - max_y)*X_const5*K_BLOCK;
                } //cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        } // if-else
                    } // w
                } // h
            } // c_block


            // ~=~=~=~=~=~=~=~= POOLING ~=~=~=~=~=~=~=~= 
            for (h = 0; h < pooled_H_const5; h++){
                for (w = 0; w < pooled_W_const5; w++){

                    int h_output = h*pooling_stride_const5;
                    int w_output = w*pooling_stride_const5;

                    int window_width = pooling_radius_const5 - mx(w_output + pooling_radius_const5 - output_W_const5, 0);
                    int window_height = pooling_radius_const5 - mx(h_output + pooling_radius_const5 - output_H_const5, 0);
                    
                    for (int kk = 0; kk < K_BLOCK; kk++){
#if K_BLOCK > N_BLOCK 
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, 0, kk, output_H_const5, output_W_const5, N_BLOCK, K_BLOCK);
#else
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, kk, 0, output_H_const5, output_W_const5, K_BLOCK, N_BLOCK);
#endif
                        int *restrict argmaxs_pointer = ARGMAXS + ti5(n_block, k + kk, h, w, 0, K_const5, pooled_H_const5, pooled_W_const5, N_BLOCK);
                        float *restrict pooled_outputs_pointer = OUTPUTS + ti5(n_block, k + kk, h, w, 0, K_const5, pooled_H_const5, pooled_W_const5, N_BLOCK);
                        pooled_outputs_pointer[0 : N_BLOCK] = -1.0e6;

                        int outputs_index = h_output*output_W_const5 + w_output;
                        for (y = 0; y < window_height; y++){
                            for (x = 0; x < window_width; x++){
#if K_BLOCK > N_BLOCK
                                if (outputs_pointer[0 : N_BLOCK : K_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK : K_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#else
                                if (outputs_pointer[0 : N_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#endif
                                outputs_index++;
                                outputs_pointer += K_BLOCK*N_BLOCK;
                            }
                            outputs_index += output_W_const5 - window_width;
                            outputs_pointer += (output_W_const5 - window_width)*K_BLOCK*N_BLOCK;
                        }
                    }

                }
            }
        } //nk

    } // pragma_offload
}

int *convolution_layer6(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){
    assert(C == C_const6);
    assert(H == H_const6);
    assert(W == W_const6);
    assert(K == K_const6);
    assert(stride == stride_const6);
    assert(padding == padding_const6);
    assert(pooling_radius == pooling_radius_const6);
    assert(pooling_stride == pooling_stride_const6);
    assert(X == X_const6);
    assert(Y == Y_const6);
    assert(output_H_const6 == (H_const6 + 2*padding_const6 - Y_const6 + 1)/stride_const6);
    assert(output_W_const6 == (W_const6 + 2*padding_const6 - X_const6 + 1)/stride_const6);
    assert(pooled_H_const6 == ceil((output_H_const6 - pooling_radius_const6 + 1.f)/pooling_stride_const6));
    assert(pooled_W_const6 == ceil((output_W_const6 - pooling_radius_const6 + 1.f)/pooling_stride_const6));

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE) \
    in(ARGMAXS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {
        int n_block, n, k_block, k, i, j, h, w, c, c_block, y, x;
        int nk, hw, ij, nkhw;

        // computation of const2ants
        int XWN = (-X_const6 + W_const6)*N,
            HYWN = (H_const6-Y_const6)*W_const6*N;        

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nk, hw, ij, n_block, n, k, k_block, h, w, c, c_block, y, x, i, j) \
            shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XWN, HYWN)
        
        // ~=~=~=~=~=~=~=~= CONVOLUTION ~=~=~=~=~=~=~=~= 
        for (nk = 0; nk < N/N_BLOCK*K_const6/K_BLOCK; nk++){
           
#if K_BLOCK > N_BLOCK
            k_block = nk / (N/N_BLOCK);
            k = k_block*K_BLOCK;
            n_block = md(nk, N/N_BLOCK);
            n = n_block*N_BLOCK;
#else 
            n_block = nk / (K_const6/K_BLOCK);
            n = n_block*N_BLOCK;
            k_block = md(nk, K_const6/K_BLOCK);
            k = k_block*K_BLOCK;
#endif

            SCRATCH[omp_get_thread_num()*output_H_const6*output_W_const6*N_BLOCK*K_BLOCK : output_H_const6*output_W_const6*N_BLOCK*K_BLOCK] = 0.f;

            for (c_block = 0; c_block < C_const6/C_BLOCK; c_block++){
                c = c_block*C_BLOCK;

                for (h = 0; h < output_H_const6; h++){
                    for (w = 0; w < output_W_const6; w++){

#if K_BLOCK > N_BLOCK
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const6, output_W_const6, N_BLOCK, K_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const6, output_W_const6, N_BLOCK, K_BLOCK);
#else
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const6, output_W_const6, K_BLOCK, N_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const6, output_W_const6, K_BLOCK, N_BLOCK);
#endif
                __assume_aligned(convolutions, 64);

#pragma unroll
            for(int i = 0; i < (N_BLOCK*K_BLOCK); i+= 16)
            {
                _mm_prefetch((char *)(convolutions_next + i), _MM_HINT_ET0);
            }

                        // if we're not on boundary (i.e not affected by padding)
                        if (w - padding_const6 >= 0 &&
                            h - padding_const6 >= 0 &&
                            output_W_const6 - 1 - w >= padding_const6  &&
                            output_H_const6 - 1 - h >= padding_const6){

#if 1 && defined __MIC__
                // The following loads the a N_BLOCK*K_BLOCK region of SCRATCH space into SIMD registers
                LOAD_OUTPUTS;
#endif

#pragma unroll (C_BLOCK)
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, h - padding_const6, w - padding_const6, 0, C_const6, H_const6, W_const6, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const6, Y_const6, X_const6, K_BLOCK);
                             
                // The idea is to ensure that the working space of INPUTS = [N_BLOCK * C_BLOCK * W * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // The idea is to ensure that the working space of FILTERS = [K_BLOCK * C_BLOCK * X * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // TODO: introduce L2 prefetch for INPUTS + ti5(n_block, c, h - padding_const6 + Y_const6, w - padding_const6 + X_const6 + 2 that may help the above problem if it exists
                            for (y = 0; y < Y_const6; ++y){                            
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const6*N_BLOCK), _MM_HINT_T0);
                                for (x = 0; x < X_const6; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                                } // x
                                inputs_pointer += (-X_const6 + W_const6)*N_BLOCK;
                            } // y
                } // cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        }

                        else{
#if 1 && defined __MIC__
                LOAD_OUTPUTS;
#endif
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, mx(mn(h-padding_const6, H_const6-1), 0), mx(mn(w-padding_const6, W_const6-1), 0), 0, C_const6, H_const6, W_const6, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const6, Y_const6, X_const6, K_BLOCK);
                            
                int min_x = mx(0, (padding_const6 - w)); 
                int max_x = mn(X_const6, (W_const6 + padding_const6 - w)); 
                int min_y = mx(0, (padding_const6 - h)); 
                int max_y = mn(Y_const6, (H_const6 + padding_const6 - h)); 

                filters_pointer += min_y*X_const6*K_BLOCK;
                //TODO: I am fairly sure more prefetches are required for FILTERS here...
                            for (y = min_y; y < max_y; ++y){
                                float *restrict inputs_pointer_y = inputs_pointer; // start-of-line pointer
                filters_pointer += min_x*K_BLOCK;
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const6*N_BLOCK), _MM_HINT_T0);
#pragma unroll (X_const6-padding_const6)
#pragma noprefetch
                                for (x = min_x; x < max_x; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                } // x
                filters_pointer += (X_const6 - max_x)*K_BLOCK;
                                inputs_pointer = inputs_pointer_y + W_const6*N_BLOCK; 
                            } //y 
                filters_pointer += (Y_const6 - max_y)*X_const6*K_BLOCK;
                } //cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        } // if-else
                    } // w
                } // h
            } // c_block


            // ~=~=~=~=~=~=~=~= POOLING ~=~=~=~=~=~=~=~= 
            for (h = 0; h < pooled_H_const6; h++){
                for (w = 0; w < pooled_W_const6; w++){

                    int h_output = h*pooling_stride_const6;
                    int w_output = w*pooling_stride_const6;

                    int window_width = pooling_radius_const6 - mx(w_output + pooling_radius_const6 - output_W_const6, 0);
                    int window_height = pooling_radius_const6 - mx(h_output + pooling_radius_const6 - output_H_const6, 0);
                    
                    for (int kk = 0; kk < K_BLOCK; kk++){
#if K_BLOCK > N_BLOCK 
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, 0, kk, output_H_const6, output_W_const6, N_BLOCK, K_BLOCK);
#else
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, kk, 0, output_H_const6, output_W_const6, K_BLOCK, N_BLOCK);
#endif
                        int *restrict argmaxs_pointer = ARGMAXS + ti5(n_block, k + kk, h, w, 0, K_const6, pooled_H_const6, pooled_W_const6, N_BLOCK);
                        float *restrict pooled_outputs_pointer = OUTPUTS + ti5(n_block, k + kk, h, w, 0, K_const6, pooled_H_const6, pooled_W_const6, N_BLOCK);
                        pooled_outputs_pointer[0 : N_BLOCK] = -1.0e6;

                        int outputs_index = h_output*output_W_const6 + w_output;
                        for (y = 0; y < window_height; y++){
                            for (x = 0; x < window_width; x++){
#if K_BLOCK > N_BLOCK
                                if (outputs_pointer[0 : N_BLOCK : K_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK : K_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#else
                                if (outputs_pointer[0 : N_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#endif
                                outputs_index++;
                                outputs_pointer += K_BLOCK*N_BLOCK;
                            }
                            outputs_index += output_W_const6 - window_width;
                            outputs_pointer += (output_W_const6 - window_width)*K_BLOCK*N_BLOCK;
                        }
                    }

                }
            }
        } //nk

    } // pragma_offload
}

int *convolution_layer7(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){
    assert(C == C_const7);
    assert(H == H_const7);
    assert(W == W_const7);
    assert(K == K_const7);
    assert(stride == stride_const7);
    assert(padding == padding_const7);
    assert(pooling_radius == pooling_radius_const7);
    assert(pooling_stride == pooling_stride_const7);
    assert(X == X_const7);
    assert(Y == Y_const7);
    assert(output_H_const7 == (H_const7 + 2*padding_const7 - Y_const7 + 1)/stride_const7);
    assert(output_W_const7 == (W_const7 + 2*padding_const7 - X_const7 + 1)/stride_const7);
    assert(pooled_H_const7 == ceil((output_H_const7 - pooling_radius_const7 + 1.f)/pooling_stride_const7));
    assert(pooled_W_const7 == ceil((output_W_const7 - pooling_radius_const7 + 1.f)/pooling_stride_const7));

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE) \
    in(ARGMAXS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {
        int n_block, n, k_block, k, i, j, h, w, c, c_block, y, x;
        int nk, hw, ij, nkhw;

        // computation of const2ants
        int XWN = (-X_const7 + W_const7)*N,
            HYWN = (H_const7-Y_const7)*W_const7*N;        

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nk, hw, ij, n_block, n, k, k_block, h, w, c, c_block, y, x, i, j) \
            shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XWN, HYWN)
        
        // ~=~=~=~=~=~=~=~= CONVOLUTION ~=~=~=~=~=~=~=~= 
        for (nk = 0; nk < N/N_BLOCK*K_const7/K_BLOCK; nk++){
           
#if K_BLOCK > N_BLOCK
            k_block = nk / (N/N_BLOCK);
            k = k_block*K_BLOCK;
            n_block = md(nk, N/N_BLOCK);
            n = n_block*N_BLOCK;
#else 
            n_block = nk / (K_const7/K_BLOCK);
            n = n_block*N_BLOCK;
            k_block = md(nk, K_const7/K_BLOCK);
            k = k_block*K_BLOCK;
#endif

            SCRATCH[omp_get_thread_num()*output_H_const7*output_W_const7*N_BLOCK*K_BLOCK : output_H_const7*output_W_const7*N_BLOCK*K_BLOCK] = 0.f;

            for (c_block = 0; c_block < C_const7/C_BLOCK; c_block++){
                c = c_block*C_BLOCK;

                for (h = 0; h < output_H_const7; h++){
                    for (w = 0; w < output_W_const7; w++){

#if K_BLOCK > N_BLOCK
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const7, output_W_const7, N_BLOCK, K_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const7, output_W_const7, N_BLOCK, K_BLOCK);
#else
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const7, output_W_const7, K_BLOCK, N_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const7, output_W_const7, K_BLOCK, N_BLOCK);
#endif
                __assume_aligned(convolutions, 64);

#pragma unroll
            for(int i = 0; i < (N_BLOCK*K_BLOCK); i+= 16)
            {
                _mm_prefetch((char *)(convolutions_next + i), _MM_HINT_ET0);
            }

                        // if we're not on boundary (i.e not affected by padding)
                        if (w - padding_const7 >= 0 &&
                            h - padding_const7 >= 0 &&
                            output_W_const7 - 1 - w >= padding_const7  &&
                            output_H_const7 - 1 - h >= padding_const7){

#if 1 && defined __MIC__
                // The following loads the a N_BLOCK*K_BLOCK region of SCRATCH space into SIMD registers
                LOAD_OUTPUTS;
#endif

#pragma unroll (C_BLOCK)
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, h - padding_const7, w - padding_const7, 0, C_const7, H_const7, W_const7, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const7, Y_const7, X_const7, K_BLOCK);
                             
                // The idea is to ensure that the working space of INPUTS = [N_BLOCK * C_BLOCK * W * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // The idea is to ensure that the working space of FILTERS = [K_BLOCK * C_BLOCK * X * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // TODO: introduce L2 prefetch for INPUTS + ti5(n_block, c, h - padding_const7 + Y_const7, w - padding_const7 + X_const7 + 2 that may help the above problem if it exists
                            for (y = 0; y < Y_const7; ++y){                            
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const7*N_BLOCK), _MM_HINT_T0);
                                for (x = 0; x < X_const7; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                                } // x
                                inputs_pointer += (-X_const7 + W_const7)*N_BLOCK;
                            } // y
                } // cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        }

                        else{
#if 1 && defined __MIC__
                LOAD_OUTPUTS;
#endif
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, mx(mn(h-padding_const7, H_const7-1), 0), mx(mn(w-padding_const7, W_const7-1), 0), 0, C_const7, H_const7, W_const7, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const7, Y_const7, X_const7, K_BLOCK);
                            
                int min_x = mx(0, (padding_const7 - w)); 
                int max_x = mn(X_const7, (W_const7 + padding_const7 - w)); 
                int min_y = mx(0, (padding_const7 - h)); 
                int max_y = mn(Y_const7, (H_const7 + padding_const7 - h)); 

                filters_pointer += min_y*X_const7*K_BLOCK;
                //TODO: I am fairly sure more prefetches are required for FILTERS here...
                            for (y = min_y; y < max_y; ++y){
                                float *restrict inputs_pointer_y = inputs_pointer; // start-of-line pointer
                filters_pointer += min_x*K_BLOCK;
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const7*N_BLOCK), _MM_HINT_T0);
#pragma unroll (X_const7-padding_const7)
#pragma noprefetch
                                for (x = min_x; x < max_x; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                } // x
                filters_pointer += (X_const7 - max_x)*K_BLOCK;
                                inputs_pointer = inputs_pointer_y + W_const7*N_BLOCK; 
                            } //y 
                filters_pointer += (Y_const7 - max_y)*X_const7*K_BLOCK;
                } //cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        } // if-else
                    } // w
                } // h
            } // c_block


            // ~=~=~=~=~=~=~=~= POOLING ~=~=~=~=~=~=~=~= 
            for (h = 0; h < pooled_H_const7; h++){
                for (w = 0; w < pooled_W_const7; w++){

                    int h_output = h*pooling_stride_const7;
                    int w_output = w*pooling_stride_const7;

                    int window_width = pooling_radius_const7 - mx(w_output + pooling_radius_const7 - output_W_const7, 0);
                    int window_height = pooling_radius_const7 - mx(h_output + pooling_radius_const7 - output_H_const7, 0);
                    
                    for (int kk = 0; kk < K_BLOCK; kk++){
#if K_BLOCK > N_BLOCK 
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, 0, kk, output_H_const7, output_W_const7, N_BLOCK, K_BLOCK);
#else
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, kk, 0, output_H_const7, output_W_const7, K_BLOCK, N_BLOCK);
#endif
                        int *restrict argmaxs_pointer = ARGMAXS + ti5(n_block, k + kk, h, w, 0, K_const7, pooled_H_const7, pooled_W_const7, N_BLOCK);
                        float *restrict pooled_outputs_pointer = OUTPUTS + ti5(n_block, k + kk, h, w, 0, K_const7, pooled_H_const7, pooled_W_const7, N_BLOCK);
                        pooled_outputs_pointer[0 : N_BLOCK] = -1.0e6;

                        int outputs_index = h_output*output_W_const7 + w_output;
                        for (y = 0; y < window_height; y++){
                            for (x = 0; x < window_width; x++){
#if K_BLOCK > N_BLOCK
                                if (outputs_pointer[0 : N_BLOCK : K_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK : K_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#else
                                if (outputs_pointer[0 : N_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#endif
                                outputs_index++;
                                outputs_pointer += K_BLOCK*N_BLOCK;
                            }
                            outputs_index += output_W_const7 - window_width;
                            outputs_pointer += (output_W_const7 - window_width)*K_BLOCK*N_BLOCK;
                        }
                    }

                }
            }
        } //nk

    } // pragma_offload
}

int *convolution_layer8(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){
    assert(C == C_const8);
    assert(H == H_const8);
    assert(W == W_const8);
    assert(K == K_const8);
    assert(stride == stride_const8);
    assert(padding == padding_const8);
    assert(pooling_radius == pooling_radius_const8);
    assert(pooling_stride == pooling_stride_const8);
    assert(X == X_const8);
    assert(Y == Y_const8);
    assert(output_H_const8 == (H_const8 + 2*padding_const8 - Y_const8 + 1)/stride_const8);
    assert(output_W_const8 == (W_const8 + 2*padding_const8 - X_const8 + 1)/stride_const8);
    assert(pooled_H_const8 == ceil((output_H_const8 - pooling_radius_const8 + 1.f)/pooling_stride_const8));
    assert(pooled_W_const8 == ceil((output_W_const8 - pooling_radius_const8 + 1.f)/pooling_stride_const8));

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE) \
    in(ARGMAXS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {
        int n_block, n, k_block, k, i, j, h, w, c, c_block, y, x;
        int nk, hw, ij, nkhw;

        // computation of const2ants
        int XWN = (-X_const8 + W_const8)*N,
            HYWN = (H_const8-Y_const8)*W_const8*N;        

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nk, hw, ij, n_block, n, k, k_block, h, w, c, c_block, y, x, i, j) \
            shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XWN, HYWN)
        
        // ~=~=~=~=~=~=~=~= CONVOLUTION ~=~=~=~=~=~=~=~= 
        for (nk = 0; nk < N/N_BLOCK*K_const8/K_BLOCK; nk++){
           
#if K_BLOCK > N_BLOCK
            k_block = nk / (N/N_BLOCK);
            k = k_block*K_BLOCK;
            n_block = md(nk, N/N_BLOCK);
            n = n_block*N_BLOCK;
#else 
            n_block = nk / (K_const8/K_BLOCK);
            n = n_block*N_BLOCK;
            k_block = md(nk, K_const8/K_BLOCK);
            k = k_block*K_BLOCK;
#endif

            SCRATCH[omp_get_thread_num()*output_H_const8*output_W_const8*N_BLOCK*K_BLOCK : output_H_const8*output_W_const8*N_BLOCK*K_BLOCK] = 0.f;

            for (c_block = 0; c_block < C_const8/C_BLOCK; c_block++){
                c = c_block*C_BLOCK;

                for (h = 0; h < output_H_const8; h++){
                    for (w = 0; w < output_W_const8; w++){

#if K_BLOCK > N_BLOCK
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const8, output_W_const8, N_BLOCK, K_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const8, output_W_const8, N_BLOCK, K_BLOCK);
#else
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const8, output_W_const8, K_BLOCK, N_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const8, output_W_const8, K_BLOCK, N_BLOCK);
#endif
                __assume_aligned(convolutions, 64);

#pragma unroll
            for(int i = 0; i < (N_BLOCK*K_BLOCK); i+= 16)
            {
                _mm_prefetch((char *)(convolutions_next + i), _MM_HINT_ET0);
            }

                        // if we're not on boundary (i.e not affected by padding)
                        if (w - padding_const8 >= 0 &&
                            h - padding_const8 >= 0 &&
                            output_W_const8 - 1 - w >= padding_const8  &&
                            output_H_const8 - 1 - h >= padding_const8){

#if 1 && defined __MIC__
                // The following loads the a N_BLOCK*K_BLOCK region of SCRATCH space into SIMD registers
                LOAD_OUTPUTS;
#endif

#pragma unroll (C_BLOCK)
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, h - padding_const8, w - padding_const8, 0, C_const8, H_const8, W_const8, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const8, Y_const8, X_const8, K_BLOCK);
                             
                // The idea is to ensure that the working space of INPUTS = [N_BLOCK * C_BLOCK * W * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // The idea is to ensure that the working space of FILTERS = [K_BLOCK * C_BLOCK * X * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // TODO: introduce L2 prefetch for INPUTS + ti5(n_block, c, h - padding_const8 + Y_const8, w - padding_const8 + X_const8 + 2 that may help the above problem if it exists
                            for (y = 0; y < Y_const8; ++y){                            
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const8*N_BLOCK), _MM_HINT_T0);
                                for (x = 0; x < X_const8; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                                } // x
                                inputs_pointer += (-X_const8 + W_const8)*N_BLOCK;
                            } // y
                } // cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        }

                        else{
#if 1 && defined __MIC__
                LOAD_OUTPUTS;
#endif
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, mx(mn(h-padding_const8, H_const8-1), 0), mx(mn(w-padding_const8, W_const8-1), 0), 0, C_const8, H_const8, W_const8, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const8, Y_const8, X_const8, K_BLOCK);
                            
                int min_x = mx(0, (padding_const8 - w)); 
                int max_x = mn(X_const8, (W_const8 + padding_const8 - w)); 
                int min_y = mx(0, (padding_const8 - h)); 
                int max_y = mn(Y_const8, (H_const8 + padding_const8 - h)); 

                filters_pointer += min_y*X_const8*K_BLOCK;
                //TODO: I am fairly sure more prefetches are required for FILTERS here...
                            for (y = min_y; y < max_y; ++y){
                                float *restrict inputs_pointer_y = inputs_pointer; // start-of-line pointer
                filters_pointer += min_x*K_BLOCK;
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const8*N_BLOCK), _MM_HINT_T0);
#pragma unroll (X_const8-padding_const8)
#pragma noprefetch
                                for (x = min_x; x < max_x; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                } // x
                filters_pointer += (X_const8 - max_x)*K_BLOCK;
                                inputs_pointer = inputs_pointer_y + W_const8*N_BLOCK; 
                            } //y 
                filters_pointer += (Y_const8 - max_y)*X_const8*K_BLOCK;
                } //cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        } // if-else
                    } // w
                } // h
            } // c_block


            // ~=~=~=~=~=~=~=~= POOLING ~=~=~=~=~=~=~=~= 
            for (h = 0; h < pooled_H_const8; h++){
                for (w = 0; w < pooled_W_const8; w++){

                    int h_output = h*pooling_stride_const8;
                    int w_output = w*pooling_stride_const8;

                    int window_width = pooling_radius_const8 - mx(w_output + pooling_radius_const8 - output_W_const8, 0);
                    int window_height = pooling_radius_const8 - mx(h_output + pooling_radius_const8 - output_H_const8, 0);
                    
                    for (int kk = 0; kk < K_BLOCK; kk++){
#if K_BLOCK > N_BLOCK 
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, 0, kk, output_H_const8, output_W_const8, N_BLOCK, K_BLOCK);
#else
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, kk, 0, output_H_const8, output_W_const8, K_BLOCK, N_BLOCK);
#endif
                        int *restrict argmaxs_pointer = ARGMAXS + ti5(n_block, k + kk, h, w, 0, K_const8, pooled_H_const8, pooled_W_const8, N_BLOCK);
                        float *restrict pooled_outputs_pointer = OUTPUTS + ti5(n_block, k + kk, h, w, 0, K_const8, pooled_H_const8, pooled_W_const8, N_BLOCK);
                        pooled_outputs_pointer[0 : N_BLOCK] = -1.0e6;

                        int outputs_index = h_output*output_W_const8 + w_output;
                        for (y = 0; y < window_height; y++){
                            for (x = 0; x < window_width; x++){
#if K_BLOCK > N_BLOCK
                                if (outputs_pointer[0 : N_BLOCK : K_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK : K_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#else
                                if (outputs_pointer[0 : N_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#endif
                                outputs_index++;
                                outputs_pointer += K_BLOCK*N_BLOCK;
                            }
                            outputs_index += output_W_const8 - window_width;
                            outputs_pointer += (output_W_const8 - window_width)*K_BLOCK*N_BLOCK;
                        }
                    }

                }
            }
        } //nk

    } // pragma_offload
}

int *convolution_layer9(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){
    assert(C == C_const9);
    assert(H == H_const9);
    assert(W == W_const9);
    assert(K == K_const9);
    assert(stride == stride_const9);
    assert(padding == padding_const9);
    assert(pooling_radius == pooling_radius_const9);
    assert(pooling_stride == pooling_stride_const9);
    assert(X == X_const9);
    assert(Y == Y_const9);
    assert(output_H_const9 == (H_const9 + 2*padding_const9 - Y_const9 + 1)/stride_const9);
    assert(output_W_const9 == (W_const9 + 2*padding_const9 - X_const9 + 1)/stride_const9);
    assert(pooled_H_const9 == ceil((output_H_const9 - pooling_radius_const9 + 1.f)/pooling_stride_const9));
    assert(pooled_W_const9 == ceil((output_W_const9 - pooling_radius_const9 + 1.f)/pooling_stride_const9));

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE) \
    in(ARGMAXS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {
        int n_block, n, k_block, k, i, j, h, w, c, c_block, y, x;
        int nk, hw, ij, nkhw;

        // computation of const2ants
        int XWN = (-X_const9 + W_const9)*N,
            HYWN = (H_const9-Y_const9)*W_const9*N;        

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nk, hw, ij, n_block, n, k, k_block, h, w, c, c_block, y, x, i, j) \
            shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XWN, HYWN)
        
        // ~=~=~=~=~=~=~=~= CONVOLUTION ~=~=~=~=~=~=~=~= 
        for (nk = 0; nk < N/N_BLOCK*K_const9/K_BLOCK; nk++){
           
#if K_BLOCK > N_BLOCK
            k_block = nk / (N/N_BLOCK);
            k = k_block*K_BLOCK;
            n_block = md(nk, N/N_BLOCK);
            n = n_block*N_BLOCK;
#else 
            n_block = nk / (K_const9/K_BLOCK);
            n = n_block*N_BLOCK;
            k_block = md(nk, K_const9/K_BLOCK);
            k = k_block*K_BLOCK;
#endif

            SCRATCH[omp_get_thread_num()*output_H_const9*output_W_const9*N_BLOCK*K_BLOCK : output_H_const9*output_W_const9*N_BLOCK*K_BLOCK] = 0.f;

            for (c_block = 0; c_block < C_const9/C_BLOCK; c_block++){
                c = c_block*C_BLOCK;

                for (h = 0; h < output_H_const9; h++){
                    for (w = 0; w < output_W_const9; w++){

#if K_BLOCK > N_BLOCK
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const9, output_W_const9, N_BLOCK, K_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const9, output_W_const9, N_BLOCK, K_BLOCK);
#else
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const9, output_W_const9, K_BLOCK, N_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const9, output_W_const9, K_BLOCK, N_BLOCK);
#endif
                __assume_aligned(convolutions, 64);

#pragma unroll
            for(int i = 0; i < (N_BLOCK*K_BLOCK); i+= 16)
            {
                _mm_prefetch((char *)(convolutions_next + i), _MM_HINT_ET0);
            }

                        // if we're not on boundary (i.e not affected by padding)
                        if (w - padding_const9 >= 0 &&
                            h - padding_const9 >= 0 &&
                            output_W_const9 - 1 - w >= padding_const9  &&
                            output_H_const9 - 1 - h >= padding_const9){

#if 1 && defined __MIC__
                // The following loads the a N_BLOCK*K_BLOCK region of SCRATCH space into SIMD registers
                LOAD_OUTPUTS;
#endif

#pragma unroll (C_BLOCK)
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, h - padding_const9, w - padding_const9, 0, C_const9, H_const9, W_const9, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const9, Y_const9, X_const9, K_BLOCK);
                             
                // The idea is to ensure that the working space of INPUTS = [N_BLOCK * C_BLOCK * W * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // The idea is to ensure that the working space of FILTERS = [K_BLOCK * C_BLOCK * X * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // TODO: introduce L2 prefetch for INPUTS + ti5(n_block, c, h - padding_const9 + Y_const9, w - padding_const9 + X_const9 + 2 that may help the above problem if it exists
                            for (y = 0; y < Y_const9; ++y){                            
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const9*N_BLOCK), _MM_HINT_T0);
                                for (x = 0; x < X_const9; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                                } // x
                                inputs_pointer += (-X_const9 + W_const9)*N_BLOCK;
                            } // y
                } // cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        }

                        else{
#if 1 && defined __MIC__
                LOAD_OUTPUTS;
#endif
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, mx(mn(h-padding_const9, H_const9-1), 0), mx(mn(w-padding_const9, W_const9-1), 0), 0, C_const9, H_const9, W_const9, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const9, Y_const9, X_const9, K_BLOCK);
                            
                int min_x = mx(0, (padding_const9 - w)); 
                int max_x = mn(X_const9, (W_const9 + padding_const9 - w)); 
                int min_y = mx(0, (padding_const9 - h)); 
                int max_y = mn(Y_const9, (H_const9 + padding_const9 - h)); 

                filters_pointer += min_y*X_const9*K_BLOCK;
                //TODO: I am fairly sure more prefetches are required for FILTERS here...
                            for (y = min_y; y < max_y; ++y){
                                float *restrict inputs_pointer_y = inputs_pointer; // start-of-line pointer
                filters_pointer += min_x*K_BLOCK;
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const9*N_BLOCK), _MM_HINT_T0);
#pragma unroll (X_const9-padding_const9)
#pragma noprefetch
                                for (x = min_x; x < max_x; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                } // x
                filters_pointer += (X_const9 - max_x)*K_BLOCK;
                                inputs_pointer = inputs_pointer_y + W_const9*N_BLOCK; 
                            } //y 
                filters_pointer += (Y_const9 - max_y)*X_const9*K_BLOCK;
                } //cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        } // if-else
                    } // w
                } // h
            } // c_block


            // ~=~=~=~=~=~=~=~= POOLING ~=~=~=~=~=~=~=~= 
            for (h = 0; h < pooled_H_const9; h++){
                for (w = 0; w < pooled_W_const9; w++){

                    int h_output = h*pooling_stride_const9;
                    int w_output = w*pooling_stride_const9;

                    int window_width = pooling_radius_const9 - mx(w_output + pooling_radius_const9 - output_W_const9, 0);
                    int window_height = pooling_radius_const9 - mx(h_output + pooling_radius_const9 - output_H_const9, 0);
                    
                    for (int kk = 0; kk < K_BLOCK; kk++){
#if K_BLOCK > N_BLOCK 
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, 0, kk, output_H_const9, output_W_const9, N_BLOCK, K_BLOCK);
#else
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, kk, 0, output_H_const9, output_W_const9, K_BLOCK, N_BLOCK);
#endif
                        int *restrict argmaxs_pointer = ARGMAXS + ti5(n_block, k + kk, h, w, 0, K_const9, pooled_H_const9, pooled_W_const9, N_BLOCK);
                        float *restrict pooled_outputs_pointer = OUTPUTS + ti5(n_block, k + kk, h, w, 0, K_const9, pooled_H_const9, pooled_W_const9, N_BLOCK);
                        pooled_outputs_pointer[0 : N_BLOCK] = -1.0e6;

                        int outputs_index = h_output*output_W_const9 + w_output;
                        for (y = 0; y < window_height; y++){
                            for (x = 0; x < window_width; x++){
#if K_BLOCK > N_BLOCK
                                if (outputs_pointer[0 : N_BLOCK : K_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK : K_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#else
                                if (outputs_pointer[0 : N_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#endif
                                outputs_index++;
                                outputs_pointer += K_BLOCK*N_BLOCK;
                            }
                            outputs_index += output_W_const9 - window_width;
                            outputs_pointer += (output_W_const9 - window_width)*K_BLOCK*N_BLOCK;
                        }
                    }

                }
            }
        } //nk

    } // pragma_offload
}

int *convolution_layer10(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){
    assert(C == C_const10);
    assert(H == H_const10);
    assert(W == W_const10);
    assert(K == K_const10);
    assert(stride == stride_const10);
    assert(padding == padding_const10);
    assert(pooling_radius == pooling_radius_const10);
    assert(pooling_stride == pooling_stride_const10);
    assert(X == X_const10);
    assert(Y == Y_const10);
    assert(output_H_const10 == (H_const10 + 2*padding_const10 - Y_const10 + 1)/stride_const10);
    assert(output_W_const10 == (W_const10 + 2*padding_const10 - X_const10 + 1)/stride_const10);
    assert(pooled_H_const10 == ceil((output_H_const10 - pooling_radius_const10 + 1.f)/pooling_stride_const10));
    assert(pooled_W_const10 == ceil((output_W_const10 - pooling_radius_const10 + 1.f)/pooling_stride_const10));

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE) \
    in(ARGMAXS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {
        int n_block, n, k_block, k, i, j, h, w, c, c_block, y, x;
        int nk, hw, ij, nkhw;

        // computation of const2ants
        int XWN = (-X_const10 + W_const10)*N,
            HYWN = (H_const10-Y_const10)*W_const10*N;        

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nk, hw, ij, n_block, n, k, k_block, h, w, c, c_block, y, x, i, j) \
            shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XWN, HYWN)
        
        // ~=~=~=~=~=~=~=~= CONVOLUTION ~=~=~=~=~=~=~=~= 
        for (nk = 0; nk < N/N_BLOCK*K_const10/K_BLOCK; nk++){
           
#if K_BLOCK > N_BLOCK
            k_block = nk / (N/N_BLOCK);
            k = k_block*K_BLOCK;
            n_block = md(nk, N/N_BLOCK);
            n = n_block*N_BLOCK;
#else 
            n_block = nk / (K_const10/K_BLOCK);
            n = n_block*N_BLOCK;
            k_block = md(nk, K_const10/K_BLOCK);
            k = k_block*K_BLOCK;
#endif

            SCRATCH[omp_get_thread_num()*output_H_const10*output_W_const10*N_BLOCK*K_BLOCK : output_H_const10*output_W_const10*N_BLOCK*K_BLOCK] = 0.f;

            for (c_block = 0; c_block < C_const10/C_BLOCK; c_block++){
                c = c_block*C_BLOCK;

                for (h = 0; h < output_H_const10; h++){
                    for (w = 0; w < output_W_const10; w++){

#if K_BLOCK > N_BLOCK
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const10, output_W_const10, N_BLOCK, K_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const10, output_W_const10, N_BLOCK, K_BLOCK);
#else
                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const10, output_W_const10, K_BLOCK, N_BLOCK);
                        float *restrict convolutions_next = SCRATCH + ti5(omp_get_thread_num(), h, w+1, 0, 0, output_H_const10, output_W_const10, K_BLOCK, N_BLOCK);
#endif
                __assume_aligned(convolutions, 64);

#pragma unroll
            for(int i = 0; i < (N_BLOCK*K_BLOCK); i+= 16)
            {
                _mm_prefetch((char *)(convolutions_next + i), _MM_HINT_ET0);
            }

                        // if we're not on boundary (i.e not affected by padding)
                        if (w - padding_const10 >= 0 &&
                            h - padding_const10 >= 0 &&
                            output_W_const10 - 1 - w >= padding_const10  &&
                            output_H_const10 - 1 - h >= padding_const10){

#if 1 && defined __MIC__
                // The following loads the a N_BLOCK*K_BLOCK region of SCRATCH space into SIMD registers
                LOAD_OUTPUTS;
#endif

#pragma unroll (C_BLOCK)
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, h - padding_const10, w - padding_const10, 0, C_const10, H_const10, W_const10, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const10, Y_const10, X_const10, K_BLOCK);
                             
                // The idea is to ensure that the working space of INPUTS = [N_BLOCK * C_BLOCK * W * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // The idea is to ensure that the working space of FILTERS = [K_BLOCK * C_BLOCK * X * Y] is small enough to fit in L2 so that we can reuse across [H,W] loops; if not we have a problem; CHECK if performance is low.  
                // TODO: introduce L2 prefetch for INPUTS + ti5(n_block, c, h - padding_const10 + Y_const10, w - padding_const10 + X_const10 + 2 that may help the above problem if it exists
                            for (y = 0; y < Y_const10; ++y){                            
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const10*N_BLOCK), _MM_HINT_T0);
                                for (x = 0; x < X_const10; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                                } // x
                                inputs_pointer += (-X_const10 + W_const10)*N_BLOCK;
                            } // y
                } // cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        }

                        else{
#if 1 && defined __MIC__
                LOAD_OUTPUTS;
#endif
                            for (int cc = 0; cc < C_BLOCK; cc++){
                            float *restrict inputs_pointer = INPUTS + ti5(n_block, c+cc, mx(mn(h-padding_const10, H_const10-1), 0), mx(mn(w-padding_const10, W_const10-1), 0), 0, C_const10, H_const10, W_const10, N_BLOCK);
                        float *restrict filters_pointer = FILTERS + ti5(k_block, c+cc, 0, 0, 0, C_const10, Y_const10, X_const10, K_BLOCK);
                            
                int min_x = mx(0, (padding_const10 - w)); 
                int max_x = mn(X_const10, (W_const10 + padding_const10 - w)); 
                int min_y = mx(0, (padding_const10 - h)); 
                int max_y = mn(Y_const10, (H_const10 + padding_const10 - h)); 

                filters_pointer += min_y*X_const10*K_BLOCK;
                //TODO: I am fairly sure more prefetches are required for FILTERS here...
                            for (y = min_y; y < max_y; ++y){
                                float *restrict inputs_pointer_y = inputs_pointer; // start-of-line pointer
                filters_pointer += min_x*K_BLOCK;
                // This prefetch is only for INPUTS order [N, C, H, W]    
                //_mm_prefetch((char *)(inputs_pointer + W_const10*N_BLOCK), _MM_HINT_T0);
#pragma unroll (X_const10-padding_const10)
#pragma noprefetch
                                for (x = min_x; x < max_x; ++x)
                {
#if 1 && defined __MIC__
                    // This assumes the filters are already in L2...
                    PREFETCH_INPUTS;
                    PREFETCH_FILTERS;
                    LOAD_INPUTS;
                    LOAD_FILTERS;
                    COMPUTE_OUTPUTS;
#endif
                                    filters_pointer += K_BLOCK;
                                    inputs_pointer += N_BLOCK;
                } // x
                filters_pointer += (X_const10 - max_x)*K_BLOCK;
                                inputs_pointer = inputs_pointer_y + W_const10*N_BLOCK; 
                            } //y 
                filters_pointer += (Y_const10 - max_y)*X_const10*K_BLOCK;
                } //cc
#if 1 && defined __MIC__
                STORE_OUTPUTS;
#endif

                        } // if-else
                    } // w
                } // h
            } // c_block


            // ~=~=~=~=~=~=~=~= POOLING ~=~=~=~=~=~=~=~= 
            for (h = 0; h < pooled_H_const10; h++){
                for (w = 0; w < pooled_W_const10; w++){

                    int h_output = h*pooling_stride_const10;
                    int w_output = w*pooling_stride_const10;

                    int window_width = pooling_radius_const10 - mx(w_output + pooling_radius_const10 - output_W_const10, 0);
                    int window_height = pooling_radius_const10 - mx(h_output + pooling_radius_const10 - output_H_const10, 0);
                    
                    for (int kk = 0; kk < K_BLOCK; kk++){
#if K_BLOCK > N_BLOCK 
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, 0, kk, output_H_const10, output_W_const10, N_BLOCK, K_BLOCK);
#else
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, kk, 0, output_H_const10, output_W_const10, K_BLOCK, N_BLOCK);
#endif
                        int *restrict argmaxs_pointer = ARGMAXS + ti5(n_block, k + kk, h, w, 0, K_const10, pooled_H_const10, pooled_W_const10, N_BLOCK);
                        float *restrict pooled_outputs_pointer = OUTPUTS + ti5(n_block, k + kk, h, w, 0, K_const10, pooled_H_const10, pooled_W_const10, N_BLOCK);
                        pooled_outputs_pointer[0 : N_BLOCK] = -1.0e6;

                        int outputs_index = h_output*output_W_const10 + w_output;
                        for (y = 0; y < window_height; y++){
                            for (x = 0; x < window_width; x++){
#if K_BLOCK > N_BLOCK
                                if (outputs_pointer[0 : N_BLOCK : K_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK : K_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#else
                                if (outputs_pointer[0 : N_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
#endif
                                outputs_index++;
                                outputs_pointer += K_BLOCK*N_BLOCK;
                            }
                            outputs_index += output_W_const10 - window_width;
                            outputs_pointer += (output_W_const10 - window_width)*K_BLOCK*N_BLOCK;
                        }
                    }

                }
            }
        } //nk

    } // pragma_offload
}


// gradient for intense blocking/data sturctures

// INPUTS data structure [N, C/C_BLOCK, H, W, C_BLOCK]
// D_FILTERS/FILTERS data structure [C/C_BLOCK, Y/Y_BLOCK, K, Y_BLOCK, X, C_BLOCK]
// ARGMAXS/OUTPUTS/D_POOLED_OUTPUTS data structure [N, pooled_H, pooled_W, K]
// void convolution_gradient_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){
//     assert(C == C_const);
//     assert(H == H_const);
//     assert(W == W_const);
//     assert(K == K_const);
//     assert(padding == padding_const);
//     assert(X == X_const);
//     assert(Y == Y_const);
//     assert(output_H_const == (H_const + 2*padding_const - Y_const + 1)/stride_const);
//     assert(output_W_const == (W_const + 2*padding_const - X_const + 1)/stride_const);
//     assert(pooled_H_const == ceil((output_H_const - pooling_radius_const + 1.f)/pooling_stride_const));
//     assert(pooled_W_const == ceil((output_W_const - pooling_radius_const + 1.f)/pooling_stride_const));


//     #pragma offload target(mic:MIC_DEV) \ 
//     in(INPUTS:length(0) REUSE) \ 
//     in(FILTERS:length(0) REUSE) \ 
//     in(ARGMAXS:length(0) REUSE) \
//     in(D_POOLED_OUTPUTS:length(0) REUSE) \
//     in(D_FILTERS:length(0) REUSE) \
//     in(SCRATCH:length(0) REUSE)
//     {

//         int C_blocks = C_const/C_BLOCK_GRAD;
//         int Y_blocks = Y_const/Y_BLOCK_GRAD;
//         int N_blocks = N/N_BLOCK_GRAD;
//         int H_blocks = output_H_const/H_ARG_BLOCK_GRAD;
//         int W_blocks = output_W_const/W_ARG_BLOCK_GRAD;
    
//     omp_lock_t writelock[C_BLOCK_GRAD*Y_BLOCK_GRAD*16];

//         for(int i = 0; i < C_BLOCK_GRAD*Y_BLOCK_GRAD; i++)      
//             omp_init_lock(&writelock[16*i]);
    
//     double st = omp_get_wtime();
//         #pragma omp parallel for \
//         default(none) \
//         schedule(dynamic) \
//         shared(N, INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, SCRATCH, C_blocks, Y_blocks, N_blocks, H_blocks, W_blocks, writelock)
//         for (int outer = 0; outer < N_blocks * H_blocks * W_blocks * C_blocks * Y_blocks; outer++){
//             int n_block = outer / (H_blocks * W_blocks * C_blocks * Y_blocks);
//             int n = n_block*N_BLOCK_GRAD;

//             int h_block = md(outer, H_blocks * W_blocks * C_blocks * Y_blocks) / (W_blocks * C_blocks * Y_blocks);
//             int h = h_block * H_ARG_BLOCK_GRAD;

//             int w_block = md(outer, W_blocks * C_blocks * Y_blocks) / (C_blocks * Y_blocks);
//             int w = w_block * W_ARG_BLOCK_GRAD;

//             int c_block = md(outer, C_blocks * Y_blocks) / (Y_blocks);
//             int c = c_block * C_BLOCK_GRAD;

//             int y_block = md(outer, Y_blocks);
//             int y = y_block * Y_BLOCK_GRAD;

//         int size_local_scratch = Y_BLOCK_GRAD * X_const * C_BLOCK_GRAD;
//             float *restrict local_scratch = SCRATCH + ti7(n_block*H_blocks*W_blocks + h_block*W_blocks + w_block, c_block,  y_block,  0,      0,        0,       0, 
//                                                                                 C_blocks, Y_blocks, K_const, Y_BLOCK_GRAD, X_const, C_BLOCK_GRAD);
//         local_scratch[ 0 : K_const*Y_BLOCK_GRAD*X_const*C_BLOCK_GRAD] = 0.f;
            
//         // for each element in the pre-pooled outputs, find out whether it is an argmax 
//             for (int n_tot = n; n_tot < n + N_BLOCK_GRAD; n_tot++){
//                 for (int h_arg = h; h_arg < mn(h + H_ARG_BLOCK_GRAD, output_H_const); h_arg++){
//                     if ((y + h_arg - padding_const < 0) || (y + h_arg - padding_const >= H_const)) continue;
//                     int h_start = mx((h_arg + pooling_stride_const - pooling_radius_const)/pooling_stride_const, 0); 
//                     int h_end = mn(h_arg/pooling_stride_const, pooled_H_const - 1); 
//                     int h_inputs = h_arg + y - padding_const;

//                     for (int w_arg = w; w_arg < mn(w + W_ARG_BLOCK_GRAD, output_W_const); w_arg++){
//                         int linear_index = h_arg*output_W_const + w_arg;

//                         // figure out the width of the window that is valid (i.e, not out-of-bounds for INPUTS)
//                         int x_invalid_left = mx(padding_const - w_arg, 0);
//                         int x_invalid_right = mx(w_arg - padding_const + X_const - W_const, 0);
//                         int x_len = mx(X_const - x_invalid_left - x_invalid_right, 0);
//                         int x_len_aligned = x_len / 2 * 2;

//                         int w_start = mx((w_arg + pooling_stride_const - pooling_radius_const)/pooling_stride_const, 0);
//                         int w_end = mn(w_arg/pooling_stride_const, pooled_W_const - 1);
//                         int w_inputs = mx(mn(w_arg - padding_const, W_const - 1), 0);
//             float * restrict INPUTS_pointer_base = INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const,     W_const, C_BLOCK_GRAD);
//             int full_x_line = (x_len == X_const);
//             __declspec(aligned(64)) float pooled_outputs_coefficients[K_const];
//             __declspec(aligned(64)) int ks[K_const];
//             for(int i = 0; i < X_const; i++) _mm_prefetch((char *)(INPUTS_pointer_base + i*16), _MM_HINT_T0); 
//             if(full_x_line)
//             {
//                         // scan over all windows in which this element appears
//                         for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
//                             for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
//                                 int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const, pooled_W_const, K_const);
//                 int cnt = K_const;
//                 #if defined __MIC__
//                 __m512i v_linear_index = _mm512_set1_epi32(linear_index);
//                 cnt = 0;
//                 __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
//                                 for (int k = 0; k < K_const; k+=16){
//                     __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
//                     __m512i v_ARGMAXS = _mm512_undefined_epi32();
//                     __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
//                     v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
//                     v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
//                     __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
//                     v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
//                     v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
//                     _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
//                     _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
//                     _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
//                     _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
//                     cnt += _mm_countbits_32(m); 
//                 }
//                 #endif
//                                 for (int k2 = 0; k2 < cnt; k2++){
//                                     // if this (h_arg, w_arg) is the argmax of the window
//                                     int k = ks[k2];
//                                     float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
//                             float * restrict local_scratch_pointer = local_scratch + k*size_local_scratch;
//                             float * restrict local_scratch_pointer_next = local_scratch + ks[k2+1]*size_local_scratch;
//                     float * restrict INPUTS_pointer = INPUTS_pointer_base;  
//                     #if (C_BLOCK_GRAD == 16) && (defined __MIC__)
//                     __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
//                     #pragma unroll (X_const)
//                     for(int i = 0; i < X_const; i++)
//                     {
//                             _mm_prefetch((char *)(local_scratch_pointer_next + i*16), _MM_HINT_ET0);
//                                 __m512 v_input_0 = _mm512_extload_ps(INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
//                                 __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
//                         v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
//                                 _mm512_extstore_ps((float *)(local_scratch_pointer), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
//                         local_scratch_pointer += 16;
//                         INPUTS_pointer += 16;
//                     }
//                     #else
//                                             local_scratch_pointer[0 : x_len*C_BLOCK_GRAD] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD]; 
//                     #endif
//                                 } // k
//                             } // w_pooled
//                         } // h_pooled
//             }
//             else
//             {
//                         // scan over all windows in which this element appears
//                         for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
//                             for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
//                                 int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const, pooled_W_const, K_const);
//                 int cnt = K_const;
//                 #if defined __MIC__
//                 __m512i v_linear_index = _mm512_set1_epi32(linear_index);
//                 cnt = 0;
//                 __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
//                                 for (int k = 0; k < K_const; k+=16){
//                     __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
//                     __m512i v_ARGMAXS = _mm512_undefined_epi32();
//                     __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
//                     v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
//                     v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
//                     v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
//                     v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
//                     __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
//                     _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
//                     _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
//                     _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
//                     _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
//                     cnt += _mm_countbits_32(m); 
//                 }
//                 #endif
//                                 for (int k2 = 0; k2 < cnt; k2++){
//                                     // if this (h_arg, w_arg) is the argmax of the window
//                                     int k = ks[k2];
//                                     float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
//                             float * restrict local_scratch_pointer = local_scratch + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD,   X_const,  C_BLOCK_GRAD); 
//                             float * restrict local_scratch_pointer_next = local_scratch + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD,   X_const,     C_BLOCK_GRAD);
//                     float * restrict INPUTS_pointer = INPUTS_pointer_base;  
//                     #if (C_BLOCK_GRAD == 16) && defined __MIC__
//                     __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
//                     for(int x = 0; x < x_len_aligned; x+=2)
//                     {
//                             _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
//                             _mm_prefetch((char *)(local_scratch_pointer_next + x*16 + 16), _MM_HINT_ET0);
//                                 __m512 v_input_0   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
//                                 __m512 v_input_1   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
//                                 __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
//                                 __m512 v_scratch_1 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
//                         v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
//                         v_scratch_1 = _mm512_fmadd_ps(v_input_1, v_pooled_outputs_coefficient, v_scratch_1);
//                                 _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD ), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
//                                 _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD + 16), v_scratch_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
//                     }
//                     for(int x = x_len_aligned; x < x_len; x++)
//                     {
//                             _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
//                                 __m512 v_input   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
//                                 __m512 v_scratch = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
//                         v_scratch = _mm512_fmadd_ps(v_input, v_pooled_outputs_coefficient, v_scratch);
//                                 _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD), v_scratch, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
//                     }
//                     #else
//                                             local_scratch_pointer[0 : x_len*C_BLOCK_GRAD] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD]; 
//                     #endif
//                                 } // k
//                             } // w_pooled
//                         } // h_pooled
//             }
                      
//                     } // w_arg
//                 } // h_arg
//             } // nn
//         omp_set_lock(&writelock[16*c_block*y_block]);
//         {
//         D_FILTERS[c_block * y_block * K_const*Y_BLOCK_GRAD*X_const*C_BLOCK_GRAD : K_const*Y_BLOCK_GRAD*X_const*C_BLOCK_GRAD] += local_scratch[ 0 : K_const*Y_BLOCK_GRAD*X_const*C_BLOCK_GRAD];
//         }
//         omp_unset_lock(&writelock[16*c_block*y_block]);
//         } // outer
//     double end = omp_get_wtime();
//     printf("Time first loop = %.5lf\n", (end - st));

// #if 0
//         for (int outer = 0; outer < N_blocks * H_blocks * W_blocks ; outer++){
//         #pragma omp parallel for schedule(dynamic)  
//         for(int inner = 0; inner < C_blocks*Y_blocks*K_const*Y_BLOCK_GRAD; inner++)
//         {
//             float *local_scratch_pointer = SCRATCH + outer*C_blocks*Y_blocks*K_const*Y_BLOCK_GRAD*X_const*C_BLOCK_GRAD + inner*X_const*C_BLOCK_GRAD;
//             float *d_filters_pointer = D_FILTERS + inner*X_const*C_BLOCK_GRAD;
//             d_filters_pointer[0: X_const*C_BLOCK_GRAD] += local_scratch_pointer[0 : X_const*C_BLOCK_GRAD];
//         }
//     }
// #endif

//     } // pragma offload
// }


// INPUTS/D_INPUTS data structure [N, C/C_BLOCK, H, W, C_BLOCK]
// D_FILTERS/FILTERS data structure [C/C_BLOCK, Y/Y_BLOCK, K, Y_BLOCK, X, C_BLOCK]
// ARGMAXS/OUTPUTS/D_POOLED_OUTPUTS data structure [N, pooled_H, pooled_W, K]
void convolution_gradient_layer2(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){
    assert(C == C_const2);
    assert(H == H_const2);
    assert(W == W_const2);
    assert(K == K_const2);
    assert(padding == padding_const2);
    assert(X == X_const2);
    assert(Y == Y_const2);
    assert(output_H_const2 == (H_const2 + 2*padding_const2 - Y_const2 + 1)/stride_const2);
    assert(output_W_const2 == (W_const2 + 2*padding_const2 - X_const2 + 1)/stride_const2);
    assert(pooled_H_const2 == ceil((output_H_const2 - pooling_radius_const2 + 1.f)/pooling_stride_const2));
    assert(pooled_W_const2 == ceil((output_W_const2 - pooling_radius_const2 + 1.f)/pooling_stride_const2));


    #pragma offload target(mic:MIC_DEV) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(D_INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(ARGMAXS:length(0) REUSE) \
    in(D_POOLED_OUTPUTS:length(0) REUSE) \
    in(D_FILTERS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {

        int C_blocks = C_const2/C_BLOCK_GRAD2;
        int Y_blocks = Y_const2/Y_BLOCK_GRAD2;
        int N_blocks = N/N_BLOCK_GRAD2;
        int H_blocks = output_H_const2/H_ARG_BLOCK_GRAD2;
        int W_blocks = output_W_const2/W_ARG_BLOCK_GRAD2;
    
        omp_lock_t writelock[C_blocks*Y_blocks*16];

        for(int i = 0; i < C_blocks*Y_blocks; i++)      
            omp_init_lock(&writelock[16*i]);
    
        // double st = omp_get_wtime();
        #pragma omp parallel for \
        default(none) \
        schedule(dynamic) \
        shared(N, INPUTS, D_INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, SCRATCH, C_blocks, Y_blocks, N_blocks, H_blocks, W_blocks, writelock)
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks * C_blocks * Y_blocks; outer++){
            int n_block = outer / (H_blocks * W_blocks * C_blocks * Y_blocks);
            int n = n_block*N_BLOCK_GRAD2;

            int h_block = md(outer, H_blocks * W_blocks * C_blocks * Y_blocks) / (W_blocks * C_blocks * Y_blocks);
            int h = h_block * H_ARG_BLOCK_GRAD2;

            int w_block = md(outer, W_blocks * C_blocks * Y_blocks) / (C_blocks * Y_blocks);
            int w = w_block * W_ARG_BLOCK_GRAD2;

            int c_block = md(outer, C_blocks * Y_blocks) / (Y_blocks);
            int c = c_block * C_BLOCK_GRAD2;

            int y_block = md(outer, Y_blocks);
            int y = y_block * Y_BLOCK_GRAD2;

        int size_local_scratch = Y_BLOCK_GRAD2 * X_const2 * C_BLOCK_GRAD2;
            float *restrict local_scratch = SCRATCH + ti7(n_block*H_blocks*W_blocks + h_block*W_blocks + w_block, c_block,  y_block,  0,      0,        0,       0, 
                                                                                C_blocks, Y_blocks, K_const2, Y_BLOCK_GRAD2, X_const2, C_BLOCK_GRAD2);
            local_scratch[ 0 : K_const2*Y_BLOCK_GRAD2*X_const2*C_BLOCK_GRAD2] = 0.f;
            
            float * restrict FILTERS_pointer_base = FILTERS + (c_block * Y_blocks + y_block) * K_const2*Y_BLOCK_GRAD2*X_const2*C_BLOCK_GRAD2;
            // for each element in the pre-pooled outputs, find out whether it is an argmax 
            for (int n_tot = n; n_tot < n + N_BLOCK_GRAD2; n_tot++){
                for (int h_arg = h; h_arg < mn(h + H_ARG_BLOCK_GRAD2, output_H_const2); h_arg++){
                    if ((y + h_arg - padding_const2 < 0) || (y + h_arg - padding_const2 >= H_const2)) continue;
                    int h_start = mx((h_arg + pooling_stride_const2 - pooling_radius_const2)/pooling_stride_const2, 0); 
                    int h_end = mn(h_arg/pooling_stride_const2, pooled_H_const2 - 1); 
                    int h_inputs = h_arg + y - padding_const2;

                    for (int w_arg = w; w_arg < mn(w + W_ARG_BLOCK_GRAD2, output_W_const2); w_arg++){
                        int linear_index = h_arg*output_W_const2 + w_arg;

                        // figure out the width of the window that is valid (i.e, not out-of-bounds for INPUTS)
                        int x_invalid_left = mx(padding_const2 - w_arg, 0);
                        int x_invalid_right = mx(w_arg - padding_const2 + X_const2 - W_const2, 0);
                        int x_len = mx(X_const2 - x_invalid_left - x_invalid_right, 0);
                        int x_len_aligned = x_len / 2 * 2;

                        int w_start = mx((w_arg + pooling_stride_const2 - pooling_radius_const2)/pooling_stride_const2, 0);
                        int w_end = mn(w_arg/pooling_stride_const2, pooled_W_const2 - 1);
                        int w_inputs = mx(mn(w_arg - padding_const2, W_const2 - 1), 0);
            
            float * restrict INPUTS_pointer_base = INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const2,     W_const2, C_BLOCK_GRAD2);
            float * restrict D_INPUTS_pointer_base = D_INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const2,     W_const2, C_BLOCK_GRAD2);
                    int full_x_line = (x_len == X_const2);
                    __declspec(aligned(64)) float pooled_outputs_coefficients[K_const2];
                    __declspec(aligned(64)) int ks[K_const2];
                    for(int i = 0; i < X_const2; i++) _mm_prefetch((char *)(INPUTS_pointer_base + i*16), _MM_HINT_T0); 
                    if(full_x_line)
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const2, pooled_W_const2, K_const2);
                                int cnt = K_const2;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const2; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + k*size_local_scratch;
                                            float * restrict local_scratch_pointer_next = local_scratch + ks[k2+1]*size_local_scratch;
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        float * restrict D_INPUTS_pointer = D_INPUTS_pointer_base;  
                                float * restrict FILTERS_pointer = FILTERS_pointer_base + k*size_local_scratch;
                                float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ks[k2+1]*size_local_scratch;
                                        #if (C_BLOCK_GRAD2 == 16) && (defined __MIC__)
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        #pragma unroll (X_const2)
                                        for(int i = 0; i < X_const2; i++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + i*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + i*16), _MM_HINT_T0);
                                                __m512 v_input_0 = _mm512_extload_ps(INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_0 = _mm512_extload_ps(D_INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_0 = _mm512_extload_ps(FILTERS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                            v_d_input_0 = _mm512_fmadd_ps(v_filters_0, v_pooled_outputs_coefficient, v_d_input_0);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer), v_d_input_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                            local_scratch_pointer += 16;
                                            INPUTS_pointer += 16;
                                            FILTERS_pointer += 16;
                                            D_INPUTS_pointer += 16;
                                        }
                                        #else
                                        local_scratch_pointer[0 : x_len*C_BLOCK_GRAD2] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD2]; 
                                        D_INPUTS_pointer[0 : x_len*C_BLOCK_GRAD2] += pooled_outputs_coefficient * FILTERS_pointer[0 : x_len*C_BLOCK_GRAD2]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                    else
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const2, pooled_W_const2, K_const2);
                                int cnt = K_const2;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const2; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD2,   X_const2,  C_BLOCK_GRAD2); 
                                            float * restrict local_scratch_pointer_next = local_scratch + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD2,   X_const2,     C_BLOCK_GRAD2);
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        float * restrict D_INPUTS_pointer = D_INPUTS_pointer_base;  
                                // float * restrict FILTERS_pointer = FILTERS_pointer_base + k*size_local_scratch;
                                // float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ks[k2+1]*size_local_scratch;
                                float * restrict FILTERS_pointer = FILTERS_pointer_base + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD2,   X_const2,  C_BLOCK_GRAD2);
                                float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD2,   X_const2,     C_BLOCK_GRAD2);
                                        #if (C_BLOCK_GRAD2 == 16) && defined __MIC__
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        for(int x = 0; x < x_len_aligned; x+=2)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16 + 16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16), _MM_HINT_T0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16 + 16), _MM_HINT_T0);
                                                __m512 v_input_0   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD2, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_input_1   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD2 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_0 = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD2, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_1 = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD2 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD2, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_1 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD2 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_0 = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD2, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_1 = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD2 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                            v_scratch_1 = _mm512_fmadd_ps(v_input_1, v_pooled_outputs_coefficient, v_scratch_1);
                                            v_d_input_0 = _mm512_fmadd_ps(v_filters_0, v_pooled_outputs_coefficient, v_d_input_0);
                                            v_d_input_1 = _mm512_fmadd_ps(v_filters_1, v_pooled_outputs_coefficient, v_d_input_1);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD2 ), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD2 + 16), v_scratch_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD2 ), v_d_input_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD2 + 16), v_d_input_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        for(int x = x_len_aligned; x < x_len; x++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16), _MM_HINT_T0);
                                                __m512 v_input   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD2, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD2, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD2, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD2, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch = _mm512_fmadd_ps(v_input, v_pooled_outputs_coefficient, v_scratch);
                                            v_d_input = _mm512_fmadd_ps(v_filters, v_pooled_outputs_coefficient, v_d_input);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD2), v_scratch, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD2 ), v_d_input, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        #else
                                                    local_scratch_pointer[0 : x_len*C_BLOCK_GRAD2] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD2]; 
                                        D_INPUTS_pointer[0 : x_len*C_BLOCK_GRAD2] += pooled_outputs_coefficient * FILTERS_pointer[0 : x_len*C_BLOCK_GRAD2]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                      
                    } // w_arg
                } // h_arg
            } // nn

            omp_set_lock(&writelock[16*c_block*y_block]);
            {
            D_FILTERS[(c_block * Y_blocks + y_block) * K_const2*Y_BLOCK_GRAD2*X_const2*C_BLOCK_GRAD2 : K_const2*Y_BLOCK_GRAD2*X_const2*C_BLOCK_GRAD2] += local_scratch[ 0 : K_const2*Y_BLOCK_GRAD2*X_const2*C_BLOCK_GRAD2];
            }
            omp_unset_lock(&writelock[16*c_block*y_block]);

        } // outer
        // double end = omp_get_wtime();
        // printf("Time first loop = %.5lf\n", (end - st));

#if 0
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks ; outer++){
        #pragma omp parallel for schedule(dynamic)  
        for(int inner = 0; inner < C_blocks*Y_blocks*K_const2*Y_BLOCK_GRAD2; inner++)
        {
            float *local_scratch_pointer = SCRATCH + outer*C_blocks*Y_blocks*K_const2*Y_BLOCK_GRAD2*X_const2*C_BLOCK_GRAD2 + inner*X_const2*C_BLOCK_GRAD2;
            float *d_filters_pointer = D_FILTERS + inner*X_const2*C_BLOCK_GRAD2;
            d_filters_pointer[0: X_const2*C_BLOCK_GRAD2] += local_scratch_pointer[0 : X_const2*C_BLOCK_GRAD2];
        }
    }
#endif

    } // pragma offload
}


void convolution_gradient_layer3(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){
    assert(C == C_const3);
    assert(H == H_const3);
    assert(W == W_const3);
    assert(K == K_const3);
    assert(padding == padding_const3);
    assert(X == X_const3);
    assert(Y == Y_const3);
    assert(output_H_const3 == (H_const3 + 2*padding_const3 - Y_const3 + 1)/stride_const3);
    assert(output_W_const3 == (W_const3 + 2*padding_const3 - X_const3 + 1)/stride_const3);
    assert(pooled_H_const3 == ceil((output_H_const3 - pooling_radius_const3 + 1.f)/pooling_stride_const3));
    assert(pooled_W_const3 == ceil((output_W_const3 - pooling_radius_const3 + 1.f)/pooling_stride_const3));


    #pragma offload target(mic:MIC_DEV) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(D_INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(ARGMAXS:length(0) REUSE) \
    in(D_POOLED_OUTPUTS:length(0) REUSE) \
    in(D_FILTERS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {

        int C_blocks = C_const3/C_BLOCK_GRAD3;
        int Y_blocks = Y_const3/Y_BLOCK_GRAD3;
        int N_blocks = N/N_BLOCK_GRAD3;
        int H_blocks = output_H_const3/H_ARG_BLOCK_GRAD3;
        int W_blocks = output_W_const3/W_ARG_BLOCK_GRAD3;
    
        omp_lock_t writelock[C_blocks*Y_blocks*16];

        for(int i = 0; i < C_blocks*Y_blocks; i++)      
            omp_init_lock(&writelock[16*i]);
    
        // double st = omp_get_wtime();
        #pragma omp parallel for \
        default(none) \
        schedule(dynamic) \
        shared(N, INPUTS, D_INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, SCRATCH, C_blocks, Y_blocks, N_blocks, H_blocks, W_blocks, writelock)
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks * C_blocks * Y_blocks; outer++){
            int n_block = outer / (H_blocks * W_blocks * C_blocks * Y_blocks);
            int n = n_block*N_BLOCK_GRAD3;

            int h_block = md(outer, H_blocks * W_blocks * C_blocks * Y_blocks) / (W_blocks * C_blocks * Y_blocks);
            int h = h_block * H_ARG_BLOCK_GRAD3;

            int w_block = md(outer, W_blocks * C_blocks * Y_blocks) / (C_blocks * Y_blocks);
            int w = w_block * W_ARG_BLOCK_GRAD3;

            int c_block = md(outer, C_blocks * Y_blocks) / (Y_blocks);
            int c = c_block * C_BLOCK_GRAD3;

            int y_block = md(outer, Y_blocks);
            int y = y_block * Y_BLOCK_GRAD3;

        int size_local_scratch = Y_BLOCK_GRAD3 * X_const3 * C_BLOCK_GRAD3;
            float *restrict local_scratch = SCRATCH + ti7(n_block*H_blocks*W_blocks + h_block*W_blocks + w_block, c_block,  y_block,  0,      0,        0,       0, 
                                                                                C_blocks, Y_blocks, K_const3, Y_BLOCK_GRAD3, X_const3, C_BLOCK_GRAD3);
            local_scratch[ 0 : K_const3*Y_BLOCK_GRAD3*X_const3*C_BLOCK_GRAD3] = 0.f;
            
            float * restrict FILTERS_pointer_base = FILTERS + (c_block * Y_blocks + y_block) * K_const3*Y_BLOCK_GRAD3*X_const3*C_BLOCK_GRAD3;
            // for each element in the pre-pooled outputs, find out whether it is an argmax 
            for (int n_tot = n; n_tot < n + N_BLOCK_GRAD3; n_tot++){
                for (int h_arg = h; h_arg < mn(h + H_ARG_BLOCK_GRAD3, output_H_const3); h_arg++){
                    if ((y + h_arg - padding_const3 < 0) || (y + h_arg - padding_const3 >= H_const3)) continue;
                    int h_start = mx((h_arg + pooling_stride_const3 - pooling_radius_const3)/pooling_stride_const3, 0); 
                    int h_end = mn(h_arg/pooling_stride_const3, pooled_H_const3 - 1); 
                    int h_inputs = h_arg + y - padding_const3;

                    for (int w_arg = w; w_arg < mn(w + W_ARG_BLOCK_GRAD3, output_W_const3); w_arg++){
                        int linear_index = h_arg*output_W_const3 + w_arg;

                        // figure out the width of the window that is valid (i.e, not out-of-bounds for INPUTS)
                        int x_invalid_left = mx(padding_const3 - w_arg, 0);
                        int x_invalid_right = mx(w_arg - padding_const3 + X_const3 - W_const3, 0);
                        int x_len = mx(X_const3 - x_invalid_left - x_invalid_right, 0);
                        int x_len_aligned = x_len / 2 * 2;

                        int w_start = mx((w_arg + pooling_stride_const3 - pooling_radius_const3)/pooling_stride_const3, 0);
                        int w_end = mn(w_arg/pooling_stride_const3, pooled_W_const3 - 1);
                        int w_inputs = mx(mn(w_arg - padding_const3, W_const3 - 1), 0);
            
            float * restrict INPUTS_pointer_base = INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const3,     W_const3, C_BLOCK_GRAD3);
            float * restrict D_INPUTS_pointer_base = D_INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const3,     W_const3, C_BLOCK_GRAD3);
                    int full_x_line = (x_len == X_const3);
                    __declspec(aligned(64)) float pooled_outputs_coefficients[K_const3];
                    __declspec(aligned(64)) int ks[K_const3];
                    for(int i = 0; i < X_const3; i++) _mm_prefetch((char *)(INPUTS_pointer_base + i*16), _MM_HINT_T0); 
                    if(full_x_line)
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const3, pooled_W_const3, K_const3);
                                int cnt = K_const3;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const3; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + k*size_local_scratch;
                                            float * restrict local_scratch_pointer_next = local_scratch + ks[k2+1]*size_local_scratch;
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        float * restrict D_INPUTS_pointer = D_INPUTS_pointer_base;  
                                float * restrict FILTERS_pointer = FILTERS_pointer_base + k*size_local_scratch;
                                float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ks[k2+1]*size_local_scratch;
                                        #if (C_BLOCK_GRAD3 == 16) && (defined __MIC__)
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        #pragma unroll (X_const3)
                                        for(int i = 0; i < X_const3; i++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + i*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + i*16), _MM_HINT_T0);
                                                __m512 v_input_0 = _mm512_extload_ps(INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_0 = _mm512_extload_ps(D_INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_0 = _mm512_extload_ps(FILTERS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                            v_d_input_0 = _mm512_fmadd_ps(v_filters_0, v_pooled_outputs_coefficient, v_d_input_0);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer), v_d_input_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                            local_scratch_pointer += 16;
                                            INPUTS_pointer += 16;
                                            FILTERS_pointer += 16;
                                            D_INPUTS_pointer += 16;
                                        }
                                        #else
                                        local_scratch_pointer[0 : x_len*C_BLOCK_GRAD3] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD3]; 
                                        D_INPUTS_pointer[0 : x_len*C_BLOCK_GRAD3] += pooled_outputs_coefficient * FILTERS_pointer[0 : x_len*C_BLOCK_GRAD3]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                    else
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const3, pooled_W_const3, K_const3);
                                int cnt = K_const3;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const3; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD3,   X_const3,  C_BLOCK_GRAD3); 
                                            float * restrict local_scratch_pointer_next = local_scratch + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD3,   X_const3,     C_BLOCK_GRAD3);
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        float * restrict D_INPUTS_pointer = D_INPUTS_pointer_base;  
                                // float * restrict FILTERS_pointer = FILTERS_pointer_base + k*size_local_scratch;
                                // float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ks[k2+1]*size_local_scratch;
                                float * restrict FILTERS_pointer = FILTERS_pointer_base + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD3,   X_const3,  C_BLOCK_GRAD3);
                                float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD3,   X_const3,     C_BLOCK_GRAD3);
                                        #if (C_BLOCK_GRAD3 == 16) && defined __MIC__
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        for(int x = 0; x < x_len_aligned; x+=2)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16 + 16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16), _MM_HINT_T0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16 + 16), _MM_HINT_T0);
                                                __m512 v_input_0   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD3, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_input_1   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD3 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_0 = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD3, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_1 = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD3 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD3, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_1 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD3 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_0 = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD3, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_1 = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD3 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                            v_scratch_1 = _mm512_fmadd_ps(v_input_1, v_pooled_outputs_coefficient, v_scratch_1);
                                            v_d_input_0 = _mm512_fmadd_ps(v_filters_0, v_pooled_outputs_coefficient, v_d_input_0);
                                            v_d_input_1 = _mm512_fmadd_ps(v_filters_1, v_pooled_outputs_coefficient, v_d_input_1);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD3 ), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD3 + 16), v_scratch_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD3 ), v_d_input_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD3 + 16), v_d_input_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        for(int x = x_len_aligned; x < x_len; x++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16), _MM_HINT_T0);
                                                __m512 v_input   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD3, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD3, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD3, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD3, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch = _mm512_fmadd_ps(v_input, v_pooled_outputs_coefficient, v_scratch);
                                            v_d_input = _mm512_fmadd_ps(v_filters, v_pooled_outputs_coefficient, v_d_input);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD3), v_scratch, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD3 ), v_d_input, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        #else
                                                    local_scratch_pointer[0 : x_len*C_BLOCK_GRAD3] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD3]; 
                                        D_INPUTS_pointer[0 : x_len*C_BLOCK_GRAD3] += pooled_outputs_coefficient * FILTERS_pointer[0 : x_len*C_BLOCK_GRAD3]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                      
                    } // w_arg
                } // h_arg
            } // nn

            omp_set_lock(&writelock[16*c_block*y_block]);
            {
            D_FILTERS[(c_block * Y_blocks + y_block) * K_const3*Y_BLOCK_GRAD3*X_const3*C_BLOCK_GRAD3 : K_const3*Y_BLOCK_GRAD3*X_const3*C_BLOCK_GRAD3] += local_scratch[ 0 : K_const3*Y_BLOCK_GRAD3*X_const3*C_BLOCK_GRAD3];
            }
            omp_unset_lock(&writelock[16*c_block*y_block]);

        } // outer
        // double end = omp_get_wtime();
        // printf("Time first loop = %.5lf\n", (end - st));

#if 0
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks ; outer++){
        #pragma omp parallel for schedule(dynamic)  
        for(int inner = 0; inner < C_blocks*Y_blocks*K_const2*Y_BLOCK_GRAD2; inner++)
        {
            float *local_scratch_pointer = SCRATCH + outer*C_blocks*Y_blocks*K_const2*Y_BLOCK_GRAD2*X_const2*C_BLOCK_GRAD2 + inner*X_const2*C_BLOCK_GRAD2;
            float *d_filters_pointer = D_FILTERS + inner*X_const2*C_BLOCK_GRAD2;
            d_filters_pointer[0: X_const2*C_BLOCK_GRAD2] += local_scratch_pointer[0 : X_const2*C_BLOCK_GRAD2];
        }
    }
#endif

    } // pragma offload
}

void convolution_gradient_layer4(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){
    assert(C == C_const4);
    assert(H == H_const4);
    assert(W == W_const4);
    assert(K == K_const4);
    assert(padding == padding_const4);
    assert(X == X_const4);
    assert(Y == Y_const4);
    assert(output_H_const4 == (H_const4 + 2*padding_const4 - Y_const4 + 1)/stride_const4);
    assert(output_W_const4 == (W_const4 + 2*padding_const4 - X_const4 + 1)/stride_const4);
    assert(pooled_H_const4 == ceil((output_H_const4 - pooling_radius_const4 + 1.f)/pooling_stride_const4));
    assert(pooled_W_const4 == ceil((output_W_const4 - pooling_radius_const4 + 1.f)/pooling_stride_const4));


    #pragma offload target(mic:MIC_DEV) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(D_INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(ARGMAXS:length(0) REUSE) \
    in(D_POOLED_OUTPUTS:length(0) REUSE) \
    in(D_FILTERS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {

        int C_blocks = C_const4/C_BLOCK_GRAD4;
        int Y_blocks = Y_const4/Y_BLOCK_GRAD4;
        int N_blocks = N/N_BLOCK_GRAD4;
        int H_blocks = output_H_const4/H_ARG_BLOCK_GRAD4;
        int W_blocks = output_W_const4/W_ARG_BLOCK_GRAD4;
    
        omp_lock_t writelock[C_blocks*Y_blocks*16];

        for(int i = 0; i < C_blocks*Y_blocks; i++)      
            omp_init_lock(&writelock[16*i]);
    
        // double st = omp_get_wtime();
        #pragma omp parallel for \
        default(none) \
        schedule(dynamic) \
        shared(N, INPUTS, D_INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, SCRATCH, C_blocks, Y_blocks, N_blocks, H_blocks, W_blocks, writelock)
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks * C_blocks * Y_blocks; outer++){
            int n_block = outer / (H_blocks * W_blocks * C_blocks * Y_blocks);
            int n = n_block*N_BLOCK_GRAD4;

            int h_block = md(outer, H_blocks * W_blocks * C_blocks * Y_blocks) / (W_blocks * C_blocks * Y_blocks);
            int h = h_block * H_ARG_BLOCK_GRAD4;

            int w_block = md(outer, W_blocks * C_blocks * Y_blocks) / (C_blocks * Y_blocks);
            int w = w_block * W_ARG_BLOCK_GRAD4;

            int c_block = md(outer, C_blocks * Y_blocks) / (Y_blocks);
            int c = c_block * C_BLOCK_GRAD4;

            int y_block = md(outer, Y_blocks);
            int y = y_block * Y_BLOCK_GRAD4;

        int size_local_scratch = Y_BLOCK_GRAD4 * X_const4 * C_BLOCK_GRAD4;
            float *restrict local_scratch = SCRATCH + ti7(n_block*H_blocks*W_blocks + h_block*W_blocks + w_block, c_block,  y_block,  0,      0,        0,       0, 
                                                                                C_blocks, Y_blocks, K_const4, Y_BLOCK_GRAD4, X_const4, C_BLOCK_GRAD4);
            local_scratch[ 0 : K_const4*Y_BLOCK_GRAD4*X_const4*C_BLOCK_GRAD4] = 0.f;
            
            float * restrict FILTERS_pointer_base = FILTERS + (c_block * Y_blocks + y_block) * K_const4*Y_BLOCK_GRAD4*X_const4*C_BLOCK_GRAD4;
            // for each element in the pre-pooled outputs, find out whether it is an argmax 
            for (int n_tot = n; n_tot < n + N_BLOCK_GRAD4; n_tot++){
                for (int h_arg = h; h_arg < mn(h + H_ARG_BLOCK_GRAD4, output_H_const4); h_arg++){
                    if ((y + h_arg - padding_const4 < 0) || (y + h_arg - padding_const4 >= H_const4)) continue;
                    int h_start = mx((h_arg + pooling_stride_const4 - pooling_radius_const4)/pooling_stride_const4, 0); 
                    int h_end = mn(h_arg/pooling_stride_const4, pooled_H_const4 - 1); 
                    int h_inputs = h_arg + y - padding_const4;

                    for (int w_arg = w; w_arg < mn(w + W_ARG_BLOCK_GRAD4, output_W_const4); w_arg++){
                        int linear_index = h_arg*output_W_const4 + w_arg;

                        // figure out the width of the window that is valid (i.e, not out-of-bounds for INPUTS)
                        int x_invalid_left = mx(padding_const4 - w_arg, 0);
                        int x_invalid_right = mx(w_arg - padding_const4 + X_const4 - W_const4, 0);
                        int x_len = mx(X_const4 - x_invalid_left - x_invalid_right, 0);
                        int x_len_aligned = x_len / 2 * 2;

                        int w_start = mx((w_arg + pooling_stride_const4 - pooling_radius_const4)/pooling_stride_const4, 0);
                        int w_end = mn(w_arg/pooling_stride_const4, pooled_W_const4 - 1);
                        int w_inputs = mx(mn(w_arg - padding_const4, W_const4 - 1), 0);
            
            float * restrict INPUTS_pointer_base = INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const4,     W_const4, C_BLOCK_GRAD4);
            float * restrict D_INPUTS_pointer_base = D_INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const4,     W_const4, C_BLOCK_GRAD4);
                    int full_x_line = (x_len == X_const4);
                    __declspec(aligned(64)) float pooled_outputs_coefficients[K_const4];
                    __declspec(aligned(64)) int ks[K_const4];
                    for(int i = 0; i < X_const4; i++) _mm_prefetch((char *)(INPUTS_pointer_base + i*16), _MM_HINT_T0); 
                    if(full_x_line)
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const4, pooled_W_const4, K_const4);
                                int cnt = K_const4;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const4; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + k*size_local_scratch;
                                            float * restrict local_scratch_pointer_next = local_scratch + ks[k2+1]*size_local_scratch;
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        float * restrict D_INPUTS_pointer = D_INPUTS_pointer_base;  
                                float * restrict FILTERS_pointer = FILTERS_pointer_base + k*size_local_scratch;
                                float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ks[k2+1]*size_local_scratch;
                                        #if (C_BLOCK_GRAD4 == 16) && (defined __MIC__)
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        #pragma unroll (X_const4)
                                        for(int i = 0; i < X_const4; i++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + i*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + i*16), _MM_HINT_T0);
                                                __m512 v_input_0 = _mm512_extload_ps(INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_0 = _mm512_extload_ps(D_INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_0 = _mm512_extload_ps(FILTERS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                            v_d_input_0 = _mm512_fmadd_ps(v_filters_0, v_pooled_outputs_coefficient, v_d_input_0);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer), v_d_input_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                            local_scratch_pointer += 16;
                                            INPUTS_pointer += 16;
                                            FILTERS_pointer += 16;
                                            D_INPUTS_pointer += 16;
                                        }
                                        #else
                                        local_scratch_pointer[0 : x_len*C_BLOCK_GRAD4] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD4]; 
                                        D_INPUTS_pointer[0 : x_len*C_BLOCK_GRAD4] += pooled_outputs_coefficient * FILTERS_pointer[0 : x_len*C_BLOCK_GRAD4]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                    else
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const4, pooled_W_const4, K_const4);
                                int cnt = K_const4;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const4; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD4,   X_const4,  C_BLOCK_GRAD4); 
                                            float * restrict local_scratch_pointer_next = local_scratch + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD4,   X_const4,     C_BLOCK_GRAD4);
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        float * restrict D_INPUTS_pointer = D_INPUTS_pointer_base;  
                                // float * restrict FILTERS_pointer = FILTERS_pointer_base + k*size_local_scratch;
                                // float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ks[k2+1]*size_local_scratch;
                                float * restrict FILTERS_pointer = FILTERS_pointer_base + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD4,   X_const4,  C_BLOCK_GRAD4);
                                float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD4,   X_const4,     C_BLOCK_GRAD4);
                                        #if (C_BLOCK_GRAD4 == 16) && defined __MIC__
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        for(int x = 0; x < x_len_aligned; x+=2)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16 + 16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16), _MM_HINT_T0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16 + 16), _MM_HINT_T0);
                                                __m512 v_input_0   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD4, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_input_1   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD4 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_0 = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD4, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_1 = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD4 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD4, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_1 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD4 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_0 = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD4, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_1 = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD4 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                            v_scratch_1 = _mm512_fmadd_ps(v_input_1, v_pooled_outputs_coefficient, v_scratch_1);
                                            v_d_input_0 = _mm512_fmadd_ps(v_filters_0, v_pooled_outputs_coefficient, v_d_input_0);
                                            v_d_input_1 = _mm512_fmadd_ps(v_filters_1, v_pooled_outputs_coefficient, v_d_input_1);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD4 ), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD4 + 16), v_scratch_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD4 ), v_d_input_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD4 + 16), v_d_input_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        for(int x = x_len_aligned; x < x_len; x++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16), _MM_HINT_T0);
                                                __m512 v_input   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD4, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD4, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD4, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD4, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch = _mm512_fmadd_ps(v_input, v_pooled_outputs_coefficient, v_scratch);
                                            v_d_input = _mm512_fmadd_ps(v_filters, v_pooled_outputs_coefficient, v_d_input);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD4), v_scratch, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD4 ), v_d_input, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        #else
                                                    local_scratch_pointer[0 : x_len*C_BLOCK_GRAD4] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD4]; 
                                        D_INPUTS_pointer[0 : x_len*C_BLOCK_GRAD4] += pooled_outputs_coefficient * FILTERS_pointer[0 : x_len*C_BLOCK_GRAD4]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                      
                    } // w_arg
                } // h_arg
            } // nn

            omp_set_lock(&writelock[16*c_block*y_block]);
            {
            D_FILTERS[(c_block * Y_blocks + y_block) * K_const4*Y_BLOCK_GRAD4*X_const4*C_BLOCK_GRAD4 : K_const4*Y_BLOCK_GRAD4*X_const4*C_BLOCK_GRAD4] += local_scratch[ 0 : K_const4*Y_BLOCK_GRAD4*X_const4*C_BLOCK_GRAD4];
            }
            omp_unset_lock(&writelock[16*c_block*y_block]);

        } // outer
        // double end = omp_get_wtime();
        // printf("Time first loop = %.5lf\n", (end - st));

#if 0
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks ; outer++){
        #pragma omp parallel for schedule(dynamic)  
        for(int inner = 0; inner < C_blocks*Y_blocks*K_const2*Y_BLOCK_GRAD2; inner++)
        {
            float *local_scratch_pointer = SCRATCH + outer*C_blocks*Y_blocks*K_const2*Y_BLOCK_GRAD2*X_const2*C_BLOCK_GRAD2 + inner*X_const2*C_BLOCK_GRAD2;
            float *d_filters_pointer = D_FILTERS + inner*X_const2*C_BLOCK_GRAD2;
            d_filters_pointer[0: X_const2*C_BLOCK_GRAD2] += local_scratch_pointer[0 : X_const2*C_BLOCK_GRAD2];
        }
    }
#endif

    } // pragma offload
}

void convolution_gradient_layer5(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){
    assert(C == C_const5);
    assert(H == H_const5);
    assert(W == W_const5);
    assert(K == K_const5);
    assert(padding == padding_const5);
    assert(X == X_const5);
    assert(Y == Y_const5);
    assert(output_H_const5 == (H_const5 + 2*padding_const5 - Y_const5 + 1)/stride_const5);
    assert(output_W_const5 == (W_const5 + 2*padding_const5 - X_const5 + 1)/stride_const5);
    assert(pooled_H_const5 == ceil((output_H_const5 - pooling_radius_const5 + 1.f)/pooling_stride_const5));
    assert(pooled_W_const5 == ceil((output_W_const5 - pooling_radius_const5 + 1.f)/pooling_stride_const5));


    #pragma offload target(mic:MIC_DEV) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(D_INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(ARGMAXS:length(0) REUSE) \
    in(D_POOLED_OUTPUTS:length(0) REUSE) \
    in(D_FILTERS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {

        int C_blocks = C_const5/C_BLOCK_GRAD5;
        int Y_blocks = Y_const5/Y_BLOCK_GRAD5;
        int N_blocks = N/N_BLOCK_GRAD5;
        int H_blocks = output_H_const5/H_ARG_BLOCK_GRAD5;
        int W_blocks = output_W_const5/W_ARG_BLOCK_GRAD5;
    
        omp_lock_t writelock[C_blocks*Y_blocks*16];

        for(int i = 0; i < C_blocks*Y_blocks; i++)      
            omp_init_lock(&writelock[16*i]);
    
        // double st = omp_get_wtime();
        #pragma omp parallel for \
        default(none) \
        schedule(dynamic) \
        shared(N, INPUTS, D_INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, SCRATCH, C_blocks, Y_blocks, N_blocks, H_blocks, W_blocks, writelock)
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks * C_blocks * Y_blocks; outer++){
            int n_block = outer / (H_blocks * W_blocks * C_blocks * Y_blocks);
            int n = n_block*N_BLOCK_GRAD5;

            int h_block = md(outer, H_blocks * W_blocks * C_blocks * Y_blocks) / (W_blocks * C_blocks * Y_blocks);
            int h = h_block * H_ARG_BLOCK_GRAD5;

            int w_block = md(outer, W_blocks * C_blocks * Y_blocks) / (C_blocks * Y_blocks);
            int w = w_block * W_ARG_BLOCK_GRAD5;

            int c_block = md(outer, C_blocks * Y_blocks) / (Y_blocks);
            int c = c_block * C_BLOCK_GRAD5;

            int y_block = md(outer, Y_blocks);
            int y = y_block * Y_BLOCK_GRAD5;

        int size_local_scratch = Y_BLOCK_GRAD5 * X_const5 * C_BLOCK_GRAD5;
            float *restrict local_scratch = SCRATCH + ti7(n_block*H_blocks*W_blocks + h_block*W_blocks + w_block, c_block,  y_block,  0,      0,        0,       0, 
                                                                                C_blocks, Y_blocks, K_const5, Y_BLOCK_GRAD5, X_const5, C_BLOCK_GRAD5);
            local_scratch[ 0 : K_const5*Y_BLOCK_GRAD5*X_const5*C_BLOCK_GRAD5] = 0.f;
            
            float * restrict FILTERS_pointer_base = FILTERS + (c_block * Y_blocks + y_block) * K_const5*Y_BLOCK_GRAD5*X_const5*C_BLOCK_GRAD5;
            // for each element in the pre-pooled outputs, find out whether it is an argmax 
            for (int n_tot = n; n_tot < n + N_BLOCK_GRAD5; n_tot++){
                for (int h_arg = h; h_arg < mn(h + H_ARG_BLOCK_GRAD5, output_H_const5); h_arg++){
                    if ((y + h_arg - padding_const5 < 0) || (y + h_arg - padding_const5 >= H_const5)) continue;
                    int h_start = mx((h_arg + pooling_stride_const5 - pooling_radius_const5)/pooling_stride_const5, 0); 
                    int h_end = mn(h_arg/pooling_stride_const5, pooled_H_const5 - 1); 
                    int h_inputs = h_arg + y - padding_const5;

                    for (int w_arg = w; w_arg < mn(w + W_ARG_BLOCK_GRAD5, output_W_const5); w_arg++){
                        int linear_index = h_arg*output_W_const5 + w_arg;

                        // figure out the width of the window that is valid (i.e, not out-of-bounds for INPUTS)
                        int x_invalid_left = mx(padding_const5 - w_arg, 0);
                        int x_invalid_right = mx(w_arg - padding_const5 + X_const5 - W_const5, 0);
                        int x_len = mx(X_const5 - x_invalid_left - x_invalid_right, 0);
                        int x_len_aligned = x_len / 2 * 2;

                        int w_start = mx((w_arg + pooling_stride_const5 - pooling_radius_const5)/pooling_stride_const5, 0);
                        int w_end = mn(w_arg/pooling_stride_const5, pooled_W_const5 - 1);
                        int w_inputs = mx(mn(w_arg - padding_const5, W_const5 - 1), 0);
            
            float * restrict INPUTS_pointer_base = INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const5,     W_const5, C_BLOCK_GRAD5);
            float * restrict D_INPUTS_pointer_base = D_INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const5,     W_const5, C_BLOCK_GRAD5);
                    int full_x_line = (x_len == X_const5);
                    __declspec(aligned(64)) float pooled_outputs_coefficients[K_const5];
                    __declspec(aligned(64)) int ks[K_const5];
                    for(int i = 0; i < X_const5; i++) _mm_prefetch((char *)(INPUTS_pointer_base + i*16), _MM_HINT_T0); 
                    if(full_x_line)
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const5, pooled_W_const5, K_const5);
                                int cnt = K_const5;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const5; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + k*size_local_scratch;
                                            float * restrict local_scratch_pointer_next = local_scratch + ks[k2+1]*size_local_scratch;
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        float * restrict D_INPUTS_pointer = D_INPUTS_pointer_base;  
                                float * restrict FILTERS_pointer = FILTERS_pointer_base + k*size_local_scratch;
                                float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ks[k2+1]*size_local_scratch;
                                        #if (C_BLOCK_GRAD5 == 16) && (defined __MIC__)
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        #pragma unroll (X_const5)
                                        for(int i = 0; i < X_const5; i++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + i*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + i*16), _MM_HINT_T0);
                                                __m512 v_input_0 = _mm512_extload_ps(INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_0 = _mm512_extload_ps(D_INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_0 = _mm512_extload_ps(FILTERS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                            v_d_input_0 = _mm512_fmadd_ps(v_filters_0, v_pooled_outputs_coefficient, v_d_input_0);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer), v_d_input_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                            local_scratch_pointer += 16;
                                            INPUTS_pointer += 16;
                                            FILTERS_pointer += 16;
                                            D_INPUTS_pointer += 16;
                                        }
                                        #else
                                        local_scratch_pointer[0 : x_len*C_BLOCK_GRAD5] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD5]; 
                                        D_INPUTS_pointer[0 : x_len*C_BLOCK_GRAD5] += pooled_outputs_coefficient * FILTERS_pointer[0 : x_len*C_BLOCK_GRAD5]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                    else
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const5, pooled_W_const5, K_const5);
                                int cnt = K_const5;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const5; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD5,   X_const5,  C_BLOCK_GRAD5); 
                                            float * restrict local_scratch_pointer_next = local_scratch + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD5,   X_const5,     C_BLOCK_GRAD5);
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        float * restrict D_INPUTS_pointer = D_INPUTS_pointer_base;  
                                // float * restrict FILTERS_pointer = FILTERS_pointer_base + k*size_local_scratch;
                                // float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ks[k2+1]*size_local_scratch;
                                float * restrict FILTERS_pointer = FILTERS_pointer_base + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD5,   X_const5,  C_BLOCK_GRAD5);
                                float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD5,   X_const5,     C_BLOCK_GRAD5);
                                        #if (C_BLOCK_GRAD5 == 16) && defined __MIC__
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        for(int x = 0; x < x_len_aligned; x+=2)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16 + 16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16), _MM_HINT_T0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16 + 16), _MM_HINT_T0);
                                                __m512 v_input_0   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD5, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_input_1   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD5 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_0 = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD5, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_1 = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD5 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD5, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_1 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD5 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_0 = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD5, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_1 = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD5 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                            v_scratch_1 = _mm512_fmadd_ps(v_input_1, v_pooled_outputs_coefficient, v_scratch_1);
                                            v_d_input_0 = _mm512_fmadd_ps(v_filters_0, v_pooled_outputs_coefficient, v_d_input_0);
                                            v_d_input_1 = _mm512_fmadd_ps(v_filters_1, v_pooled_outputs_coefficient, v_d_input_1);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD5 ), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD5 + 16), v_scratch_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD5 ), v_d_input_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD5 + 16), v_d_input_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        for(int x = x_len_aligned; x < x_len; x++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16), _MM_HINT_T0);
                                                __m512 v_input   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD5, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD5, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD5, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD5, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch = _mm512_fmadd_ps(v_input, v_pooled_outputs_coefficient, v_scratch);
                                            v_d_input = _mm512_fmadd_ps(v_filters, v_pooled_outputs_coefficient, v_d_input);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD5), v_scratch, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD5 ), v_d_input, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        #else
                                                    local_scratch_pointer[0 : x_len*C_BLOCK_GRAD5] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD5]; 
                                        D_INPUTS_pointer[0 : x_len*C_BLOCK_GRAD5] += pooled_outputs_coefficient * FILTERS_pointer[0 : x_len*C_BLOCK_GRAD5]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                      
                    } // w_arg
                } // h_arg
            } // nn

            omp_set_lock(&writelock[16*c_block*y_block]);
            {
            D_FILTERS[(c_block * Y_blocks + y_block) * K_const5*Y_BLOCK_GRAD5*X_const5*C_BLOCK_GRAD5 : K_const5*Y_BLOCK_GRAD5*X_const5*C_BLOCK_GRAD5] += local_scratch[ 0 : K_const5*Y_BLOCK_GRAD5*X_const5*C_BLOCK_GRAD5];
            }
            omp_unset_lock(&writelock[16*c_block*y_block]);

        } // outer
        // double end = omp_get_wtime();
        // printf("Time first loop = %.5lf\n", (end - st));

#if 0
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks ; outer++){
        #pragma omp parallel for schedule(dynamic)  
        for(int inner = 0; inner < C_blocks*Y_blocks*K_const2*Y_BLOCK_GRAD2; inner++)
        {
            float *local_scratch_pointer = SCRATCH + outer*C_blocks*Y_blocks*K_const2*Y_BLOCK_GRAD2*X_const2*C_BLOCK_GRAD2 + inner*X_const2*C_BLOCK_GRAD2;
            float *d_filters_pointer = D_FILTERS + inner*X_const2*C_BLOCK_GRAD2;
            d_filters_pointer[0: X_const2*C_BLOCK_GRAD2] += local_scratch_pointer[0 : X_const2*C_BLOCK_GRAD2];
        }
    }
#endif

    } // pragma offload
}

void convolution_gradient_layer6(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){
    assert(C == C_const6);
    assert(H == H_const6);
    assert(W == W_const6);
    assert(K == K_const6);
    assert(padding == padding_const6);
    assert(X == X_const6);
    assert(Y == Y_const6);
    assert(output_H_const6 == (H_const6 + 2*padding_const6 - Y_const6 + 1)/stride_const6);
    assert(output_W_const6 == (W_const6 + 2*padding_const6 - X_const6 + 1)/stride_const6);
    assert(pooled_H_const6 == ceil((output_H_const6 - pooling_radius_const6 + 1.f)/pooling_stride_const6));
    assert(pooled_W_const6 == ceil((output_W_const6 - pooling_radius_const6 + 1.f)/pooling_stride_const6));


    #pragma offload target(mic:MIC_DEV) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(D_INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(ARGMAXS:length(0) REUSE) \
    in(D_POOLED_OUTPUTS:length(0) REUSE) \
    in(D_FILTERS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {

        int C_blocks = C_const6/C_BLOCK_GRAD6;
        int Y_blocks = Y_const6/Y_BLOCK_GRAD6;
        int N_blocks = N/N_BLOCK_GRAD6;
        int H_blocks = output_H_const6/H_ARG_BLOCK_GRAD6;
        int W_blocks = output_W_const6/W_ARG_BLOCK_GRAD6;
    
        omp_lock_t writelock[C_blocks*Y_blocks*16];

        for(int i = 0; i < C_blocks*Y_blocks; i++)      
            omp_init_lock(&writelock[16*i]);
    
        // double st = omp_get_wtime();
        #pragma omp parallel for \
        default(none) \
        schedule(dynamic) \
        shared(N, INPUTS, D_INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, SCRATCH, C_blocks, Y_blocks, N_blocks, H_blocks, W_blocks, writelock)
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks * C_blocks * Y_blocks; outer++){
            int n_block = outer / (H_blocks * W_blocks * C_blocks * Y_blocks);
            int n = n_block*N_BLOCK_GRAD6;

            int h_block = md(outer, H_blocks * W_blocks * C_blocks * Y_blocks) / (W_blocks * C_blocks * Y_blocks);
            int h = h_block * H_ARG_BLOCK_GRAD6;

            int w_block = md(outer, W_blocks * C_blocks * Y_blocks) / (C_blocks * Y_blocks);
            int w = w_block * W_ARG_BLOCK_GRAD6;

            int c_block = md(outer, C_blocks * Y_blocks) / (Y_blocks);
            int c = c_block * C_BLOCK_GRAD6;

            int y_block = md(outer, Y_blocks);
            int y = y_block * Y_BLOCK_GRAD6;

        int size_local_scratch = Y_BLOCK_GRAD6 * X_const6 * C_BLOCK_GRAD6;
            float *restrict local_scratch = SCRATCH + ti7(n_block*H_blocks*W_blocks + h_block*W_blocks + w_block, c_block,  y_block,  0,      0,        0,       0, 
                                                                                C_blocks, Y_blocks, K_const6, Y_BLOCK_GRAD6, X_const6, C_BLOCK_GRAD6);
            local_scratch[ 0 : K_const6*Y_BLOCK_GRAD6*X_const6*C_BLOCK_GRAD6] = 0.f;
            
            float * restrict FILTERS_pointer_base = FILTERS + (c_block * Y_blocks + y_block) * K_const6*Y_BLOCK_GRAD6*X_const6*C_BLOCK_GRAD6;
            // for each element in the pre-pooled outputs, find out whether it is an argmax 
            for (int n_tot = n; n_tot < n + N_BLOCK_GRAD6; n_tot++){
                for (int h_arg = h; h_arg < mn(h + H_ARG_BLOCK_GRAD6, output_H_const6); h_arg++){
                    if ((y + h_arg - padding_const6 < 0) || (y + h_arg - padding_const6 >= H_const6)) continue;
                    int h_start = mx((h_arg + pooling_stride_const6 - pooling_radius_const6)/pooling_stride_const6, 0); 
                    int h_end = mn(h_arg/pooling_stride_const6, pooled_H_const6 - 1); 
                    int h_inputs = h_arg + y - padding_const6;

                    for (int w_arg = w; w_arg < mn(w + W_ARG_BLOCK_GRAD6, output_W_const6); w_arg++){
                        int linear_index = h_arg*output_W_const6 + w_arg;

                        // figure out the width of the window that is valid (i.e, not out-of-bounds for INPUTS)
                        int x_invalid_left = mx(padding_const6 - w_arg, 0);
                        int x_invalid_right = mx(w_arg - padding_const6 + X_const6 - W_const6, 0);
                        int x_len = mx(X_const6 - x_invalid_left - x_invalid_right, 0);
                        int x_len_aligned = x_len / 2 * 2;

                        int w_start = mx((w_arg + pooling_stride_const6 - pooling_radius_const6)/pooling_stride_const6, 0);
                        int w_end = mn(w_arg/pooling_stride_const6, pooled_W_const6 - 1);
                        int w_inputs = mx(mn(w_arg - padding_const6, W_const6 - 1), 0);
            
            float * restrict INPUTS_pointer_base = INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const6,     W_const6, C_BLOCK_GRAD6);
            float * restrict D_INPUTS_pointer_base = D_INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const6,     W_const6, C_BLOCK_GRAD6);
                    int full_x_line = (x_len == X_const6);
                    __declspec(aligned(64)) float pooled_outputs_coefficients[K_const6];
                    __declspec(aligned(64)) int ks[K_const6];
                    for(int i = 0; i < X_const6; i++) _mm_prefetch((char *)(INPUTS_pointer_base + i*16), _MM_HINT_T0); 
                    if(full_x_line)
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const6, pooled_W_const6, K_const6);
                                int cnt = K_const6;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const6; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + k*size_local_scratch;
                                            float * restrict local_scratch_pointer_next = local_scratch + ks[k2+1]*size_local_scratch;
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        float * restrict D_INPUTS_pointer = D_INPUTS_pointer_base;  
                                float * restrict FILTERS_pointer = FILTERS_pointer_base + k*size_local_scratch;
                                float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ks[k2+1]*size_local_scratch;
                                        #if (C_BLOCK_GRAD6 == 16) && (defined __MIC__)
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        #pragma unroll (X_const6)
                                        for(int i = 0; i < X_const6; i++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + i*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + i*16), _MM_HINT_T0);
                                                __m512 v_input_0 = _mm512_extload_ps(INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_0 = _mm512_extload_ps(D_INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_0 = _mm512_extload_ps(FILTERS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                            v_d_input_0 = _mm512_fmadd_ps(v_filters_0, v_pooled_outputs_coefficient, v_d_input_0);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer), v_d_input_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                            local_scratch_pointer += 16;
                                            INPUTS_pointer += 16;
                                            FILTERS_pointer += 16;
                                            D_INPUTS_pointer += 16;
                                        }
                                        #else
                                        local_scratch_pointer[0 : x_len*C_BLOCK_GRAD6] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD6]; 
                                        D_INPUTS_pointer[0 : x_len*C_BLOCK_GRAD6] += pooled_outputs_coefficient * FILTERS_pointer[0 : x_len*C_BLOCK_GRAD6]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                    else
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const6, pooled_W_const6, K_const6);
                                int cnt = K_const6;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const6; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD6,   X_const6,  C_BLOCK_GRAD6); 
                                            float * restrict local_scratch_pointer_next = local_scratch + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD6,   X_const6,     C_BLOCK_GRAD6);
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        float * restrict D_INPUTS_pointer = D_INPUTS_pointer_base;  
                                // float * restrict FILTERS_pointer = FILTERS_pointer_base + k*size_local_scratch;
                                // float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ks[k2+1]*size_local_scratch;
                                float * restrict FILTERS_pointer = FILTERS_pointer_base + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD6,   X_const6,  C_BLOCK_GRAD6);
                                float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD6,   X_const6,     C_BLOCK_GRAD6);
                                        #if (C_BLOCK_GRAD6 == 16) && defined __MIC__
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        for(int x = 0; x < x_len_aligned; x+=2)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16 + 16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16), _MM_HINT_T0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16 + 16), _MM_HINT_T0);
                                                __m512 v_input_0   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD6, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_input_1   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD6 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_0 = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD6, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_1 = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD6 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD6, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_1 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD6 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_0 = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD6, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_1 = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD6 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                            v_scratch_1 = _mm512_fmadd_ps(v_input_1, v_pooled_outputs_coefficient, v_scratch_1);
                                            v_d_input_0 = _mm512_fmadd_ps(v_filters_0, v_pooled_outputs_coefficient, v_d_input_0);
                                            v_d_input_1 = _mm512_fmadd_ps(v_filters_1, v_pooled_outputs_coefficient, v_d_input_1);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD6 ), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD6 + 16), v_scratch_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD6 ), v_d_input_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD6 + 16), v_d_input_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        for(int x = x_len_aligned; x < x_len; x++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16), _MM_HINT_T0);
                                                __m512 v_input   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD6, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD6, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD6, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD6, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch = _mm512_fmadd_ps(v_input, v_pooled_outputs_coefficient, v_scratch);
                                            v_d_input = _mm512_fmadd_ps(v_filters, v_pooled_outputs_coefficient, v_d_input);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD6), v_scratch, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD6 ), v_d_input, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        #else
                                                    local_scratch_pointer[0 : x_len*C_BLOCK_GRAD6] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD6]; 
                                        D_INPUTS_pointer[0 : x_len*C_BLOCK_GRAD6] += pooled_outputs_coefficient * FILTERS_pointer[0 : x_len*C_BLOCK_GRAD6]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                      
                    } // w_arg
                } // h_arg
            } // nn

            omp_set_lock(&writelock[16*c_block*y_block]);
            {
            D_FILTERS[(c_block * Y_blocks + y_block) * K_const6*Y_BLOCK_GRAD6*X_const6*C_BLOCK_GRAD6 : K_const6*Y_BLOCK_GRAD6*X_const6*C_BLOCK_GRAD6] += local_scratch[ 0 : K_const6*Y_BLOCK_GRAD6*X_const6*C_BLOCK_GRAD6];
            }
            omp_unset_lock(&writelock[16*c_block*y_block]);

        } // outer
        // double end = omp_get_wtime();
        // printf("Time first loop = %.5lf\n", (end - st));

#if 0
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks ; outer++){
        #pragma omp parallel for schedule(dynamic)  
        for(int inner = 0; inner < C_blocks*Y_blocks*K_const2*Y_BLOCK_GRAD2; inner++)
        {
            float *local_scratch_pointer = SCRATCH + outer*C_blocks*Y_blocks*K_const2*Y_BLOCK_GRAD2*X_const2*C_BLOCK_GRAD2 + inner*X_const2*C_BLOCK_GRAD2;
            float *d_filters_pointer = D_FILTERS + inner*X_const2*C_BLOCK_GRAD2;
            d_filters_pointer[0: X_const2*C_BLOCK_GRAD2] += local_scratch_pointer[0 : X_const2*C_BLOCK_GRAD2];
        }
    }
#endif

    } // pragma offload
}

void convolution_gradient_layer7(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){
    assert(C == C_const7);
    assert(H == H_const7);
    assert(W == W_const7);
    assert(K == K_const7);
    assert(padding == padding_const7);
    assert(X == X_const7);
    assert(Y == Y_const7);
    assert(output_H_const7 == (H_const7 + 2*padding_const7 - Y_const7 + 1)/stride_const7);
    assert(output_W_const7 == (W_const7 + 2*padding_const7 - X_const7 + 1)/stride_const7);
    assert(pooled_H_const7 == ceil((output_H_const7 - pooling_radius_const7 + 1.f)/pooling_stride_const7));
    assert(pooled_W_const7 == ceil((output_W_const7 - pooling_radius_const7 + 1.f)/pooling_stride_const7));


    #pragma offload target(mic:MIC_DEV) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(D_INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(ARGMAXS:length(0) REUSE) \
    in(D_POOLED_OUTPUTS:length(0) REUSE) \
    in(D_FILTERS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {

        int C_blocks = C_const7/C_BLOCK_GRAD7;
        int Y_blocks = Y_const7/Y_BLOCK_GRAD7;
        int N_blocks = N/N_BLOCK_GRAD7;
        int H_blocks = output_H_const7/H_ARG_BLOCK_GRAD7;
        int W_blocks = output_W_const7/W_ARG_BLOCK_GRAD7;
    
        omp_lock_t writelock[C_blocks*Y_blocks*16];

        for(int i = 0; i < C_blocks*Y_blocks; i++)      
            omp_init_lock(&writelock[16*i]);
    
        // double st = omp_get_wtime();
        #pragma omp parallel for \
        default(none) \
        schedule(dynamic) \
        shared(N, INPUTS, D_INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, SCRATCH, C_blocks, Y_blocks, N_blocks, H_blocks, W_blocks, writelock)
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks * C_blocks * Y_blocks; outer++){
            int n_block = outer / (H_blocks * W_blocks * C_blocks * Y_blocks);
            int n = n_block*N_BLOCK_GRAD7;

            int h_block = md(outer, H_blocks * W_blocks * C_blocks * Y_blocks) / (W_blocks * C_blocks * Y_blocks);
            int h = h_block * H_ARG_BLOCK_GRAD7;

            int w_block = md(outer, W_blocks * C_blocks * Y_blocks) / (C_blocks * Y_blocks);
            int w = w_block * W_ARG_BLOCK_GRAD7;

            int c_block = md(outer, C_blocks * Y_blocks) / (Y_blocks);
            int c = c_block * C_BLOCK_GRAD7;

            int y_block = md(outer, Y_blocks);
            int y = y_block * Y_BLOCK_GRAD7;

        int size_local_scratch = Y_BLOCK_GRAD7 * X_const7 * C_BLOCK_GRAD7;
            float *restrict local_scratch = SCRATCH + ti7(n_block*H_blocks*W_blocks + h_block*W_blocks + w_block, c_block,  y_block,  0,      0,        0,       0, 
                                                                                C_blocks, Y_blocks, K_const7, Y_BLOCK_GRAD7, X_const7, C_BLOCK_GRAD7);
            local_scratch[ 0 : K_const7*Y_BLOCK_GRAD7*X_const7*C_BLOCK_GRAD7] = 0.f;
            
            float * restrict FILTERS_pointer_base = FILTERS + (c_block * Y_blocks + y_block) * K_const7*Y_BLOCK_GRAD7*X_const7*C_BLOCK_GRAD7;
            // for each element in the pre-pooled outputs, find out whether it is an argmax 
            for (int n_tot = n; n_tot < n + N_BLOCK_GRAD7; n_tot++){
                for (int h_arg = h; h_arg < mn(h + H_ARG_BLOCK_GRAD7, output_H_const7); h_arg++){
                    if ((y + h_arg - padding_const7 < 0) || (y + h_arg - padding_const7 >= H_const7)) continue;
                    int h_start = mx((h_arg + pooling_stride_const7 - pooling_radius_const7)/pooling_stride_const7, 0); 
                    int h_end = mn(h_arg/pooling_stride_const7, pooled_H_const7 - 1); 
                    int h_inputs = h_arg + y - padding_const7;

                    for (int w_arg = w; w_arg < mn(w + W_ARG_BLOCK_GRAD7, output_W_const7); w_arg++){
                        int linear_index = h_arg*output_W_const7 + w_arg;

                        // figure out the width of the window that is valid (i.e, not out-of-bounds for INPUTS)
                        int x_invalid_left = mx(padding_const7 - w_arg, 0);
                        int x_invalid_right = mx(w_arg - padding_const7 + X_const7 - W_const7, 0);
                        int x_len = mx(X_const7 - x_invalid_left - x_invalid_right, 0);
                        int x_len_aligned = x_len / 2 * 2;

                        int w_start = mx((w_arg + pooling_stride_const7 - pooling_radius_const7)/pooling_stride_const7, 0);
                        int w_end = mn(w_arg/pooling_stride_const7, pooled_W_const7 - 1);
                        int w_inputs = mx(mn(w_arg - padding_const7, W_const7 - 1), 0);
            
            float * restrict INPUTS_pointer_base = INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const7,     W_const7, C_BLOCK_GRAD7);
            float * restrict D_INPUTS_pointer_base = D_INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const7,     W_const7, C_BLOCK_GRAD7);
                    int full_x_line = (x_len == X_const7);
                    __declspec(aligned(64)) float pooled_outputs_coefficients[K_const7];
                    __declspec(aligned(64)) int ks[K_const7];
                    for(int i = 0; i < X_const7; i++) _mm_prefetch((char *)(INPUTS_pointer_base + i*16), _MM_HINT_T0); 
                    if(full_x_line)
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const7, pooled_W_const7, K_const7);
                                int cnt = K_const7;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const7; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + k*size_local_scratch;
                                            float * restrict local_scratch_pointer_next = local_scratch + ks[k2+1]*size_local_scratch;
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        float * restrict D_INPUTS_pointer = D_INPUTS_pointer_base;  
                                float * restrict FILTERS_pointer = FILTERS_pointer_base + k*size_local_scratch;
                                float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ks[k2+1]*size_local_scratch;
                                        #if (C_BLOCK_GRAD7 == 16) && (defined __MIC__)
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        #pragma unroll (X_const7)
                                        for(int i = 0; i < X_const7; i++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + i*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + i*16), _MM_HINT_T0);
                                                __m512 v_input_0 = _mm512_extload_ps(INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_0 = _mm512_extload_ps(D_INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_0 = _mm512_extload_ps(FILTERS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                            v_d_input_0 = _mm512_fmadd_ps(v_filters_0, v_pooled_outputs_coefficient, v_d_input_0);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer), v_d_input_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                            local_scratch_pointer += 16;
                                            INPUTS_pointer += 16;
                                            FILTERS_pointer += 16;
                                            D_INPUTS_pointer += 16;
                                        }
                                        #else
                                        local_scratch_pointer[0 : x_len*C_BLOCK_GRAD7] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD7]; 
                                        D_INPUTS_pointer[0 : x_len*C_BLOCK_GRAD7] += pooled_outputs_coefficient * FILTERS_pointer[0 : x_len*C_BLOCK_GRAD7]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                    else
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const7, pooled_W_const7, K_const7);
                                int cnt = K_const7;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const7; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD7,   X_const7,  C_BLOCK_GRAD7); 
                                            float * restrict local_scratch_pointer_next = local_scratch + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD7,   X_const7,     C_BLOCK_GRAD7);
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        float * restrict D_INPUTS_pointer = D_INPUTS_pointer_base;  
                                // float * restrict FILTERS_pointer = FILTERS_pointer_base + k*size_local_scratch;
                                // float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ks[k2+1]*size_local_scratch;
                                float * restrict FILTERS_pointer = FILTERS_pointer_base + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD7,   X_const7,  C_BLOCK_GRAD7);
                                float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD7,   X_const7,     C_BLOCK_GRAD7);
                                        #if (C_BLOCK_GRAD7 == 16) && defined __MIC__
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        for(int x = 0; x < x_len_aligned; x+=2)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16 + 16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16), _MM_HINT_T0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16 + 16), _MM_HINT_T0);
                                                __m512 v_input_0   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD7, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_input_1   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD7 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_0 = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD7, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_1 = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD7 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD7, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_1 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD7 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_0 = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD7, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_1 = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD7 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                            v_scratch_1 = _mm512_fmadd_ps(v_input_1, v_pooled_outputs_coefficient, v_scratch_1);
                                            v_d_input_0 = _mm512_fmadd_ps(v_filters_0, v_pooled_outputs_coefficient, v_d_input_0);
                                            v_d_input_1 = _mm512_fmadd_ps(v_filters_1, v_pooled_outputs_coefficient, v_d_input_1);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD7 ), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD7 + 16), v_scratch_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD7 ), v_d_input_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD7 + 16), v_d_input_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        for(int x = x_len_aligned; x < x_len; x++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16), _MM_HINT_T0);
                                                __m512 v_input   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD7, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD7, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD7, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD7, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch = _mm512_fmadd_ps(v_input, v_pooled_outputs_coefficient, v_scratch);
                                            v_d_input = _mm512_fmadd_ps(v_filters, v_pooled_outputs_coefficient, v_d_input);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD7), v_scratch, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD7 ), v_d_input, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        #else
                                                    local_scratch_pointer[0 : x_len*C_BLOCK_GRAD7] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD7]; 
                                        D_INPUTS_pointer[0 : x_len*C_BLOCK_GRAD7] += pooled_outputs_coefficient * FILTERS_pointer[0 : x_len*C_BLOCK_GRAD7]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                      
                    } // w_arg
                } // h_arg
            } // nn

            omp_set_lock(&writelock[16*c_block*y_block]);
            {
            D_FILTERS[(c_block * Y_blocks + y_block) * K_const7*Y_BLOCK_GRAD7*X_const7*C_BLOCK_GRAD7 : K_const7*Y_BLOCK_GRAD7*X_const7*C_BLOCK_GRAD7] += local_scratch[ 0 : K_const7*Y_BLOCK_GRAD7*X_const7*C_BLOCK_GRAD7];
            }
            omp_unset_lock(&writelock[16*c_block*y_block]);

        } // outer
        // double end = omp_get_wtime();
        // printf("Time first loop = %.5lf\n", (end - st));

#if 0
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks ; outer++){
        #pragma omp parallel for schedule(dynamic)  
        for(int inner = 0; inner < C_blocks*Y_blocks*K_const2*Y_BLOCK_GRAD2; inner++)
        {
            float *local_scratch_pointer = SCRATCH + outer*C_blocks*Y_blocks*K_const2*Y_BLOCK_GRAD2*X_const2*C_BLOCK_GRAD2 + inner*X_const2*C_BLOCK_GRAD2;
            float *d_filters_pointer = D_FILTERS + inner*X_const2*C_BLOCK_GRAD2;
            d_filters_pointer[0: X_const2*C_BLOCK_GRAD2] += local_scratch_pointer[0 : X_const2*C_BLOCK_GRAD2];
        }
    }
#endif

    } // pragma offload
}

void convolution_gradient_layer8(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){
    assert(C == C_const8);
    assert(H == H_const8);
    assert(W == W_const8);
    assert(K == K_const8);
    assert(padding == padding_const8);
    assert(X == X_const8);
    assert(Y == Y_const8);
    assert(output_H_const8 == (H_const8 + 2*padding_const8 - Y_const8 + 1)/stride_const8);
    assert(output_W_const8 == (W_const8 + 2*padding_const8 - X_const8 + 1)/stride_const8);
    assert(pooled_H_const8 == ceil((output_H_const8 - pooling_radius_const8 + 1.f)/pooling_stride_const8));
    assert(pooled_W_const8 == ceil((output_W_const8 - pooling_radius_const8 + 1.f)/pooling_stride_const8));


    #pragma offload target(mic:MIC_DEV) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(D_INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(ARGMAXS:length(0) REUSE) \
    in(D_POOLED_OUTPUTS:length(0) REUSE) \
    in(D_FILTERS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {

        int C_blocks = C_const8/C_BLOCK_GRAD8;
        int Y_blocks = Y_const8/Y_BLOCK_GRAD8;
        int N_blocks = N/N_BLOCK_GRAD8;
        int H_blocks = output_H_const8/H_ARG_BLOCK_GRAD8;
        int W_blocks = output_W_const8/W_ARG_BLOCK_GRAD8;
    
        omp_lock_t writelock[C_blocks*Y_blocks*16];

        for(int i = 0; i < C_blocks*Y_blocks; i++)      
            omp_init_lock(&writelock[16*i]);
    
        // double st = omp_get_wtime();
        #pragma omp parallel for \
        default(none) \
        schedule(dynamic) \
        shared(N, INPUTS, D_INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, SCRATCH, C_blocks, Y_blocks, N_blocks, H_blocks, W_blocks, writelock)
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks * C_blocks * Y_blocks; outer++){
            int n_block = outer / (H_blocks * W_blocks * C_blocks * Y_blocks);
            int n = n_block*N_BLOCK_GRAD8;

            int h_block = md(outer, H_blocks * W_blocks * C_blocks * Y_blocks) / (W_blocks * C_blocks * Y_blocks);
            int h = h_block * H_ARG_BLOCK_GRAD8;

            int w_block = md(outer, W_blocks * C_blocks * Y_blocks) / (C_blocks * Y_blocks);
            int w = w_block * W_ARG_BLOCK_GRAD8;

            int c_block = md(outer, C_blocks * Y_blocks) / (Y_blocks);
            int c = c_block * C_BLOCK_GRAD8;

            int y_block = md(outer, Y_blocks);
            int y = y_block * Y_BLOCK_GRAD8;

        int size_local_scratch = Y_BLOCK_GRAD8 * X_const8 * C_BLOCK_GRAD8;
            float *restrict local_scratch = SCRATCH + ti7(n_block*H_blocks*W_blocks + h_block*W_blocks + w_block, c_block,  y_block,  0,      0,        0,       0, 
                                                                                C_blocks, Y_blocks, K_const8, Y_BLOCK_GRAD8, X_const8, C_BLOCK_GRAD8);
            local_scratch[ 0 : K_const8*Y_BLOCK_GRAD8*X_const8*C_BLOCK_GRAD8] = 0.f;
            
            float * restrict FILTERS_pointer_base = FILTERS + (c_block * Y_blocks + y_block) * K_const8*Y_BLOCK_GRAD8*X_const8*C_BLOCK_GRAD8;
            // for each element in the pre-pooled outputs, find out whether it is an argmax 
            for (int n_tot = n; n_tot < n + N_BLOCK_GRAD8; n_tot++){
                for (int h_arg = h; h_arg < mn(h + H_ARG_BLOCK_GRAD8, output_H_const8); h_arg++){
                    if ((y + h_arg - padding_const8 < 0) || (y + h_arg - padding_const8 >= H_const8)) continue;
                    int h_start = mx((h_arg + pooling_stride_const8 - pooling_radius_const8)/pooling_stride_const8, 0); 
                    int h_end = mn(h_arg/pooling_stride_const8, pooled_H_const8 - 1); 
                    int h_inputs = h_arg + y - padding_const8;

                    for (int w_arg = w; w_arg < mn(w + W_ARG_BLOCK_GRAD8, output_W_const8); w_arg++){
                        int linear_index = h_arg*output_W_const8 + w_arg;

                        // figure out the width of the window that is valid (i.e, not out-of-bounds for INPUTS)
                        int x_invalid_left = mx(padding_const8 - w_arg, 0);
                        int x_invalid_right = mx(w_arg - padding_const8 + X_const8 - W_const8, 0);
                        int x_len = mx(X_const8 - x_invalid_left - x_invalid_right, 0);
                        int x_len_aligned = x_len / 2 * 2;

                        int w_start = mx((w_arg + pooling_stride_const8 - pooling_radius_const8)/pooling_stride_const8, 0);
                        int w_end = mn(w_arg/pooling_stride_const8, pooled_W_const8 - 1);
                        int w_inputs = mx(mn(w_arg - padding_const8, W_const8 - 1), 0);
            
            float * restrict INPUTS_pointer_base = INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const8,     W_const8, C_BLOCK_GRAD8);
            float * restrict D_INPUTS_pointer_base = D_INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const8,     W_const8, C_BLOCK_GRAD8);
                    int full_x_line = (x_len == X_const8);
                    __declspec(aligned(64)) float pooled_outputs_coefficients[K_const8];
                    __declspec(aligned(64)) int ks[K_const8];
                    for(int i = 0; i < X_const8; i++) _mm_prefetch((char *)(INPUTS_pointer_base + i*16), _MM_HINT_T0); 
                    if(full_x_line)
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const8, pooled_W_const8, K_const8);
                                int cnt = K_const8;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const8; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + k*size_local_scratch;
                                            float * restrict local_scratch_pointer_next = local_scratch + ks[k2+1]*size_local_scratch;
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        float * restrict D_INPUTS_pointer = D_INPUTS_pointer_base;  
                                float * restrict FILTERS_pointer = FILTERS_pointer_base + k*size_local_scratch;
                                float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ks[k2+1]*size_local_scratch;
                                        #if (C_BLOCK_GRAD8 == 16) && (defined __MIC__)
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        #pragma unroll (X_const8)
                                        for(int i = 0; i < X_const8; i++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + i*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + i*16), _MM_HINT_T0);
                                                __m512 v_input_0 = _mm512_extload_ps(INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_0 = _mm512_extload_ps(D_INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_0 = _mm512_extload_ps(FILTERS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                            v_d_input_0 = _mm512_fmadd_ps(v_filters_0, v_pooled_outputs_coefficient, v_d_input_0);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer), v_d_input_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                            local_scratch_pointer += 16;
                                            INPUTS_pointer += 16;
                                            FILTERS_pointer += 16;
                                            D_INPUTS_pointer += 16;
                                        }
                                        #else
                                        local_scratch_pointer[0 : x_len*C_BLOCK_GRAD8] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD8]; 
                                        D_INPUTS_pointer[0 : x_len*C_BLOCK_GRAD8] += pooled_outputs_coefficient * FILTERS_pointer[0 : x_len*C_BLOCK_GRAD8]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                    else
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const8, pooled_W_const8, K_const8);
                                int cnt = K_const8;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const8; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD8,   X_const8,  C_BLOCK_GRAD8); 
                                            float * restrict local_scratch_pointer_next = local_scratch + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD8,   X_const8,     C_BLOCK_GRAD8);
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        float * restrict D_INPUTS_pointer = D_INPUTS_pointer_base;  
                                // float * restrict FILTERS_pointer = FILTERS_pointer_base + k*size_local_scratch;
                                // float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ks[k2+1]*size_local_scratch;
                                float * restrict FILTERS_pointer = FILTERS_pointer_base + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD8,   X_const8,  C_BLOCK_GRAD8);
                                float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD8,   X_const8,     C_BLOCK_GRAD8);
                                        #if (C_BLOCK_GRAD8 == 16) && defined __MIC__
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        for(int x = 0; x < x_len_aligned; x+=2)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16 + 16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16), _MM_HINT_T0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16 + 16), _MM_HINT_T0);
                                                __m512 v_input_0   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD8, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_input_1   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD8 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_0 = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD8, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_1 = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD8 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD8, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_1 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD8 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_0 = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD8, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_1 = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD8 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                            v_scratch_1 = _mm512_fmadd_ps(v_input_1, v_pooled_outputs_coefficient, v_scratch_1);
                                            v_d_input_0 = _mm512_fmadd_ps(v_filters_0, v_pooled_outputs_coefficient, v_d_input_0);
                                            v_d_input_1 = _mm512_fmadd_ps(v_filters_1, v_pooled_outputs_coefficient, v_d_input_1);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD8 ), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD8 + 16), v_scratch_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD8 ), v_d_input_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD8 + 16), v_d_input_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        for(int x = x_len_aligned; x < x_len; x++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16), _MM_HINT_T0);
                                                __m512 v_input   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD8, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD8, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD8, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD8, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch = _mm512_fmadd_ps(v_input, v_pooled_outputs_coefficient, v_scratch);
                                            v_d_input = _mm512_fmadd_ps(v_filters, v_pooled_outputs_coefficient, v_d_input);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD8), v_scratch, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD8 ), v_d_input, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        #else
                                                    local_scratch_pointer[0 : x_len*C_BLOCK_GRAD8] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD8]; 
                                        D_INPUTS_pointer[0 : x_len*C_BLOCK_GRAD8] += pooled_outputs_coefficient * FILTERS_pointer[0 : x_len*C_BLOCK_GRAD8]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                      
                    } // w_arg
                } // h_arg
            } // nn

            omp_set_lock(&writelock[16*c_block*y_block]);
            {
            D_FILTERS[(c_block * Y_blocks + y_block) * K_const8*Y_BLOCK_GRAD8*X_const8*C_BLOCK_GRAD8 : K_const8*Y_BLOCK_GRAD8*X_const8*C_BLOCK_GRAD8] += local_scratch[ 0 : K_const8*Y_BLOCK_GRAD8*X_const8*C_BLOCK_GRAD8];
            }
            omp_unset_lock(&writelock[16*c_block*y_block]);

        } // outer
        // double end = omp_get_wtime();
        // printf("Time first loop = %.5lf\n", (end - st));

#if 0
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks ; outer++){
        #pragma omp parallel for schedule(dynamic)  
        for(int inner = 0; inner < C_blocks*Y_blocks*K_const2*Y_BLOCK_GRAD2; inner++)
        {
            float *local_scratch_pointer = SCRATCH + outer*C_blocks*Y_blocks*K_const2*Y_BLOCK_GRAD2*X_const2*C_BLOCK_GRAD2 + inner*X_const2*C_BLOCK_GRAD2;
            float *d_filters_pointer = D_FILTERS + inner*X_const2*C_BLOCK_GRAD2;
            d_filters_pointer[0: X_const2*C_BLOCK_GRAD2] += local_scratch_pointer[0 : X_const2*C_BLOCK_GRAD2];
        }
    }
#endif

    } // pragma offload
}

void convolution_gradient_layer9(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){
    assert(C == C_const9);
    assert(H == H_const9);
    assert(W == W_const9);
    assert(K == K_const9);
    assert(padding == padding_const9);
    assert(X == X_const9);
    assert(Y == Y_const9);
    assert(output_H_const9 == (H_const9 + 2*padding_const9 - Y_const9 + 1)/stride_const9);
    assert(output_W_const9 == (W_const9 + 2*padding_const9 - X_const9 + 1)/stride_const9);
    assert(pooled_H_const9 == ceil((output_H_const9 - pooling_radius_const9 + 1.f)/pooling_stride_const9));
    assert(pooled_W_const9 == ceil((output_W_const9 - pooling_radius_const9 + 1.f)/pooling_stride_const9));


    #pragma offload target(mic:MIC_DEV) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(D_INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(ARGMAXS:length(0) REUSE) \
    in(D_POOLED_OUTPUTS:length(0) REUSE) \
    in(D_FILTERS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {

        int C_blocks = C_const9/C_BLOCK_GRAD9;
        int Y_blocks = Y_const9/Y_BLOCK_GRAD9;
        int N_blocks = N/N_BLOCK_GRAD9;
        int H_blocks = output_H_const9/H_ARG_BLOCK_GRAD9;
        int W_blocks = output_W_const9/W_ARG_BLOCK_GRAD9;
    
        omp_lock_t writelock[C_blocks*Y_blocks*16];

        for(int i = 0; i < C_blocks*Y_blocks; i++)      
            omp_init_lock(&writelock[16*i]);
    
        // double st = omp_get_wtime();
        #pragma omp parallel for \
        default(none) \
        schedule(dynamic) \
        shared(N, INPUTS, D_INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, SCRATCH, C_blocks, Y_blocks, N_blocks, H_blocks, W_blocks, writelock)
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks * C_blocks * Y_blocks; outer++){
            int n_block = outer / (H_blocks * W_blocks * C_blocks * Y_blocks);
            int n = n_block*N_BLOCK_GRAD9;

            int h_block = md(outer, H_blocks * W_blocks * C_blocks * Y_blocks) / (W_blocks * C_blocks * Y_blocks);
            int h = h_block * H_ARG_BLOCK_GRAD9;

            int w_block = md(outer, W_blocks * C_blocks * Y_blocks) / (C_blocks * Y_blocks);
            int w = w_block * W_ARG_BLOCK_GRAD9;

            int c_block = md(outer, C_blocks * Y_blocks) / (Y_blocks);
            int c = c_block * C_BLOCK_GRAD9;

            int y_block = md(outer, Y_blocks);
            int y = y_block * Y_BLOCK_GRAD9;

        int size_local_scratch = Y_BLOCK_GRAD9 * X_const9 * C_BLOCK_GRAD9;
            float *restrict local_scratch = SCRATCH + ti7(n_block*H_blocks*W_blocks + h_block*W_blocks + w_block, c_block,  y_block,  0,      0,        0,       0, 
                                                                                C_blocks, Y_blocks, K_const9, Y_BLOCK_GRAD9, X_const9, C_BLOCK_GRAD9);
            local_scratch[ 0 : K_const9*Y_BLOCK_GRAD9*X_const9*C_BLOCK_GRAD9] = 0.f;
            
            float * restrict FILTERS_pointer_base = FILTERS + (c_block * Y_blocks + y_block) * K_const9*Y_BLOCK_GRAD9*X_const9*C_BLOCK_GRAD9;
            // for each element in the pre-pooled outputs, find out whether it is an argmax 
            for (int n_tot = n; n_tot < n + N_BLOCK_GRAD9; n_tot++){
                for (int h_arg = h; h_arg < mn(h + H_ARG_BLOCK_GRAD9, output_H_const9); h_arg++){
                    if ((y + h_arg - padding_const9 < 0) || (y + h_arg - padding_const9 >= H_const9)) continue;
                    int h_start = mx((h_arg + pooling_stride_const9 - pooling_radius_const9)/pooling_stride_const9, 0); 
                    int h_end = mn(h_arg/pooling_stride_const9, pooled_H_const9 - 1); 
                    int h_inputs = h_arg + y - padding_const9;

                    for (int w_arg = w; w_arg < mn(w + W_ARG_BLOCK_GRAD9, output_W_const9); w_arg++){
                        int linear_index = h_arg*output_W_const9 + w_arg;

                        // figure out the width of the window that is valid (i.e, not out-of-bounds for INPUTS)
                        int x_invalid_left = mx(padding_const9 - w_arg, 0);
                        int x_invalid_right = mx(w_arg - padding_const9 + X_const9 - W_const9, 0);
                        int x_len = mx(X_const9 - x_invalid_left - x_invalid_right, 0);
                        int x_len_aligned = x_len / 2 * 2;

                        int w_start = mx((w_arg + pooling_stride_const9 - pooling_radius_const9)/pooling_stride_const9, 0);
                        int w_end = mn(w_arg/pooling_stride_const9, pooled_W_const9 - 1);
                        int w_inputs = mx(mn(w_arg - padding_const9, W_const9 - 1), 0);
            
            float * restrict INPUTS_pointer_base = INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const9,     W_const9, C_BLOCK_GRAD9);
            float * restrict D_INPUTS_pointer_base = D_INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const9,     W_const9, C_BLOCK_GRAD9);
                    int full_x_line = (x_len == X_const9);
                    __declspec(aligned(64)) float pooled_outputs_coefficients[K_const9];
                    __declspec(aligned(64)) int ks[K_const9];
                    for(int i = 0; i < X_const9; i++) _mm_prefetch((char *)(INPUTS_pointer_base + i*16), _MM_HINT_T0); 
                    if(full_x_line)
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const9, pooled_W_const9, K_const9);
                                int cnt = K_const9;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const9; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + k*size_local_scratch;
                                            float * restrict local_scratch_pointer_next = local_scratch + ks[k2+1]*size_local_scratch;
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        float * restrict D_INPUTS_pointer = D_INPUTS_pointer_base;  
                                float * restrict FILTERS_pointer = FILTERS_pointer_base + k*size_local_scratch;
                                float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ks[k2+1]*size_local_scratch;
                                        #if (C_BLOCK_GRAD9 == 16) && (defined __MIC__)
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        #pragma unroll (X_const9)
                                        for(int i = 0; i < X_const9; i++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + i*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + i*16), _MM_HINT_T0);
                                                __m512 v_input_0 = _mm512_extload_ps(INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_0 = _mm512_extload_ps(D_INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_0 = _mm512_extload_ps(FILTERS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                            v_d_input_0 = _mm512_fmadd_ps(v_filters_0, v_pooled_outputs_coefficient, v_d_input_0);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer), v_d_input_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                            local_scratch_pointer += 16;
                                            INPUTS_pointer += 16;
                                            FILTERS_pointer += 16;
                                            D_INPUTS_pointer += 16;
                                        }
                                        #else
                                        local_scratch_pointer[0 : x_len*C_BLOCK_GRAD9] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD9]; 
                                        D_INPUTS_pointer[0 : x_len*C_BLOCK_GRAD9] += pooled_outputs_coefficient * FILTERS_pointer[0 : x_len*C_BLOCK_GRAD9]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                    else
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const9, pooled_W_const9, K_const9);
                                int cnt = K_const9;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const9; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD9,   X_const9,  C_BLOCK_GRAD9); 
                                            float * restrict local_scratch_pointer_next = local_scratch + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD9,   X_const9,     C_BLOCK_GRAD9);
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        float * restrict D_INPUTS_pointer = D_INPUTS_pointer_base;  
                                // float * restrict FILTERS_pointer = FILTERS_pointer_base + k*size_local_scratch;
                                // float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ks[k2+1]*size_local_scratch;
                                float * restrict FILTERS_pointer = FILTERS_pointer_base + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD9,   X_const9,  C_BLOCK_GRAD9);
                                float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD9,   X_const9,     C_BLOCK_GRAD9);
                                        #if (C_BLOCK_GRAD9 == 16) && defined __MIC__
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        for(int x = 0; x < x_len_aligned; x+=2)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16 + 16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16), _MM_HINT_T0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16 + 16), _MM_HINT_T0);
                                                __m512 v_input_0   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD9, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_input_1   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD9 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_0 = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD9, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_1 = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD9 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD9, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_1 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD9 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_0 = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD9, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_1 = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD9 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                            v_scratch_1 = _mm512_fmadd_ps(v_input_1, v_pooled_outputs_coefficient, v_scratch_1);
                                            v_d_input_0 = _mm512_fmadd_ps(v_filters_0, v_pooled_outputs_coefficient, v_d_input_0);
                                            v_d_input_1 = _mm512_fmadd_ps(v_filters_1, v_pooled_outputs_coefficient, v_d_input_1);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD9 ), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD9 + 16), v_scratch_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD9 ), v_d_input_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD9 + 16), v_d_input_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        for(int x = x_len_aligned; x < x_len; x++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16), _MM_HINT_T0);
                                                __m512 v_input   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD9, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD9, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD9, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD9, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch = _mm512_fmadd_ps(v_input, v_pooled_outputs_coefficient, v_scratch);
                                            v_d_input = _mm512_fmadd_ps(v_filters, v_pooled_outputs_coefficient, v_d_input);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD9), v_scratch, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD9 ), v_d_input, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        #else
                                                    local_scratch_pointer[0 : x_len*C_BLOCK_GRAD9] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD9]; 
                                        D_INPUTS_pointer[0 : x_len*C_BLOCK_GRAD9] += pooled_outputs_coefficient * FILTERS_pointer[0 : x_len*C_BLOCK_GRAD9]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                      
                    } // w_arg
                } // h_arg
            } // nn

            omp_set_lock(&writelock[16*c_block*y_block]);
            {
            D_FILTERS[(c_block * Y_blocks + y_block) * K_const9*Y_BLOCK_GRAD9*X_const9*C_BLOCK_GRAD9 : K_const9*Y_BLOCK_GRAD9*X_const9*C_BLOCK_GRAD9] += local_scratch[ 0 : K_const9*Y_BLOCK_GRAD9*X_const9*C_BLOCK_GRAD9];
            }
            omp_unset_lock(&writelock[16*c_block*y_block]);

        } // outer
        // double end = omp_get_wtime();
        // printf("Time first loop = %.5lf\n", (end - st));

#if 0
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks ; outer++){
        #pragma omp parallel for schedule(dynamic)  
        for(int inner = 0; inner < C_blocks*Y_blocks*K_const2*Y_BLOCK_GRAD2; inner++)
        {
            float *local_scratch_pointer = SCRATCH + outer*C_blocks*Y_blocks*K_const2*Y_BLOCK_GRAD2*X_const2*C_BLOCK_GRAD2 + inner*X_const2*C_BLOCK_GRAD2;
            float *d_filters_pointer = D_FILTERS + inner*X_const2*C_BLOCK_GRAD2;
            d_filters_pointer[0: X_const2*C_BLOCK_GRAD2] += local_scratch_pointer[0 : X_const2*C_BLOCK_GRAD2];
        }
    }
#endif

    } // pragma offload
}

void convolution_gradient_layer10(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){
    assert(C == C_const10);
    assert(H == H_const10);
    assert(W == W_const10);
    assert(K == K_const10);
    assert(padding == padding_const10);
    assert(X == X_const10);
    assert(Y == Y_const10);
    assert(output_H_const10 == (H_const10 + 2*padding_const10 - Y_const10 + 1)/stride_const10);
    assert(output_W_const10 == (W_const10 + 2*padding_const10 - X_const10 + 1)/stride_const10);
    assert(pooled_H_const10 == ceil((output_H_const10 - pooling_radius_const10 + 1.f)/pooling_stride_const10));
    assert(pooled_W_const10 == ceil((output_W_const10 - pooling_radius_const10 + 1.f)/pooling_stride_const10));


    #pragma offload target(mic:MIC_DEV) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(D_INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(ARGMAXS:length(0) REUSE) \
    in(D_POOLED_OUTPUTS:length(0) REUSE) \
    in(D_FILTERS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {

        int C_blocks = C_const10/C_BLOCK_GRAD10;
        int Y_blocks = Y_const10/Y_BLOCK_GRAD10;
        int N_blocks = N/N_BLOCK_GRAD10;
        int H_blocks = output_H_const10/H_ARG_BLOCK_GRAD10;
        int W_blocks = output_W_const10/W_ARG_BLOCK_GRAD10;
    
        omp_lock_t writelock[C_blocks*Y_blocks*16];

        for(int i = 0; i < C_blocks*Y_blocks; i++)      
            omp_init_lock(&writelock[16*i]);
    
        // double st = omp_get_wtime();
        #pragma omp parallel for \
        default(none) \
        schedule(dynamic) \
        shared(N, INPUTS, D_INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, SCRATCH, C_blocks, Y_blocks, N_blocks, H_blocks, W_blocks, writelock)
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks * C_blocks * Y_blocks; outer++){
            int n_block = outer / (H_blocks * W_blocks * C_blocks * Y_blocks);
            int n = n_block*N_BLOCK_GRAD10;

            int h_block = md(outer, H_blocks * W_blocks * C_blocks * Y_blocks) / (W_blocks * C_blocks * Y_blocks);
            int h = h_block * H_ARG_BLOCK_GRAD10;

            int w_block = md(outer, W_blocks * C_blocks * Y_blocks) / (C_blocks * Y_blocks);
            int w = w_block * W_ARG_BLOCK_GRAD10;

            int c_block = md(outer, C_blocks * Y_blocks) / (Y_blocks);
            int c = c_block * C_BLOCK_GRAD10;

            int y_block = md(outer, Y_blocks);
            int y = y_block * Y_BLOCK_GRAD10;

        int size_local_scratch = Y_BLOCK_GRAD10 * X_const10 * C_BLOCK_GRAD10;
            float *restrict local_scratch = SCRATCH + ti7(n_block*H_blocks*W_blocks + h_block*W_blocks + w_block, c_block,  y_block,  0,      0,        0,       0, 
                                                                                C_blocks, Y_blocks, K_const10, Y_BLOCK_GRAD10, X_const10, C_BLOCK_GRAD10);
            local_scratch[ 0 : K_const10*Y_BLOCK_GRAD10*X_const10*C_BLOCK_GRAD10] = 0.f;
            
            float * restrict FILTERS_pointer_base = FILTERS + (c_block * Y_blocks + y_block) * K_const10*Y_BLOCK_GRAD10*X_const10*C_BLOCK_GRAD10;
            // for each element in the pre-pooled outputs, find out whether it is an argmax 
            for (int n_tot = n; n_tot < n + N_BLOCK_GRAD10; n_tot++){
                for (int h_arg = h; h_arg < mn(h + H_ARG_BLOCK_GRAD10, output_H_const10); h_arg++){
                    if ((y + h_arg - padding_const10 < 0) || (y + h_arg - padding_const10 >= H_const10)) continue;
                    int h_start = mx((h_arg + pooling_stride_const10 - pooling_radius_const10)/pooling_stride_const10, 0); 
                    int h_end = mn(h_arg/pooling_stride_const10, pooled_H_const10 - 1); 
                    int h_inputs = h_arg + y - padding_const10;

                    for (int w_arg = w; w_arg < mn(w + W_ARG_BLOCK_GRAD10, output_W_const10); w_arg++){
                        int linear_index = h_arg*output_W_const10 + w_arg;

                        // figure out the width of the window that is valid (i.e, not out-of-bounds for INPUTS)
                        int x_invalid_left = mx(padding_const10 - w_arg, 0);
                        int x_invalid_right = mx(w_arg - padding_const10 + X_const10 - W_const10, 0);
                        int x_len = mx(X_const10 - x_invalid_left - x_invalid_right, 0);
                        int x_len_aligned = x_len / 2 * 2;

                        int w_start = mx((w_arg + pooling_stride_const10 - pooling_radius_const10)/pooling_stride_const10, 0);
                        int w_end = mn(w_arg/pooling_stride_const10, pooled_W_const10 - 1);
                        int w_inputs = mx(mn(w_arg - padding_const10, W_const10 - 1), 0);
            
            float * restrict INPUTS_pointer_base = INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const10,     W_const10, C_BLOCK_GRAD10);
            float * restrict D_INPUTS_pointer_base = D_INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const10,     W_const10, C_BLOCK_GRAD10);
                    int full_x_line = (x_len == X_const10);
                    __declspec(aligned(64)) float pooled_outputs_coefficients[K_const10];
                    __declspec(aligned(64)) int ks[K_const10];
                    for(int i = 0; i < X_const10; i++) _mm_prefetch((char *)(INPUTS_pointer_base + i*16), _MM_HINT_T0); 
                    if(full_x_line)
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const10, pooled_W_const10, K_const10);
                                int cnt = K_const10;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const10; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + k*size_local_scratch;
                                            float * restrict local_scratch_pointer_next = local_scratch + ks[k2+1]*size_local_scratch;
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        float * restrict D_INPUTS_pointer = D_INPUTS_pointer_base;  
                                float * restrict FILTERS_pointer = FILTERS_pointer_base + k*size_local_scratch;
                                float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ks[k2+1]*size_local_scratch;
                                        #if (C_BLOCK_GRAD10 == 16) && (defined __MIC__)
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        #pragma unroll (X_const10)
                                        for(int i = 0; i < X_const10; i++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + i*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + i*16), _MM_HINT_T0);
                                                __m512 v_input_0 = _mm512_extload_ps(INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_0 = _mm512_extload_ps(D_INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_0 = _mm512_extload_ps(FILTERS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                            v_d_input_0 = _mm512_fmadd_ps(v_filters_0, v_pooled_outputs_coefficient, v_d_input_0);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer), v_d_input_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                            local_scratch_pointer += 16;
                                            INPUTS_pointer += 16;
                                            FILTERS_pointer += 16;
                                            D_INPUTS_pointer += 16;
                                        }
                                        #else
                                        local_scratch_pointer[0 : x_len*C_BLOCK_GRAD10] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD10]; 
                                        D_INPUTS_pointer[0 : x_len*C_BLOCK_GRAD10] += pooled_outputs_coefficient * FILTERS_pointer[0 : x_len*C_BLOCK_GRAD10]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                    else
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const10, pooled_W_const10, K_const10);
                                int cnt = K_const10;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const10; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD10,   X_const10,  C_BLOCK_GRAD10); 
                                            float * restrict local_scratch_pointer_next = local_scratch + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD10,   X_const10,     C_BLOCK_GRAD10);
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        float * restrict D_INPUTS_pointer = D_INPUTS_pointer_base;  
                                // float * restrict FILTERS_pointer = FILTERS_pointer_base + k*size_local_scratch;
                                // float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ks[k2+1]*size_local_scratch;
                                float * restrict FILTERS_pointer = FILTERS_pointer_base + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD10,   X_const10,  C_BLOCK_GRAD10);
                                float * restrict FILTERS_pointer_next = FILTERS_pointer_base + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD10,   X_const10,     C_BLOCK_GRAD10);
                                        #if (C_BLOCK_GRAD10 == 16) && defined __MIC__
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        for(int x = 0; x < x_len_aligned; x+=2)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16 + 16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16), _MM_HINT_T0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16 + 16), _MM_HINT_T0);
                                                __m512 v_input_0   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD10, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_input_1   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD10 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_0 = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD10, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input_1 = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD10 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD10, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_1 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD10 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_0 = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD10, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters_1 = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD10 + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                            v_scratch_1 = _mm512_fmadd_ps(v_input_1, v_pooled_outputs_coefficient, v_scratch_1);
                                            v_d_input_0 = _mm512_fmadd_ps(v_filters_0, v_pooled_outputs_coefficient, v_d_input_0);
                                            v_d_input_1 = _mm512_fmadd_ps(v_filters_1, v_pooled_outputs_coefficient, v_d_input_1);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD10 ), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD10 + 16), v_scratch_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD10 ), v_d_input_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD10 + 16), v_d_input_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        for(int x = x_len_aligned; x < x_len; x++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(FILTERS_pointer_next + x*16), _MM_HINT_T0);
                                                __m512 v_input   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD10, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_d_input = _mm512_extload_ps(D_INPUTS_pointer + x*C_BLOCK_GRAD10, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD10, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_filters = _mm512_extload_ps(FILTERS_pointer + x*C_BLOCK_GRAD10, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch = _mm512_fmadd_ps(v_input, v_pooled_outputs_coefficient, v_scratch);
                                            v_d_input = _mm512_fmadd_ps(v_filters, v_pooled_outputs_coefficient, v_d_input);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD10), v_scratch, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(D_INPUTS_pointer + x*C_BLOCK_GRAD10 ), v_d_input, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        #else
                                                    local_scratch_pointer[0 : x_len*C_BLOCK_GRAD10] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD10]; 
                                        D_INPUTS_pointer[0 : x_len*C_BLOCK_GRAD10] += pooled_outputs_coefficient * FILTERS_pointer[0 : x_len*C_BLOCK_GRAD10]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                      
                    } // w_arg
                } // h_arg
            } // nn

            omp_set_lock(&writelock[16*c_block*y_block]);
            {
            D_FILTERS[(c_block * Y_blocks + y_block) * K_const10*Y_BLOCK_GRAD10*X_const10*C_BLOCK_GRAD10 : K_const10*Y_BLOCK_GRAD10*X_const10*C_BLOCK_GRAD10] += local_scratch[ 0 : K_const10*Y_BLOCK_GRAD10*X_const10*C_BLOCK_GRAD10];
            }
            omp_unset_lock(&writelock[16*c_block*y_block]);

        } // outer
        // double end = omp_get_wtime();
        // printf("Time first loop = %.5lf\n", (end - st));

#if 0
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks ; outer++){
        #pragma omp parallel for schedule(dynamic)  
        for(int inner = 0; inner < C_blocks*Y_blocks*K_const2*Y_BLOCK_GRAD2; inner++)
        {
            float *local_scratch_pointer = SCRATCH + outer*C_blocks*Y_blocks*K_const2*Y_BLOCK_GRAD2*X_const2*C_BLOCK_GRAD2 + inner*X_const2*C_BLOCK_GRAD2;
            float *d_filters_pointer = D_FILTERS + inner*X_const2*C_BLOCK_GRAD2;
            d_filters_pointer[0: X_const2*C_BLOCK_GRAD2] += local_scratch_pointer[0 : X_const2*C_BLOCK_GRAD2];
        }
    }
#endif

    } // pragma offload
}

// INPUTS data structure [N, C/C_BLOCK, H, W, C_BLOCK]
// D_FILTERS/FILTERS data structure [K, Y, X, C]
// ARGMAXS/OUTPUTS/D_POOLED_OUTPUTS data structure [N, pooled_H, pooled_W, K]
// void convolution_gradient_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){
//     assert(C == C_const);
//     assert(H == H_const);
//     assert(W == W_const);
//     assert(K == K_const);
//     assert(padding == padding_const);
//     assert(X == X_const);
//     assert(Y == Y_const);
//     assert(output_H_const == (H_const + 2*padding_const - Y_const + 1)/stride_const);
//     assert(output_W_const == (W_const + 2*padding_const - X_const + 1)/stride_const);
//     assert(pooled_H_const == ceil((output_H_const - pooling_radius_const + 1.f)/pooling_stride_const));
//     assert(pooled_W_const == ceil((output_W_const - pooling_radius_const + 1.f)/pooling_stride_const));


//     #pragma offload target(mic:MIC_DEV) \ 
//     in(INPUTS:length(0) REUSE) \ 
//     in(FILTERS:length(0) REUSE) \ 
//     in(ARGMAXS:length(0) REUSE) \
//     in(D_POOLED_OUTPUTS:length(0) REUSE) \
//     in(D_FILTERS:length(0) REUSE) \
//     in(SCRATCH:length(0) REUSE)
//     {

//         int C_blocks = C_const/C_BLOCK_GRAD;
//         int Y_blocks = Y_const/Y_BLOCK_GRAD;
//         int N_blocks = N/N_BLOCK_GRAD;
//         int H_blocks = output_H_const/H_ARG_BLOCK_GRAD;
//         int W_blocks = output_W_const/W_ARG_BLOCK_GRAD;

//         SCRATCH[0 : omp_get_max_threads()*C_blocks*Y_blocks*K_const*Y_BLOCK_GRAD*X_const*C_BLOCK_GRAD] = 0.f;

//         #pragma omp parallel for \
//         default(none) \
//         schedule(dynamic) \
//         shared(N, INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, SCRATCH, C_blocks, Y_blocks, N_blocks, H_blocks, W_blocks)

//         for (int outer = 0; outer < N_blocks * H_blocks * W_blocks * C_blocks * Y_blocks; outer++){
//             int n_block = outer / (H_blocks * W_blocks * C_blocks * Y_blocks);
//             int n = n_block*N_BLOCK_GRAD;

//             int h_block = md(outer, H_blocks * W_blocks * C_blocks * Y_blocks) / (W_blocks * C_blocks * Y_blocks);
//             int h = h_block * H_ARG_BLOCK_GRAD;

//             int w_block = md(outer, W_blocks * C_blocks * Y_blocks) / (C_blocks * Y_blocks);
//             int w = w_block * W_ARG_BLOCK_GRAD;

//             int c_block = md(outer, C_blocks * Y_blocks) / (Y_blocks);
//             int c = c_block * C_BLOCK_GRAD;

//             int y_block = md(outer, Y_blocks);
//             int y = y_block * Y_BLOCK_GRAD;

//             assert(outer == ti5(n_block, h_block, w_block, c_block, y_block, H_blocks, W_blocks, C_blocks, Y_blocks));

//             float *restrict local_scratch = SCRATCH + ti7(omp_get_thread_num(), c_block,  y_block,  0,      0,        0,       0, 
//                                                                                 C_blocks, Y_blocks, K_const, Y_BLOCK_GRAD, X_const, C_BLOCK_GRAD);

//             // for each element in the pre-pooled outputs, find out whether it is an argmax 
//             for (int n_tot = n; n_tot < n + N_BLOCK_GRAD; n_tot++){
//                 for (int h_arg = h; h_arg < mn(h + H_ARG_BLOCK_GRAD, output_H_const); h_arg++){
//                     for (int w_arg = w; w_arg < mn(w + W_ARG_BLOCK_GRAD, output_W_const); w_arg++){

//                         int linear_index = h_arg*output_W_const + w_arg;

//                         int h_start = mx((h_arg + pooling_stride_const - pooling_radius_const)/pooling_stride_const, 0); // ceil((h_arg - pooling_radius + 1)/pooling_stride)
//                         int w_start = mx((w_arg + pooling_stride_const - pooling_radius_const)/pooling_stride_const, 0);

//                         int h_end = mn(h_arg/pooling_stride_const, pooled_H_const - 1); // floor(h_arg/pooling_stride_const)
//                         int w_end = mn(w_arg/pooling_stride_const, pooled_W_const - 1);
                    
//                         // scan over all windows in which this element appears
//                         for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
//                             for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
//                         // for (int h_pooled = 0; h_pooled < pooled_H_const; h_pooled++){
//                             // for (int w_pooled = 0; w_pooled < pooled_W_const; w_pooled++){
//                                 for (int k = 0; k < K_const; k++){

//                                     int pooled_index = ti(n_tot, h_pooled, w_pooled, k, pooled_H_const, pooled_W_const, K_const);

//                                     // if this (h_arg, w_arg) is the argmax of the window
//                                     if (ARGMAXS[pooled_index] == linear_index){
//                                         float pooled_outputs_coefficient = D_POOLED_OUTPUTS[pooled_index];

//                                         // figure out the width of the window that is valid (i.e, not out-of-bounds for INPUTS)
//                                         int x_invalid_left = mx(padding_const - w_arg, 0);
//                                         int x_invalid_right = mx(w_arg - padding_const + X_const - W_const, 0);
//                                         int x_len = mx(X_const - x_invalid_left - x_invalid_right, 0);
                                        
//                                         int w_inputs = mx(mn(w_arg - padding_const, W_const - 1), 0);
                                        
//                                         for (int yy = 0; yy < Y_BLOCK_GRAD; yy++){
//                                             if ((y + yy + h_arg - padding_const >= 0) && (y + yy + h_arg - padding_const < H_const)){
//                                                 int h_inputs = h_arg + y + yy - padding_const;
                                              
//                                                 local_scratch[ti(k,   yy,        x_invalid_left,   0, 
//                                                                       Y_BLOCK_GRAD,   X_const,          C_BLOCK_GRAD) : x_len*C_BLOCK_GRAD] += 
//                                                     pooled_outputs_coefficient *
//                                                     INPUTS[ti5(n_tot,  c_block,    h_inputs,    w_inputs,   0, 
//                                                                        C_blocks,   H_const,     W_const,    C_BLOCK_GRAD) : x_len*C_BLOCK_GRAD];
//                                             } // if
//                                         } //yy
//                                     } // if

//                                 } // k
//                             } // w_pooled
//                         } // h_pooled
                      
//                     } // w_arg
//                 } // h_arg
//             } // nn

//         } // outer


//         for (int thread = 0; thread < omp_get_max_threads(); thread++){

//             #pragma omp parallel for
//             for (int k = 0; k < K_const; k++){
//                 for (int c_block = 0; c_block < C_blocks; c_block++){
//                     int c = c_block*C_BLOCK_GRAD;

//                     for (int y = 0; y < Y_const; y++){
//                         int y_block = y / Y_BLOCK_GRAD;
//                         int yy = md(y, Y_BLOCK_GRAD);

//                         for (int x = 0; x < X_const; x++){
//                             D_FILTERS[ti(k, y, x, c, Y_const, X_const, C_const) : C_BLOCK_GRAD] += 
//                              SCRATCH[ti7(thread,  c_block,  y_block,  k,       yy,      x,       0, 
//                                                   C_blocks, Y_blocks, K_const, Y_BLOCK_GRAD, X_const, C_BLOCK_GRAD) : C_BLOCK_GRAD];
//                         }

//                     }
//                 }
//             }
//         }

//     } // pragma offload
// }

void convolution_gradient_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){
    assert(C == C_const);
    assert(H == H_const);
    assert(W == W_const);
    assert(K == K_const);
    assert(padding == padding_const);
    assert(X == X_const);
    assert(Y == Y_const);
    assert(output_H_const == (H_const + 2*padding_const - Y_const + 1)/stride_const);
    assert(output_W_const == (W_const + 2*padding_const - X_const + 1)/stride_const);
    assert(pooled_H_const == ceil((output_H_const - pooling_radius_const + 1.f)/pooling_stride_const));
    assert(pooled_W_const == ceil((output_W_const - pooling_radius_const + 1.f)/pooling_stride_const));


    #pragma offload target(mic:MIC_DEV) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(D_INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(ARGMAXS:length(0) REUSE) \
    in(D_POOLED_OUTPUTS:length(0) REUSE) \
    in(D_FILTERS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {

        int C_blocks = C_const/C_BLOCK_GRAD;
        int Y_blocks = Y_const/Y_BLOCK_GRAD;
        int N_blocks = N/N_BLOCK_GRAD;
        int H_blocks = output_H_const/H_ARG_BLOCK_GRAD;
        int W_blocks = output_W_const/W_ARG_BLOCK_GRAD;
    
        omp_lock_t writelock[C_blocks*Y_blocks*16];

        for(int i = 0; i < C_blocks*Y_blocks; i++)      
            omp_init_lock(&writelock[16*i]);
    
        // double st = omp_get_wtime();
        #pragma omp parallel for \
        default(none) \
        schedule(dynamic) \
        shared(N, INPUTS, D_INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, SCRATCH, C_blocks, Y_blocks, N_blocks, H_blocks, W_blocks, writelock)
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks * C_blocks * Y_blocks; outer++){
            int n_block = outer / (H_blocks * W_blocks * C_blocks * Y_blocks);
            int n = n_block*N_BLOCK_GRAD;

            int h_block = md(outer, H_blocks * W_blocks * C_blocks * Y_blocks) / (W_blocks * C_blocks * Y_blocks);
            int h = h_block * H_ARG_BLOCK_GRAD;

            int w_block = md(outer, W_blocks * C_blocks * Y_blocks) / (C_blocks * Y_blocks);
            int w = w_block * W_ARG_BLOCK_GRAD;

            int c_block = md(outer, C_blocks * Y_blocks) / (Y_blocks);
            int c = c_block * C_BLOCK_GRAD;

            int y_block = md(outer, Y_blocks);
            int y = y_block * Y_BLOCK_GRAD;

        int size_local_scratch = Y_BLOCK_GRAD * X_const * C_BLOCK_GRAD;
            float *restrict local_scratch = SCRATCH + ti7(n_block*H_blocks*W_blocks + h_block*W_blocks + w_block, c_block,  y_block,  0,      0,        0,       0, 
                                                                                C_blocks, Y_blocks, K_const, Y_BLOCK_GRAD, X_const, C_BLOCK_GRAD);
            local_scratch[ 0 : K_const*Y_BLOCK_GRAD*X_const*C_BLOCK_GRAD] = 0.f;
            
            // for each element in the pre-pooled outputs, find out whether it is an argmax 
            for (int n_tot = n; n_tot < n + N_BLOCK_GRAD; n_tot++){
                for (int h_arg = h; h_arg < mn(h + H_ARG_BLOCK_GRAD, output_H_const); h_arg++){
                    if ((y + h_arg - padding_const < 0) || (y + h_arg - padding_const >= H_const)) continue;
                    int h_start = mx((h_arg + pooling_stride_const - pooling_radius_const)/pooling_stride_const, 0); 
                    int h_end = mn(h_arg/pooling_stride_const, pooled_H_const - 1); 
                    int h_inputs = h_arg + y - padding_const;

                    for (int w_arg = w; w_arg < mn(w + W_ARG_BLOCK_GRAD, output_W_const); w_arg++){
                        int linear_index = h_arg*output_W_const + w_arg;

                        // figure out the width of the window that is valid (i.e, not out-of-bounds for INPUTS)
                        int x_invalid_left = mx(padding_const - w_arg, 0);
                        int x_invalid_right = mx(w_arg - padding_const + X_const - W_const, 0);
                        int x_len = mx(X_const - x_invalid_left - x_invalid_right, 0);
                        int x_len_aligned = x_len / 2 * 2;

                        int w_start = mx((w_arg + pooling_stride_const - pooling_radius_const)/pooling_stride_const, 0);
                        int w_end = mn(w_arg/pooling_stride_const, pooled_W_const - 1);
                        int w_inputs = mx(mn(w_arg - padding_const, W_const - 1), 0);
            
            float * restrict INPUTS_pointer_base = INPUTS + ti5(n_tot,  c_block,  h_inputs,  w_inputs,   0, C_blocks,   H_const,     W_const, C_BLOCK_GRAD);
                    int full_x_line = (x_len == X_const);
                    __declspec(aligned(64)) float pooled_outputs_coefficients[K_const];
                    __declspec(aligned(64)) int ks[K_const];
                    for(int i = 0; i < X_const; i++) _mm_prefetch((char *)(INPUTS_pointer_base + i*16), _MM_HINT_T0); 
                    if(full_x_line)
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const, pooled_W_const, K_const);
                                int cnt = K_const;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + k*size_local_scratch;
                                            float * restrict local_scratch_pointer_next = local_scratch + ks[k2+1]*size_local_scratch;
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        #if (C_BLOCK_GRAD == 16) && (defined __MIC__)
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        #pragma unroll (X_const)
                                        for(int i = 0; i < X_const; i++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + i*16), _MM_HINT_ET0);
                                                __m512 v_input_0 = _mm512_extload_ps(INPUTS_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                            local_scratch_pointer += 16;
                                            INPUTS_pointer += 16;
                                        }
                                        #else
                                        local_scratch_pointer[0 : x_len*C_BLOCK_GRAD] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                    else
                    {
                            // scan over all windows in which this element appears
                            for (int h_pooled = h_start; h_pooled <= h_end; h_pooled++){
                                    for (int w_pooled = w_start; w_pooled <= w_end; w_pooled++){
                                        int pooled_index = ti(n_tot, h_pooled, w_pooled, 0, pooled_H_const, pooled_W_const, K_const);
                                int cnt = K_const;
                                #if defined __MIC__
                                __m512i v_linear_index = _mm512_set1_epi32(linear_index);
                                cnt = 0;
                                __m512i v_0to15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                                        for (int k = 0; k < K_const; k+=16){
                                        __m512i v_k = _mm512_add_epi32(v_0to15, _mm512_set1_epi32(k));
                                        __m512i v_ARGMAXS = _mm512_undefined_epi32();
                                        __m512 v_D_POOLED_OUTPUTS = _mm512_undefined_ps();
                                        v_ARGMAXS = _mm512_loadunpacklo_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k);
                                        v_ARGMAXS = _mm512_loadunpackhi_epi32(v_ARGMAXS, ARGMAXS + pooled_index + k + 16);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpacklo_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k);
                                        v_D_POOLED_OUTPUTS = _mm512_loadunpackhi_ps(v_D_POOLED_OUTPUTS, D_POOLED_OUTPUTS + pooled_index + k + 16);
                                        __mmask16 m = _mm512_cmpeq_epi32_mask(v_ARGMAXS, v_linear_index);
                                        _mm512_mask_packstorelo_ps((float *)(pooled_outputs_coefficients + cnt), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorehi_ps((float *)(pooled_outputs_coefficients + cnt + 16), m, v_D_POOLED_OUTPUTS); 
                                        _mm512_mask_packstorelo_epi32((int *)(ks + cnt), m, v_k); 
                                        _mm512_mask_packstorehi_epi32((int *)(ks + cnt + 16), m, v_k); 
                                        cnt += _mm_countbits_32(m); 
                                }
                                #endif
                                        for (int k2 = 0; k2 < cnt; k2++){
                                                // if this (h_arg, w_arg) is the argmax of the window
                                                int k = ks[k2];
                                                float pooled_outputs_coefficient = pooled_outputs_coefficients[k2];
                                            float * restrict local_scratch_pointer = local_scratch + ti(k,   0,   x_invalid_left,   0, Y_BLOCK_GRAD,   X_const,  C_BLOCK_GRAD); 
                                            float * restrict local_scratch_pointer_next = local_scratch + ti(ks[k2+1],  0,  0,   0, Y_BLOCK_GRAD,   X_const,     C_BLOCK_GRAD);
                                        float * restrict INPUTS_pointer = INPUTS_pointer_base;  
                                        #if (C_BLOCK_GRAD == 16) && defined __MIC__
                                        __m512 v_pooled_outputs_coefficient = _mm512_set1_ps(pooled_outputs_coefficient);
                                        for(int x = 0; x < x_len_aligned; x+=2)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16 + 16), _MM_HINT_ET0);
                                                __m512 v_input_0   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_input_1   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_0 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch_1 = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch_0 = _mm512_fmadd_ps(v_input_0, v_pooled_outputs_coefficient, v_scratch_0);
                                            v_scratch_1 = _mm512_fmadd_ps(v_input_1, v_pooled_outputs_coefficient, v_scratch_1);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD ), v_scratch_0, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD + 16), v_scratch_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        for(int x = x_len_aligned; x < x_len; x++)
                                        {
                                                _mm_prefetch((char *)(local_scratch_pointer_next + x*16), _MM_HINT_ET0);
                                                __m512 v_input   = _mm512_extload_ps(INPUTS_pointer + x*C_BLOCK_GRAD, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                                __m512 v_scratch = _mm512_extload_ps(local_scratch_pointer + x*C_BLOCK_GRAD, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                                            v_scratch = _mm512_fmadd_ps(v_input, v_pooled_outputs_coefficient, v_scratch);
                                                _mm512_extstore_ps((float *)(local_scratch_pointer + x*C_BLOCK_GRAD), v_scratch, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
                                        }
                                        #else
                                                    local_scratch_pointer[0 : x_len*C_BLOCK_GRAD] += pooled_outputs_coefficient * INPUTS_pointer[0 : x_len*C_BLOCK_GRAD]; 
                                        #endif
                                        } // k
                                    } // w_pooled
                            } // h_pooled
                    }
                      
                    } // w_arg
                } // h_arg
            } // nn

            omp_set_lock(&writelock[16*c_block*y_block]);
            {
            D_FILTERS[(c_block * Y_blocks + y_block) * K_const*Y_BLOCK_GRAD*X_const*C_BLOCK_GRAD : K_const*Y_BLOCK_GRAD*X_const*C_BLOCK_GRAD] += local_scratch[ 0 : K_const*Y_BLOCK_GRAD*X_const*C_BLOCK_GRAD];
            }
            omp_unset_lock(&writelock[16*c_block*y_block]);

        } // outer
        // double end = omp_get_wtime();
        // printf("Time first loop = %.5lf\n", (end - st));

#if 0
        for (int outer = 0; outer < N_blocks * H_blocks * W_blocks ; outer++){
        #pragma omp parallel for schedule(dynamic)  
        for(int inner = 0; inner < C_blocks*Y_blocks*K_const*Y_BLOCK_GRAD; inner++)
        {
            float *local_scratch_pointer = SCRATCH + outer*C_blocks*Y_blocks*K_const*Y_BLOCK_GRAD*X_const*C_BLOCK_GRAD + inner*X_const*C_BLOCK_GRAD;
            float *d_filters_pointer = D_FILTERS + inner*X_const*C_BLOCK_GRAD;
            d_filters_pointer[0: X_const*C_BLOCK_GRAD] += local_scratch_pointer[0 : X_const*C_BLOCK_GRAD];
        }
    }
#endif

    } // pragma offload
}


// local filtering after interleaving N and K and blocking C, before intrinsics
int *local_filtering_layer1(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){

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
    assert(pooled_H_const == ceil((output_H_const - pooling_radius_const + 1.f)/pooling_stride_const));
    assert(pooled_W_const == ceil((output_W_const - pooling_radius_const + 1.f)/pooling_stride_const));

    #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
    in(INPUTS:length(0) REUSE) \ 
    in(FILTERS:length(0) REUSE) \ 
    in(OUTPUTS:length(0) REUSE) \
    in(ARGMAXS:length(0) REUSE) \
    in(SCRATCH:length(0) REUSE)
    {
        int n_block, n, k_block, k, i, j, h, w, c, c_block, y, x;
        int nk, hw, ij, nkhw;

        int XWN = (-X_const + W_const)*N,
            HYWN = (H_const-Y_const)*W_const*N;        

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(nk, hw, ij, n_block, n, k, k_block, h, w, c, c_block, y, x, i, j) \
            shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XWN, HYWN)
        
        // ~=~=~=~=~=~=~=~= CONVOLUTION ~=~=~=~=~=~=~=~= 
        for (nk = 0; nk < N/N_BLOCK*K_const/K_BLOCK; nk++){
            
            n_block = nk / (K_const/K_BLOCK);
            n = n_block*N_BLOCK;
            k_block = md(nk, K_const/K_BLOCK);
            k = k_block*K_BLOCK;

            SCRATCH[omp_get_thread_num()*output_H_const*output_W_const*N_BLOCK*K_BLOCK : output_H_const*output_W_const*N_BLOCK*K_BLOCK] = 0.f;

            for (c_block = 0; c_block < C_const/C_BLOCK; c_block++){
                c = c_block*C_BLOCK;

                for (h = 0; h < output_H_const; h++){
                    for (w = 0; w < output_W_const; w++){

                        float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const, output_W_const, K_BLOCK, N_BLOCK);
                              
                        for (y = 0; y < Y_const; ++y){                                
                            for (x = 0; x < X_const; ++x){
                                
                                if ((h + y - padding_const >= 0) && (h + y - padding_const < H_const) && (w + x - padding_const >= 0) && (w + x - padding_const < W_const)){
                                    float *restrict filters_pointer = FILTERS + ti7(k_block, h, w, c, y, x, 0, output_H_const, output_W_const, C_const, Y_const, X_const, K_BLOCK);
                                    float *restrict inputs_pointer = INPUTS + ti5(n_block, c, h + y - padding_const, w + x - padding_const, 0, C_const, H_const, W_const, N_BLOCK);

                                    for (int cc = 0; cc < C_BLOCK; cc++){
                                        for (int kk = 0; kk < K_BLOCK; kk++){
                                            convolutions[kk*N_BLOCK : N_BLOCK] += inputs_pointer[0 : N_BLOCK] * (*filters_pointer);
                                            filters_pointer++;
                                        } //kk

                                        filters_pointer += Y_const*X_const*K_BLOCK - K_BLOCK;
                                        inputs_pointer += H_const*W_const*N_BLOCK;
                                    } // cc
                                } // if

                            } // x
                        } // y
                    } // w
                } // h
            } // c_block


            // ~=~=~=~=~=~=~=~= POOLING ~=~=~=~=~=~=~=~= 
            for (h = 0; h < pooled_H_const; h++){
                for (w = 0; w < pooled_W_const; w++){

                    int h_output = h*pooling_stride_const;
                    int w_output = w*pooling_stride_const;

                    int window_width = pooling_radius_const - mx(w_output + pooling_radius_const - output_W_const, 0);
                    int window_height = pooling_radius_const - mx(h_output + pooling_radius_const - output_H_const, 0);

                    for (int kk = 0; kk < K_BLOCK; kk++){
                        float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, kk, 0, output_H_const, output_W_const, K_BLOCK, N_BLOCK);
                        int *restrict argmaxs_pointer = ARGMAXS + ti5(n_block, k + kk, h, w, 0, K_const, pooled_H_const, pooled_W_const, N_BLOCK);
                        float *restrict pooled_outputs_pointer = OUTPUTS + ti5(n_block, k + kk, h, w, 0, K_const, pooled_H_const, pooled_W_const, N_BLOCK);
                        
                        pooled_outputs_pointer[0 : N_BLOCK] = -1.0e6;

                        int outputs_index = h_output*output_W_const + w_output;
                        for (y = 0; y < window_height; y++){
                            for (x = 0; x < window_width; x++){
                                if (outputs_pointer[0 : N_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
                                    pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK];
                                    argmaxs_pointer[0 : N_BLOCK] = outputs_index;
                                }
                                outputs_index++;
                                outputs_pointer += K_BLOCK*N_BLOCK;
                            }
                            outputs_index += output_W_const - window_width;
                            outputs_pointer += (output_W_const - window_width)*K_BLOCK*N_BLOCK;
                        }
                    }

                }
            }
        } //nk

    } // pragma_offload
}


void get_argmaxs(int N, int C, int H, int W, float *restrict INPUTS, float *restrict OUTPUTS, int *restrict ARGMAXS){

    #pragma offload target(mic:MIC_DEV) \ 
        in(INPUTS:length(0) REUSE) \
        in(OUTPUTS:length(0) REUSE) \
        in(ARGMAXS:length(0) REUSE)
    {
        int n_block, n, c, h, w;

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(n, c, h, w) \
            shared(N, INPUTS, OUTPUTS, ARGMAXS, C, H, W)

        for (int nc = 0; nc < N*C; nc++){
            n = nc / C;
            c = md(nc, C);
            
            for (h = 0; h < H; h++){ 
                for (w = 0; w < W; w++){
                    OUTPUTS[ti(c, h, w, 0, H, W, N) : N] = INPUTS[ARGMAXS[ti(c, h, w, 0, H, W, N): N] + c*H*W*N];
                } // w
            } // h
        } // nc
    } // pragma offload
}

// void permute_dimensions(int D1, int D2, int D3, int D4, int perm1, int perm2, int perm3, int perm4, float *restrict TENSOR, float *restrict SCRATCH){

//     #pragma offload target(mic:MIC_DEV) \ 
//         in(TENSOR:length(0) REUSE) \
//         in(SCRATCH:length(0) REUSE)
//     {   
//         int i1, i2, i3, i4;
//         SCRATCH[0 : D1*D2*D3*D4] = TENSOR[0 : D1*D2*D3*D4];

//         int dimensions[4] = {D1, D2, D3, D4};
//         int P1 = dimensions[perm1];
//         int P2 = dimensions[perm2];
//         int P3 = dimensions[perm3];
//         int P4 = dimensions[perm4];

//         // printf("(%d, %d, %d, %d)\n (%d, %d, %d, %d)\n", D1, D2, D3, D4, P1, P2, P3, P4);

//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(i1, i2, i3, i4) \
//             shared(TENSOR, SCRATCH, D1, D2, D3, D4, P1, P2, P3, P4, perm1, perm2, perm3, perm4)

//         for (int i1i2 = 0; i1i2 < D1*D2; i1i2++){
//             i1 = i1i2 / D2;
//             i2 = md(i1i2, D2);

//             int indices[4];
//             for (i3 = 0; i3 < D3; i3++){
//                 for (i4 = 0; i4 < D4; i4++){
//                     indices[0] = i1;
//                     indices[1] = i2;
//                     indices[2] = i3;
//                     indices[3] = i4;
//                     int p1 = indices[perm1];
//                     int p2 = indices[perm2];
//                     int p3 = indices[perm3];
//                     int p4 = indices[perm4];
//                     TENSOR[ti(p1, p2, p3, p4, P2, P3, P4)] = SCRATCH[ti(i1, i2, i3, i4, D2, D3, D4)];
//                 }
//             }
//         }

//     } // pragma offload
// }

// void permute_dimensions_int(int D1, int D2, int D3, int D4, int perm1, int perm2, int perm3, int perm4, int *restrict TENSOR, float *restrict SCRATCH){

//     #pragma offload target(mic:MIC_DEV) \ 
//         in(TENSOR:length(0) REUSE) \
//         in(SCRATCH:length(0) REUSE)
//     {   
//         int i1, i2, i3, i4;
//         SCRATCH[0 : D1*D2*D3*D4] = (float) TENSOR[0 : D1*D2*D3*D4];

//         int dimensions[4] = {D1, D2, D3, D4};
//         int P1 = dimensions[perm1];
//         int P2 = dimensions[perm2];
//         int P3 = dimensions[perm3];
//         int P4 = dimensions[perm4];

//         // printf("(%d, %d, %d, %d)\n (%d, %d, %d, %d)\n", D1, D2, D3, D4, P1, P2, P3, P4);

//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(i1, i2, i3, i4) \
//             shared(TENSOR, SCRATCH, D1, D2, D3, D4, P1, P2, P3, P4, perm1, perm2, perm3, perm4)

//         for (int i1i2 = 0; i1i2 < D1*D2; i1i2++){
//             i1 = i1i2 / D2;
//             i2 = md(i1i2, D2);

//             int indices[4];
//             for (i3 = 0; i3 < D3; i3++){
//                 for (i4 = 0; i4 < D4; i4++){
//                     indices[0] = i1;
//                     indices[1] = i2;
//                     indices[2] = i3;
//                     indices[3] = i4;
//                     int p1 = indices[perm1];
//                     int p2 = indices[perm2];
//                     int p3 = indices[perm3];
//                     int p4 = indices[perm4];
//                     TENSOR[ti(p1, p2, p3, p4, P2, P3, P4)] = (int) SCRATCH[ti(i1, i2, i3, i4, D2, D3, D4)];
//                 }
//             }
//         }

//     } // pragma offload
// }

void permute_dimensions(int D1, int D2, int D3, int D4, int perm1, int perm2, int perm3, int perm4, float *restrict TENSOR, float *restrict SCRATCH){

    #pragma offload target(mic:MIC_DEV) \ 
        in(TENSOR:length(0) REUSE) \
        in(SCRATCH:length(0) REUSE)
    {   
        int i1, i2, i3, i4;
    int D = D1*D2*D3*D4;
    // copy D elements from TENSOR to SCRATCH
    int num_cache_lines_64 = D/64;  
    int D_aligned = num_cache_lines_64*64;
    int D_remaining = D - D_aligned;

        #pragma omp parallel for schedule(static,16) default(none) shared(D, num_cache_lines_64, TENSOR, SCRATCH)
    for(int d = 0; d < num_cache_lines_64; d++)
    {
            float *restrict tensor_pointer = TENSOR + d*64;
            float *restrict scratch_pointer = SCRATCH + d*64;
        #if defined __MIC__
                __m512 tens_1 = _mm512_extload_ps(tensor_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_2 = _mm512_extload_ps(tensor_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_3 = _mm512_extload_ps(tensor_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_4 = _mm512_extload_ps(tensor_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        _mm_prefetch((char *)(tensor_pointer + 64     ), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 16), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 32), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 48), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 256), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 16), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 32), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 48), _MM_HINT_T1);
                _mm512_storenrngo_ps((float *)(scratch_pointer),      tens_1);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 16), tens_2);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 32), tens_3);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 48), tens_4);
        #endif
    }
    //copy remaining unaligned elements
    SCRATCH[D_aligned : D_remaining] = TENSOR[D_aligned : D_remaining];

        //SCRATCH[0 : D1*D2*D3*D4] = TENSOR[0 : D1*D2*D3*D4];

        int dimensions[4] = {D1, D2, D3, D4};
        int P1 = dimensions[perm1];
        int P2 = dimensions[perm2];
        int P3 = dimensions[perm3];
        int P4 = dimensions[perm4];
    __declspec(aligned(64)) int inverse_perm[4];
    inverse_perm[perm1] = 0;
    inverse_perm[perm2] = 1;
    inverse_perm[perm3] = 2;
    inverse_perm[perm4] = 3;

        //printf("(%d, %d, %d, %d) (%d, %d, %d, %d) \n", D1, D2, D3, D4, P1, P2, P3, P4); 

        #pragma omp parallel for \
            schedule(static,8) \
            default(none) \
            private(i1, i2, i3, i4) \
            shared(TENSOR, SCRATCH, D1, D2, D3, D4, P1, P2, P3, P4, perm1, perm2, perm3, perm4, inverse_perm)

        for (int i1i2 = 0; i1i2 < D1*D2; i1i2++){
            i1 = i1i2 / D2;
            i2 = md(i1i2, D2);
        float * SCRATCH_base = SCRATCH + i1*D2*D3*D4 + i2*D3*D4;

            __declspec(aligned(64)) int p[4];
        p[inverse_perm[0]] = i1;
        p[inverse_perm[1]] = i2;

            for (i3 = 0; i3 < D3; i3++){
            p[inverse_perm[2]] = i3;
            float * SCRATCH_ptr = SCRATCH_base + i3*D4;
                for (i4 = 0; i4 < D4; i4++){
                p[inverse_perm[3]] = i4;
                    TENSOR[ti(p[0], p[1], p[2], p[3], P2, P3, P4)] = SCRATCH_ptr[i4];
                }
            }
        }

    } // pragma offload
}

void permute_dimensions_int(int D1, int D2, int D3, int D4, int perm1, int perm2, int perm3, int perm4, int *restrict TENSOR, float *restrict SCRATCH){

    #pragma offload target(mic:MIC_DEV) \ 
        in(TENSOR:length(0) REUSE) \
        in(SCRATCH:length(0) REUSE)
    {   
        int i1, i2, i3, i4;

    int D = D1*D2*D3*D4;
    // copy D elements from TENSOR to SCRATCH
    int num_cache_lines_64 = D/64;  
    int D_aligned = num_cache_lines_64*64;
    int D_remaining = D - D_aligned;
    float * restrict TENSOR_FLOAT = (float *)(TENSOR);

        #pragma omp parallel for schedule(static,16) default(none) shared(D, num_cache_lines_64, TENSOR_FLOAT, SCRATCH)
    for(int d = 0; d < num_cache_lines_64; d++)
    {
            float *restrict tensor_pointer = (float *)(TENSOR_FLOAT) + d*64;
            float *restrict scratch_pointer = SCRATCH + d*64;
        #if defined __MIC__
                __m512 tens_1 = _mm512_extload_ps(tensor_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_2 = _mm512_extload_ps(tensor_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_3 = _mm512_extload_ps(tensor_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_4 = _mm512_extload_ps(tensor_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        _mm_prefetch((char *)(tensor_pointer + 64     ), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 16), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 32), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 48), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 256), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 16), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 32), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 48), _MM_HINT_T1);
                _mm512_storenrngo_ps((float *)(scratch_pointer),      tens_1);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 16), tens_2);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 32), tens_3);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 48), tens_4);
        #endif
    }
    //copy remaining unaligned elements
    SCRATCH[D_aligned : D_remaining] = TENSOR_FLOAT[D_aligned : D_remaining];

        int dimensions[4] = {D1, D2, D3, D4};
        int P1 = dimensions[perm1];
        int P2 = dimensions[perm2];
        int P3 = dimensions[perm3];
        int P4 = dimensions[perm4];
    __declspec(aligned(64)) int inverse_perm[4];
    inverse_perm[perm1] = 0;
    inverse_perm[perm2] = 1;
    inverse_perm[perm3] = 2;
    inverse_perm[perm4] = 3;

        //printf("(%d, %d, %d, %d) (%d, %d, %d, %d) \n", D1, D2, D3, D4, P1, P2, P3, P4); 

        #pragma omp parallel for \
            schedule(static,8) \
            default(none) \
            private(i1, i2, i3, i4) \
            shared(TENSOR_FLOAT, SCRATCH, D1, D2, D3, D4, P1, P2, P3, P4, perm1, perm2, perm3, perm4, inverse_perm)

        for (int i1i2 = 0; i1i2 < D1*D2; i1i2++){
            i1 = i1i2 / D2;
            i2 = md(i1i2, D2);
        float * SCRATCH_base = SCRATCH + i1*D2*D3*D4 + i2*D3*D4;

            __declspec(aligned(64)) int p[4];
        p[inverse_perm[0]] = i1;
        p[inverse_perm[1]] = i2;

            for (i3 = 0; i3 < D3; i3++){
            p[inverse_perm[2]] = i3;
            float * SCRATCH_ptr = SCRATCH_base + i3*D4;
                for (i4 = 0; i4 < D4; i4++){
                p[inverse_perm[3]] = i4;
                    TENSOR_FLOAT[ti(p[0], p[1], p[2], p[3], P2, P3, P4)] = SCRATCH_ptr[i4];
                }
            }
        }
    } // pragma offload
}

void transpose_replace(int N, int C, float *restrict TENSOR, float *restrict SCRATCH){

    #pragma offload target(mic:MIC_DEV) \ 
        in(TENSOR:length(0) REUSE) \
        in(SCRATCH:length(0) REUSE)
    {   
        int n, c;
    int D = N*C;
    int num_cache_lines_64 = D/64;  
    int D_aligned = num_cache_lines_64*64;
    int D_remaining = D - D_aligned;

        #pragma omp parallel for schedule(static,16) default(none) shared(D, num_cache_lines_64, TENSOR, SCRATCH)
    for(int d = 0; d < num_cache_lines_64; d++)
    {
            float *restrict tensor_pointer = (float *)(TENSOR) + d*64;
            float *restrict scratch_pointer = SCRATCH + d*64;
        #if defined __MIC__
                __m512 tens_1 = _mm512_extload_ps(tensor_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_2 = _mm512_extload_ps(tensor_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_3 = _mm512_extload_ps(tensor_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_4 = _mm512_extload_ps(tensor_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        _mm_prefetch((char *)(tensor_pointer + 64     ), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 16), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 32), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 48), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 256), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 16), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 32), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 48), _MM_HINT_T1);
                _mm512_storenrngo_ps((float *)(scratch_pointer),      tens_1);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 16), tens_2);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 32), tens_3);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 48), tens_4);
        #endif
    }
    //copy remaining unaligned elements
    SCRATCH[D_aligned : D_remaining] = TENSOR[D_aligned : D_remaining];

    int N_aligned = (N/64)*64;
        #pragma omp parallel for schedule(static,16) default(none) private(n, c) shared(N, TENSOR, SCRATCH, C)
        for (int nc = 0; nc < C * (N/64); nc++){
            int c = nc % C;
            int n = (nc / C)*64;
                TENSOR[c*N + n : 64] = SCRATCH[c + n*C : 64 : C];
        }
        #pragma omp parallel for schedule(static,16) default(none) private(n, c) shared(N, N_aligned, TENSOR, SCRATCH, C)
        for (int c = 0; c < C; c++){
            TENSOR[c*N + N_aligned : (N-N_aligned)] = SCRATCH[c + N_aligned*C: (N-N_aligned) : C];
        }
    } // pragma offload
}

void transpose_replace_int(int N, int C, int *restrict TENSOR, float *restrict SCRATCH){

    #pragma offload target(mic:MIC_DEV) \ 
        in(TENSOR:length(0) REUSE) \
        in(SCRATCH:length(0) REUSE)
    {   

        int n, c;
    int D = N*C;
    int num_cache_lines_64 = D/64;  
    int D_aligned = num_cache_lines_64*64;
    int D_remaining = D - D_aligned;
    float * restrict TENSOR_FLOAT = (float *)(TENSOR);

        #pragma omp parallel for schedule(static,16) default(none) shared(D, num_cache_lines_64, TENSOR_FLOAT, SCRATCH)
    for(int d = 0; d < num_cache_lines_64; d++)
    {
            float *restrict tensor_pointer = (float *)(TENSOR_FLOAT) + d*64;
            float *restrict scratch_pointer = SCRATCH + d*64;
        #if defined __MIC__
                __m512 tens_1 = _mm512_extload_ps(tensor_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_2 = _mm512_extload_ps(tensor_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_3 = _mm512_extload_ps(tensor_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_4 = _mm512_extload_ps(tensor_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        _mm_prefetch((char *)(tensor_pointer + 64     ), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 16), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 32), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 48), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 256), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 16), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 32), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 48), _MM_HINT_T1);
                _mm512_storenrngo_ps((float *)(scratch_pointer),      tens_1);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 16), tens_2);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 32), tens_3);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 48), tens_4);
        #endif
    }
    //copy remaining unaligned elements
    SCRATCH[D_aligned : D_remaining] = TENSOR_FLOAT[D_aligned : D_remaining];

    int N_aligned = (N/64)*64;
        #pragma omp parallel for schedule(static,16) default(none) private(n, c) shared(N, TENSOR_FLOAT, SCRATCH, C)
        for (int nc = 0; nc < C * (N/64); nc++){
            int c = nc % C;
            int n = nc / C;
                TENSOR_FLOAT[c*N + n*64 : 64] = SCRATCH[c + n*64*C : 64 : C];
        }
        #pragma omp parallel for schedule(static,16) default(none) private(n, c) shared(N, N_aligned, TENSOR_FLOAT, SCRATCH, C)
        for (int c = 0; c < C; c++){
            TENSOR_FLOAT[c*N + N_aligned : (N-N_aligned)] = SCRATCH[c + N_aligned*C: (N-N_aligned) : C];
        }
    } // pragma offload
}

// void transpose_replace(int N, int C, float *restrict TENSOR, float *restrict SCRATCH){

//     #pragma offload target(mic:MIC_DEV) \ 
//         in(TENSOR:length(0) REUSE) \
//         in(SCRATCH:length(0) REUSE)
//     {   
        
//         // mkl_somatcopy('R', 'T', N, C, 1.0, TENSOR, C, SCRATCH, N);
//         int n, c;
//         SCRATCH[0 : N*C] = TENSOR[0 : N*C];

//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(n, c) \
//             shared(N, TENSOR, SCRATCH, C)

//         for (int c = 0; c < C; c++){
//             TENSOR[c*N : N] = SCRATCH[c : N : C];
//         }            

//     } // pragma offload
// }

// void transpose_replace_int(int N, int C, int *restrict TENSOR, float *restrict SCRATCH){

//     #pragma offload target(mic:MIC_DEV) \ 
//         in(TENSOR:length(0) REUSE) \
//         in(SCRATCH:length(0) REUSE)
//     {   
        
//         // mkl_somatcopy('R', 'T', N, C, 1.0, TENSOR, C, SCRATCH, N);
//         int n, c;
//         SCRATCH[0 : N*C] = (float) TENSOR[0 : N*C];

//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(n, c) \
//             shared(N, TENSOR, SCRATCH, C)

//         for (int c = 0; c < C; c++){
//             TENSOR[c*N : N] = (int) SCRATCH[c : N : C];
//         }            

//     } // pragma offload
// }

// [C/C_BLOCK, Y/Y_BLOCK, K, Y_BLOCK, X, C_BLOCK] 

// void interleave_filters_for_gradient(const int K, const int C, const int Y, const int X, const int C_BLOCKSIZE, const int Y_BLOCKSIZE, float *restrict TENSOR, float *restrict SCRATCH){

//     #pragma offload target(mic:MIC_DEV) \ 
//         in(TENSOR:length(0) REUSE) \
//         in(SCRATCH:length(0) REUSE)
//     {
//         int c_block, cc, y_block, yy, k, c, y, x;
        
//         SCRATCH[0 : K*C*Y*X] = TENSOR[0 : K*C*Y*X];

//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(c_block, cc, y_block, yy, k, c, y, x) \
//             shared(K, C, Y, X, TENSOR, SCRATCH, C_BLOCKSIZE, Y_BLOCKSIZE)

//         for (int nc = 0; nc < N*C; nc++){
//             n = nc / C;
//             c = md(nc, C);
//             c_block = c/BLOCKSIZE;
//             cc = md(c, BLOCKSIZE);

            
//             for (h = 0; h < H; h++){
//                 for (w = 0; w < W; w++){
//                     TENSOR[ti5(n, c_block, h, w, cc, C/BLOCKSIZE, H, W, BLOCKSIZE)] = SCRATCH[ti(n, c, h, w, C, H, W)];
//                 }
//             }

//         } // nc
//     } // pragma offload
// }

void interleave_for_gradient(const int N, const int C, const int H, const int W, const int BLOCKSIZE, float *restrict TENSOR, float *restrict SCRATCH){

    #pragma offload target(mic:MIC_DEV) \ 
        in(TENSOR:length(0) REUSE) \
        in(SCRATCH:length(0) REUSE)
    {
        int c_block, cc, n, c, h, w;
        int C_BLOCKSIZE = C/BLOCKSIZE;
 
    int D = N*C*H*W;
    int num_cache_lines_64 = D/64;  
    int D_aligned = num_cache_lines_64*64;
    int D_remaining = D - D_aligned;

        #pragma omp parallel for schedule(static,16) default(none) shared(D, num_cache_lines_64, TENSOR, SCRATCH)
    for(int d = 0; d < num_cache_lines_64; d++)
    {
            float *restrict tensor_pointer = (float *)(TENSOR) + d*64;
            float *restrict scratch_pointer = SCRATCH + d*64;
        #if defined __MIC__
                __m512 tens_1 = _mm512_extload_ps(tensor_pointer , _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_2 = _mm512_extload_ps(tensor_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_3 = _mm512_extload_ps(tensor_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_4 = _mm512_extload_ps(tensor_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        _mm_prefetch((char *)(tensor_pointer + 64     ), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 16), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 32), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 48), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 256), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 16), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 32), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 48), _MM_HINT_T1);
                _mm512_storenrngo_ps((float *)(scratch_pointer),      tens_1);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 16), tens_2);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 32), tens_3);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 48), tens_4);
        #endif
    }
    //copy remaining unaligned elements
    SCRATCH[D_aligned : D_remaining] = TENSOR[D_aligned : D_remaining];


        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(c_block, n, c, cc, h, w) \
            shared(N, H, W, C, C_BLOCKSIZE, TENSOR, SCRATCH, BLOCKSIZE)

        for (int nc = 0; nc < N*C; nc++){
            n = nc / C;
            c = md(nc, C);
            c_block = c/BLOCKSIZE;
            cc = md(c, BLOCKSIZE);

            // TENSOR[ti5(n, c_block, 0, 0, cc, C_BLOCKSIZE, H, W, BLOCKSIZE) : H*W : BLOCKSIZE] = SCRATCH[ti(n, c, 0, 0, C, H, W) : H*W];
            float *restrict TENSOR_pointer = TENSOR + ti5(n, c_block, 0, 0, cc, C_BLOCKSIZE, H, W, BLOCKSIZE);
            float *restrict SCRATCH_pointer = SCRATCH + ti(n, c, 0, 0, C, H, W);
            for (h = 0; h < H; h++){
                for (w = 0; w < W; w++){
                    // TENSOR[ti5(n, c_block, h, w, cc, C_BLOCKSIZE, H, W, BLOCKSIZE)] = SCRATCH[ti(n, c, h, w, C, H, W)];
                    *TENSOR_pointer = *SCRATCH_pointer;
                    TENSOR_pointer += BLOCKSIZE;
                    SCRATCH_pointer++;
                }
            }

        } // nc
    } // pragma offload
}

void uninterleave_for_gradient(const int N, const int C, const int H, const int W, const int BLOCKSIZE, float *restrict TENSOR, float *restrict SCRATCH){

    #pragma offload target(mic:MIC_DEV) \ 
        in(TENSOR:length(0) REUSE) \
        in(SCRATCH:length(0) REUSE)
    {
        int c_block, cc, n, c, h, w;
        int C_BLOCKSIZE = C/BLOCKSIZE;

    int D = N*C*H*W;
    int num_cache_lines_64 = D/64;  
    int D_aligned = num_cache_lines_64*64;
    int D_remaining = D - D_aligned;

        #pragma omp parallel for schedule(static,16) default(none) shared(D, num_cache_lines_64, TENSOR, SCRATCH)
    for(int d = 0; d < num_cache_lines_64; d++)
    {
            float *restrict tensor_pointer = (float *)(TENSOR) + d*64;
            float *restrict scratch_pointer = SCRATCH + d*64;
        #if defined __MIC__
                __m512 tens_1 = _mm512_extload_ps(tensor_pointer , _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_2 = _mm512_extload_ps(tensor_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_3 = _mm512_extload_ps(tensor_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_4 = _mm512_extload_ps(tensor_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        _mm_prefetch((char *)(tensor_pointer + 64     ), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 16), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 32), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 48), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 256), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 16), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 32), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 48), _MM_HINT_T1);
                _mm512_storenrngo_ps((float *)(scratch_pointer),      tens_1);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 16), tens_2);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 32), tens_3);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 48), tens_4);
        #endif
    }
    //copy remaining unaligned elements
    SCRATCH[D_aligned : D_remaining] = TENSOR[D_aligned : D_remaining];

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(c_block, n, c, cc, h, w) \
            shared(N, H, W, C, C_BLOCKSIZE, TENSOR, SCRATCH, BLOCKSIZE)

        for (int nc = 0; nc < N*C; nc++){
            n = nc / C;
            c = md(nc, C);
            c_block = c/BLOCKSIZE;
            cc = md(c, BLOCKSIZE);

            // for (h = 0; h < H; h++){
            //     for (w = 0; w < W; w++){
            //         TENSOR[ti(n, c, h, w, C, H, W)] = SCRATCH[ti5(n, c_block, h, w, cc, C_BLOCKSIZE, H, W, BLOCKSIZE)];
            //     }
            // }

            float *restrict TENSOR_pointer = TENSOR + ti(n, c, 0, 0, C, H, W);
            float *restrict SCRATCH_pointer = SCRATCH + ti5(n, c_block, 0, 0, cc, C_BLOCKSIZE, H, W, BLOCKSIZE);
            for (h = 0; h < H; h++){
                for (w = 0; w < W; w++){
                    // TENSOR[ti(n, c, h, w, C, H, W)] = SCRATCH[ti5(n, c_block, h, w, cc, C_BLOCKSIZE, H, W, BLOCKSIZE)];
                    *TENSOR_pointer = *SCRATCH_pointer;
                    SCRATCH_pointer += BLOCKSIZE;
                    TENSOR_pointer++;
                }
            }

        } // nc
    } // pragma offload
}

// void interleave_for_gradient(const int N, const int C, const int H, const int W, const int BLOCKSIZE, float *restrict TENSOR, float *restrict SCRATCH){

//     #pragma offload target(mic:MIC_DEV) \ 
//         in(TENSOR:length(0) REUSE) \
//         in(SCRATCH:length(0) REUSE)
//     {
//         int c_block, cc, n, c, h, w;
//         int C_BLOCKSIZE = C/BLOCKSIZE;
 
//         SCRATCH[0 : N*C*H*W] = TENSOR[0 : N*C*H*W];

//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(c_block, n, c, cc, h, w) \
//             shared(N, H, W, C, C_BLOCKSIZE, TENSOR, SCRATCH, BLOCKSIZE)

//         for (int nc = 0; nc < N*C; nc++){
//             n = nc / C;
//             c = md(nc, C);
//             c_block = c/BLOCKSIZE;
//             cc = md(c, BLOCKSIZE);

            
//             for (h = 0; h < H; h++){
//                 for (w = 0; w < W; w++){
//                     TENSOR[ti5(n, c_block, h, w, cc, C_BLOCKSIZE, H, W, BLOCKSIZE)] = SCRATCH[ti(n, c, h, w, C, H, W)];
//                 }
//             }

//         } // nc
//     } // pragma offload
// }

// void uninterleave_for_gradient(const int N, const int C, const int H, const int W, const int BLOCKSIZE, float *restrict TENSOR, float *restrict SCRATCH){

//     #pragma offload target(mic:MIC_DEV) \ 
//         in(TENSOR:length(0) REUSE) \
//         in(SCRATCH:length(0) REUSE)
//     {
//         int c_block, cc, n, c, h, w;
//         int C_BLOCKSIZE = C/BLOCKSIZE;
        
//         SCRATCH[0 : N*C*H*W] = TENSOR[0 : N*C*H*W];

//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(c_block, n, c, cc, h, w) \
//             shared(N, H, W, C, C_BLOCKSIZE, TENSOR, SCRATCH, BLOCKSIZE)

//         for (int nc = 0; nc < N*C; nc++){
//             n = nc / C;
//             c = md(nc, C);
//             c_block = c/BLOCKSIZE;
//             cc = md(c, BLOCKSIZE);

            
//             for (h = 0; h < H; h++){
//                 for (w = 0; w < W; w++){
//                     TENSOR[ti(n, c, h, w, C, H, W)] = SCRATCH[ti5(n, c_block, h, w, cc, C_BLOCKSIZE, H, W, BLOCKSIZE)];
//                 }
//             }

//         } // nc
//     } // pragma offload
// }

void interleave_block(const int N, const int C, const int BLOCKSIZE, float *restrict TENSOR, float *restrict SCRATCH){

    #pragma offload target(mic:MIC_DEV) \ 
        in(TENSOR:length(0) REUSE) \
        in(SCRATCH:length(0) REUSE)
    {
    assert((N % BLOCKSIZE) == 0);   
    int D = N*C;
    int num_cache_lines_64 = D/64;  
    int D_aligned = num_cache_lines_64*64;
    int D_remaining = D - D_aligned;

        #pragma omp parallel for schedule(static,16) default(none) shared(D, num_cache_lines_64, TENSOR, SCRATCH)
    for(int d = 0; d < num_cache_lines_64; d++)
    {
            float *restrict tensor_pointer = (float *)(TENSOR) + d*64;
            float *restrict scratch_pointer = SCRATCH + d*64;
        #if defined __MIC__
                __m512 tens_1 = _mm512_extload_ps(tensor_pointer , _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_2 = _mm512_extload_ps(tensor_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_3 = _mm512_extload_ps(tensor_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_4 = _mm512_extload_ps(tensor_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        _mm_prefetch((char *)(tensor_pointer + 64     ), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 16), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 32), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 48), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 256), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 16), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 32), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 48), _MM_HINT_T1);
                _mm512_storenrngo_ps((float *)(scratch_pointer),      tens_1);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 16), tens_2);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 32), tens_3);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 48), tens_4);
        #endif
    }
    //copy remaining unaligned elements
    SCRATCH[D_aligned : D_remaining] = TENSOR[D_aligned : D_remaining];

    #pragma omp parallel for \
            schedule(static,16) \
            default(none) \
            shared(N, TENSOR, SCRATCH, C, BLOCKSIZE)
    for(int c = 0; c < C; c++){
        float *restrict scratch_pointer = SCRATCH + c*N;
#pragma noprefetch
#pragma unroll (4)
        for(int n_block = 0; n_block < N/BLOCKSIZE; n_block++)
        {
                float *restrict tensor_pointer = TENSOR + ti3(n_block, c, 0, C, BLOCKSIZE);
#if BLOCKSIZE==16 && defined __MIC__
                __m512 scratch_1 = _mm512_extload_ps(scratch_pointer + n_block*BLOCKSIZE, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        _mm_prefetch((char *)(scratch_pointer + n_block*BLOCKSIZE + N), _MM_HINT_T1);
        _mm_prefetch((char *)(scratch_pointer + (n_block+2)*BLOCKSIZE), _MM_HINT_T0);
                _mm512_storenrngo_ps((float *)(tensor_pointer),  scratch_1);
#else
                tensor_pointer[0 : BLOCKSIZE] = scratch_pointer[n_block*BLOCKSIZE : BLOCKSIZE];
#endif
        }
    }

    } // pragma offload
}

void interleave_block_int(const int N, const int C, const int BLOCKSIZE, int *restrict TENSOR, float *restrict SCRATCH){

    #pragma offload target(mic:MIC_DEV) \ 
        in(TENSOR:length(0) REUSE) \
        in(SCRATCH:length(0) REUSE)
    {
        int n_block, n, c;
    assert((N % BLOCKSIZE) == 0);   
    int D = N*C;
    int num_cache_lines_64 = D/64;  
    int D_aligned = num_cache_lines_64*64;
    int D_remaining = D - D_aligned;

        #pragma omp parallel for schedule(static,16) default(none) shared(D, num_cache_lines_64, TENSOR, SCRATCH)
    for(int d = 0; d < num_cache_lines_64; d++)
    {
            float *restrict tensor_pointer = (float *)(TENSOR) + d*64;
            float *restrict scratch_pointer = SCRATCH + d*64;
        #if defined __MIC__
                __m512 tens_1 = _mm512_extload_ps(tensor_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_2 = _mm512_extload_ps(tensor_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_3 = _mm512_extload_ps(tensor_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_4 = _mm512_extload_ps(tensor_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        _mm_prefetch((char *)(tensor_pointer + 64     ), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 16), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 32), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 48), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 256), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 16), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 32), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 48), _MM_HINT_T1);
                _mm512_storenrngo_ps((float *)(scratch_pointer),      tens_1);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 16), tens_2);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 32), tens_3);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 48), tens_4);
        #endif
    }
    //copy remaining unaligned elements
    SCRATCH[D_aligned : D_remaining] = TENSOR[D_aligned : D_remaining];

        #pragma omp parallel for \
            schedule(static,16) \
            default(none) \
            shared(N, TENSOR, SCRATCH, C, BLOCKSIZE)
    for(int c = 0; c < C; c++){
        float *restrict scratch_pointer = SCRATCH + c*N;
#pragma noprefetch
#pragma unroll (4)
        for(int n_block = 0; n_block < N/BLOCKSIZE; n_block++)
        {
                float *restrict tensor_pointer = (float *)(TENSOR) + ti3(n_block, c, 0, C, BLOCKSIZE);
#if BLOCKSIZE==16 && defined __MIC__
                __m512 scratch_1 = _mm512_extload_ps(scratch_pointer + n_block*BLOCKSIZE, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        _mm_prefetch((char *)(scratch_pointer + n_block*BLOCKSIZE + N), _MM_HINT_T1);
        _mm_prefetch((char *)(scratch_pointer + (n_block+2)*BLOCKSIZE), _MM_HINT_T0);
                _mm512_storenrngo_ps((float *)(tensor_pointer),  scratch_1);
                
#else
                tensor_pointer[0 : BLOCKSIZE] = scratch_pointer[n_block*BLOCKSIZE : BLOCKSIZE];
#endif
        }
    }
    } // pragma offload
}

void uninterleave_block(const int N, const int C, const int BLOCKSIZE, float *restrict TENSOR, float *restrict SCRATCH){

    #pragma offload target(mic:MIC_DEV) \ 
        in(TENSOR:length(0) REUSE) \
        in(SCRATCH:length(0) REUSE)
    {
        int n_block, n, c;
    assert((N % BLOCKSIZE) == 0);   
    int D = N*C;
    int num_cache_lines_64 = D/64;  
    int D_aligned = num_cache_lines_64*64;
    int D_remaining = D - D_aligned;

        #pragma omp parallel for schedule(static,16) default(none) shared(D, num_cache_lines_64, TENSOR, SCRATCH)
    for(int d = 0; d < num_cache_lines_64; d++)
    {
            float *restrict tensor_pointer = (float *)(TENSOR) + d*64;
            float *restrict scratch_pointer = SCRATCH + d*64;
        #if defined __MIC__
                __m512 tens_1 = _mm512_extload_ps(tensor_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_2 = _mm512_extload_ps(tensor_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_3 = _mm512_extload_ps(tensor_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_4 = _mm512_extload_ps(tensor_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        _mm_prefetch((char *)(tensor_pointer + 64     ), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 16), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 32), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 48), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 256), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 16), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 32), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 48), _MM_HINT_T1);
                _mm512_storenrngo_ps((float *)(scratch_pointer),      tens_1);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 16), tens_2);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 32), tens_3);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 48), tens_4);
        #endif
    }
    //copy remaining unaligned elements
    SCRATCH[D_aligned : D_remaining] = TENSOR[D_aligned : D_remaining];
        
    #pragma omp parallel for \
            schedule(static,16) \
            default(none) \
            shared(N, TENSOR, SCRATCH, C, BLOCKSIZE)
    for(int c = 0; c < C; c++){
        float *restrict tensor_pointer = TENSOR + c*N;
#pragma noprefetch
#pragma unroll (4)
        for(int n_block = 0; n_block < N/BLOCKSIZE; n_block++)
        {
                float *restrict scratch_pointer = SCRATCH + ti3(n_block, c, 0, C, BLOCKSIZE);
#if BLOCKSIZE==16 && defined __MIC__
                __m512 scratch_1 = _mm512_extload_ps(scratch_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        _mm_prefetch((char *)(scratch_pointer + 6*C*BLOCKSIZE), _MM_HINT_T1);
        _mm_prefetch((char *)(scratch_pointer + 2*C*BLOCKSIZE), _MM_HINT_T0);
                _mm512_storenrngo_ps((float *)(tensor_pointer + n_block*BLOCKSIZE),  scratch_1);
#else
                tensor_pointer[n_block*BLOCKSIZE : BLOCKSIZE] = scratch_pointer[0 : BLOCKSIZE];
#endif
        }
    }



#if 0 
        SCRATCH[0 : N*C] = TENSOR[0 : N*C];

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(n_block, n, c) \
            shared(N, TENSOR, SCRATCH, C, BLOCKSIZE)

        for (int nc = 0; nc < N/BLOCKSIZE*C; nc++){
            n_block = nc / C;
            n = n_block*BLOCKSIZE;
            c = md(nc, C);

            TENSOR[ti2(c, n, N) : BLOCKSIZE] = SCRATCH[ti3(n_block, c, 0, C, BLOCKSIZE) : BLOCKSIZE];
        } // nc
#endif
    } // pragma offload
}

void uninterleave_block_int(const int N, const int C, const int BLOCKSIZE, int *restrict TENSOR, float *restrict SCRATCH){

    #pragma offload target(mic:MIC_DEV) \ 
        in(TENSOR:length(0) REUSE) \
        in(SCRATCH:length(0) REUSE)
    {
        int n_block, n, c;
    assert((N % BLOCKSIZE) == 0);   
    int D = N*C;
    int num_cache_lines_64 = D/64;  
    int D_aligned = num_cache_lines_64*64;
    int D_remaining = D - D_aligned;

        #pragma omp parallel for schedule(static,16) default(none) shared(D, num_cache_lines_64, TENSOR, SCRATCH)
    for(int d = 0; d < num_cache_lines_64; d++)
    {
            float *restrict tensor_pointer = (float *)(TENSOR) + d*64;
            float *restrict scratch_pointer = SCRATCH + d*64;
        #if defined __MIC__
                __m512 tens_1 = _mm512_extload_ps(tensor_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_2 = _mm512_extload_ps(tensor_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_3 = _mm512_extload_ps(tensor_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
                __m512 tens_4 = _mm512_extload_ps(tensor_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        _mm_prefetch((char *)(tensor_pointer + 64     ), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 16), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 32), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 64 + 48), _MM_HINT_T0);
        _mm_prefetch((char *)(tensor_pointer + 256), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 16), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 32), _MM_HINT_T1);
        _mm_prefetch((char *)(tensor_pointer + 256 + 48), _MM_HINT_T1);
                _mm512_storenrngo_ps((float *)(scratch_pointer),      tens_1);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 16), tens_2);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 32), tens_3);
                _mm512_storenrngo_ps((float *)(scratch_pointer + 48), tens_4);
        #endif
    }
    //copy remaining unaligned elements
    SCRATCH[D_aligned : D_remaining] = TENSOR[D_aligned : D_remaining];
        
    #pragma omp parallel for \
            schedule(static,16) \
            default(none) \
            shared(N, TENSOR, SCRATCH, C, BLOCKSIZE)
    for(int c = 0; c < C; c++){
        float *restrict tensor_pointer = (float *)(TENSOR) + c*N;
#pragma noprefetch
#pragma unroll (4)
        for(int n_block = 0; n_block < N/BLOCKSIZE; n_block++)
        {
                float *restrict scratch_pointer = SCRATCH + ti3(n_block, c, 0, C, BLOCKSIZE);
#if BLOCKSIZE==16 && defined __MIC__
                __m512 scratch_1 = _mm512_extload_ps(scratch_pointer, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); 
        _mm_prefetch((char *)(scratch_pointer + 6*C*BLOCKSIZE), _MM_HINT_T1);
        _mm_prefetch((char *)(scratch_pointer + 2*C*BLOCKSIZE), _MM_HINT_T0);
                _mm512_storenrngo_ps((float *)(tensor_pointer + n_block*BLOCKSIZE),  scratch_1);
#else
                tensor_pointer[n_block*BLOCKSIZE : BLOCKSIZE] = scratch_pointer[0 : BLOCKSIZE];
#endif
        }
    }
        
    } // pragma offload
}

// void interleave_block(const int N, const int C, const int BLOCKSIZE, float *restrict TENSOR, float *restrict SCRATCH){

//     #pragma offload target(mic:MIC_DEV) \ 
//         in(TENSOR:length(0) REUSE) \
//         in(SCRATCH:length(0) REUSE)
//     {
//         int n_block, n, c;
        
//         SCRATCH[0 : N*C] = TENSOR[0 : N*C];

//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(n_block, n, c) \
//             shared(N, TENSOR, SCRATCH, C, BLOCKSIZE)

//         for (int c = 0; c < C; c++){
//             // float *restrict tensor_pointer = TENSOR + ti3(n_block, c, 0, C, BLOCKSIZE);
//             // float *restrict scratch_pointer = SCRATCH + ti2(c, n, N);

//             for (n_block = 0; n_block < N/BLOCKSIZE; n_block++){
//                 TENSOR[(n_block*C + c)*BLOCKSIZE : BLOCKSIZE] = SCRATCH[c*N + n_block*BLOCKSIZE : BLOCKSIZE];    
//             }
            
//         } // nc
//     } // pragma offload
// }

// void interleave_block_int(const int N, const int C, const int BLOCKSIZE, int *restrict TENSOR, float *restrict SCRATCH){

//     #pragma offload target(mic:MIC_DEV) \ 
//         in(TENSOR:length(0) REUSE) \
//         in(SCRATCH:length(0) REUSE)
//     {
//         int n_block, n, c;
        
//         SCRATCH[0 : N*C] = (float) TENSOR[0 : N*C];

//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(n_block, n, c) \
//             shared(N, TENSOR, SCRATCH, C, BLOCKSIZE)

//         for (int nc = 0; nc < N/BLOCKSIZE*C; nc++){
//             n_block = nc / C;
//             n = n_block*BLOCKSIZE;
//             c = md(nc, C);
            
//             int *restrict tensor_pointer = TENSOR + ti3(n_block, c, 0, C, BLOCKSIZE);
//             float *restrict scratch_pointer = SCRATCH + ti2(c, n, N);

//             tensor_pointer[0 : BLOCKSIZE] = (int) scratch_pointer[0 : BLOCKSIZE];
//         } // nc
//     } // pragma offload
// }

// void uninterleave_block(const int N, const int C, const int BLOCKSIZE, float *restrict TENSOR, float *restrict SCRATCH){

//     #pragma offload target(mic:MIC_DEV) \ 
//         in(TENSOR:length(0) REUSE) \
//         in(SCRATCH:length(0) REUSE)
//     {
//         int n_block, n, c;
        
//         SCRATCH[0 : N*C] = TENSOR[0 : N*C];

//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(n_block, n, c) \
//             shared(N, TENSOR, SCRATCH, C, BLOCKSIZE)

//         for (int nc = 0; nc < N/BLOCKSIZE*C; nc++){
//             n_block = nc / C;
//             n = n_block*BLOCKSIZE;
//             c = md(nc, C);

//             TENSOR[ti2(c, n, N) : BLOCKSIZE] = SCRATCH[ti3(n_block, c, 0, C, BLOCKSIZE) : BLOCKSIZE];
//         } // nc
//     } // pragma offload
// }

// void uninterleave_block_int(const int N, const int C, const int BLOCKSIZE, int *restrict TENSOR, float *restrict SCRATCH){

//     #pragma offload target(mic:MIC_DEV) \ 
//         in(TENSOR:length(0) REUSE) \
//         in(SCRATCH:length(0) REUSE)
//     {
//         int n_block, n, c;
        
//         SCRATCH[0 : N*C] = (float) TENSOR[0 : N*C];

//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(n_block, n, c) \
//             shared(N, TENSOR, SCRATCH, C, BLOCKSIZE)

//         for (int nc = 0; nc < N/BLOCKSIZE*C; nc++){
//             n_block = nc / C;
//             n = n_block*BLOCKSIZE;
//             c = md(nc, C);
            
//             TENSOR[ti2(c, n, N) : BLOCKSIZE] = (int) SCRATCH[ti3(n_block, c, 0, C, BLOCKSIZE) : BLOCKSIZE];
//         } // nc
//     } // pragma offload
// }




// Convolution before changing data structure to [N, C, H, W, BLOCK]
// int *convolution_layer1(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){

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
//         int n_block, n, k, i, j, h, w, c, y, x;
//         int nk, hw, ij, nkhw;

//         // computation of constants
//         int XWN = (-X_const + W_const)*N,
//             HYWN = (H_const-Y_const)*W_const*N;        

//         // SCRATCH[0:K_const*N*output_H_const*output_W_const] = 0.f;
//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(nk, hw, ij, n_block, n, k, h, w, c, y, x, i, j) \
//             shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XWN, HYWN)
        
//         // ~=~=~=~=~=~=~=~= CONVOLUTION ~=~=~=~=~=~=~=~= 
//         for (nk = 0; nk < N/BLOCK*K_const; nk++){
            
//             n_block = nk / K_const;
//             n = n_block*BLOCK;
//             k = md(nk, K_const);

//             SCRATCH[omp_get_thread_num()*output_H_const*output_W_const*BLOCK : output_H_const*output_W_const*BLOCK] = 0.f;

//             for (c = 0; c < C_const; c++){
//                 for (h = 0; h < output_H_const; h++){
//                     for (w = 0; w < output_W_const; w++){

//                         // float *restrict convolutions = SCRATCH + ti(k, h, w, n, output_H_const, output_W_const, N);
//                         float *restrict convolutions = SCRATCH + ti(omp_get_thread_num(), h, w, 0, output_H_const, output_W_const, BLOCK);
//                         float *restrict convolutions_next = SCRATCH + ti(omp_get_thread_num(), h, w+1, 0, output_H_const, output_W_const, BLOCK);
//                 __assume_aligned(convolutions, 64);
//             _mm_prefetch((char *)(convolutions_next), _MM_HINT_ET0);
//             _mm_prefetch((char *)(convolutions_next + 16), _MM_HINT_ET0);
//             _mm_prefetch((char *)(convolutions_next + 32), _MM_HINT_ET0);
//             _mm_prefetch((char *)(convolutions_next + 48), _MM_HINT_ET0);

//                         int kcyx_shift = (k*C_const + c)*Y_const*X_const - 1; // filters
                                
//                         float *restrict filters_pointer = FILTERS + kcyx_shift;
                       
//                         // if we're not on boundary (i.e not affected by padding)
//                         if (w - padding_const >= 0 &&
//                             h - padding_const >= 0 &&
//                             output_W_const - 1 - w >= padding_const  &&
//                             output_H_const - 1 - h >= padding_const){

// #if 1 && defined __MIC__
//             __m512 res_1 = _mm512_extload_ps(convolutions, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
//             __m512 res_2 = _mm512_extload_ps(convolutions + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
//             __m512 res_3 = _mm512_extload_ps(convolutions + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
//             __m512 res_4 = _mm512_extload_ps(convolutions + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
// #endif
//                             float *restrict inputs_pointer = INPUTS + ti(c, h - padding_const, w - padding_const, n, H_const, W_const, N);
                              
//                             for (y = 0; y < Y_const; ++y){                                
//                 _mm_prefetch((char *)(inputs_pointer + N), _MM_HINT_T0);
//                 _mm_prefetch((char *)(inputs_pointer + N + 16), _MM_HINT_T0);
//                 _mm_prefetch((char *)(inputs_pointer + N + 32), _MM_HINT_T0);
//                 _mm_prefetch((char *)(inputs_pointer + N + 48), _MM_HINT_T0);
//                                 for (x = 0; x < X_const; ++x)
//                 {
//                         __assume_aligned(inputs_pointer, 64);
//                     filters_pointer++;
//                     _mm_prefetch((char *)(inputs_pointer + N), _MM_HINT_T0);
//                     _mm_prefetch((char *)(inputs_pointer + N + 16), _MM_HINT_T0);
//                     _mm_prefetch((char *)(inputs_pointer + N + 32), _MM_HINT_T0);
//                     _mm_prefetch((char *)(inputs_pointer + N + 48), _MM_HINT_T0);
// #if 1 && defined __MIC__
//                     __m512 v_filters = _mm512_set1_ps(*filters_pointer);
//                     {
//                     res_1 = _mm512_fmadd_ps(_mm512_extload_ps(inputs_pointer +  0, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE), v_filters, res_1); 
//                     res_2 = _mm512_fmadd_ps(_mm512_extload_ps(inputs_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE), v_filters, res_2); 
//                     res_3 = _mm512_fmadd_ps(_mm512_extload_ps(inputs_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE), v_filters, res_3); 
//                     res_4 = _mm512_fmadd_ps(_mm512_extload_ps(inputs_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE), v_filters, res_4); 
//                     }
// #else
//                                     convolutions[0 : BLOCK] += inputs_pointer[0 : BLOCK] * (*(filters_pointer));
// #endif
//                                     inputs_pointer += N;
//                                 } // x

//                                 inputs_pointer += XWN;
//                             } // y
// #if 1 && defined __MIC__
//             _mm512_extstore_ps((float *)(convolutions), res_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
//             _mm512_extstore_ps((float *)(convolutions + 16), res_2, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
//             _mm512_extstore_ps((float *)(convolutions + 32), res_3, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
//             _mm512_extstore_ps((float *)(convolutions + 48), res_4, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
// #endif

//                         }

//                         else{
// #if 1 && defined __MIC__
//             __m512 res_1 = _mm512_extload_ps(convolutions, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
//             __m512 res_2 = _mm512_extload_ps(convolutions + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
//             __m512 res_3 = _mm512_extload_ps(convolutions + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
//             __m512 res_4 = _mm512_extload_ps(convolutions + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
// #endif
//                             float *restrict inputs_pointer = INPUTS + ti(c, mx(mn(h-padding_const, H_const-1), 0), mx(mn(w-padding_const, W_const-1), 0), n, H_const, W_const, N);
                            
//             int min_x = mx(0, (padding_const - w)); 
//             int max_x = mn(X_const, (W_const + padding_const - w)); 
//             int min_y = mx(0, (padding_const - h)); 
//             int max_y = mn(Y_const, (H_const + padding_const - h)); 
// #if 1
//                 filters_pointer += min_y*X_const;
//                             for (y = min_y; y < max_y; ++y){
//                                 float *restrict inputs_pointer_y = inputs_pointer; // start-of-line pointer
//                 filters_pointer += min_x;
// #pragma unroll (X_const-padding_const)
// #pragma noprefetch
//                                 for (x = min_x; x < max_x; ++x)
//                 {
//                             __assume_aligned(inputs_pointer, 64);
//                         filters_pointer++;
//                             _mm_prefetch((char *)(inputs_pointer + N), _MM_HINT_T0);
//                             _mm_prefetch((char *)(inputs_pointer + N + 16), _MM_HINT_T0);
//                             _mm_prefetch((char *)(inputs_pointer + N + 32), _MM_HINT_T0);
//                             _mm_prefetch((char *)(inputs_pointer + N + 48), _MM_HINT_T0);
// #if 1 && defined __MIC__
//                             __m512 v_filters = _mm512_set1_ps(*filters_pointer);
//                             {
//                         res_1 = _mm512_fmadd_ps(_mm512_extload_ps(inputs_pointer +  0, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE), v_filters, res_1); 
//                         res_2 = _mm512_fmadd_ps(_mm512_extload_ps(inputs_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE), v_filters, res_2); 
//                         res_3 = _mm512_fmadd_ps(_mm512_extload_ps(inputs_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE), v_filters, res_3); 
//                         res_4 = _mm512_fmadd_ps(_mm512_extload_ps(inputs_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE), v_filters, res_4); 
//                            }
// #else
//                                            convolutions[0 : BLOCK] += inputs_pointer[0 : BLOCK] * (*(filters_pointer));
// #endif
//                                            inputs_pointer += N;
//                 } // x
//                 filters_pointer += (X_const - max_x);
//                                 inputs_pointer = inputs_pointer_y + W_const*N; 
//                             } 
// #if 1 && defined __MIC__
//             _mm512_extstore_ps((float *)(convolutions), res_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
//             _mm512_extstore_ps((float *)(convolutions + 16), res_2, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
//             _mm512_extstore_ps((float *)(convolutions + 32), res_3, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
//             _mm512_extstore_ps((float *)(convolutions + 48), res_4, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
// #endif
//                 filters_pointer += (Y_const - max_y)*X_const;
// #else
//                             for (y = 0; y < Y_const; ++y){
                                
//                                 float *restrict inputs_pointer_y = inputs_pointer; // start-of-line pointer
                                
//                                 if ((y + h - padding_const >= 0) && (y + h - padding_const < H_const)){ // i.e, are there any elements in this row that overlap with the image?
//                                     for (x = 0; x < X_const; ++x){
//                                         filters_pointer++;
                                        
//                                         if ((x + w - padding_const >= 0) && (x + w - padding_const < W_const)){
//                                             convolutions[0 : BLOCK] += inputs_pointer[0 : BLOCK] * (*filters_pointer);
//                                             inputs_pointer += N;
//                                         }
//                                     } // x

//                                     inputs_pointer = inputs_pointer_y + W_const*N; 
//                                 }

//                                 else filters_pointer += X_const;
//                             } // y
// #endif 

//                         } // if-else
//                     } // w
//                 } // h
//             } // c


//             // ~=~=~=~=~=~=~=~= POOLING ~=~=~=~=~=~=~=~= 
//             for (h = 0; h < pooled_H_const; h++){
//                 for (w = 0; w < pooled_W_const; w++){

//                     int h_output = h*pooling_stride_const;
//                     int w_output = w*pooling_stride_const;

//                     // float *restrict outputs_pointer = SCRATCH + ti(k, h_output, w_output, n, output_H_const, output_W_const, N);
//                     float *restrict outputs_pointer = SCRATCH + ti(omp_get_thread_num(), h_output, w_output, 0, output_H_const, output_W_const, BLOCK);
                    
//                     // int *restrict argmaxs_pointer = ARGMAXS + ti(k, h_output, w_output, n, output_H_const, output_W_const, N);
//                     int *restrict argmaxs_pointer = ARGMAXS + ti(k, h, w, n, pooled_H_const, pooled_W_const, N);
//                     float *restrict pooled_outputs_pointer = OUTPUTS + ti(k, h, w, n, pooled_H_const, pooled_W_const, N);
//                     pooled_outputs_pointer[0 : BLOCK] = -1.0e6;
                    
//                     // float *restrict argmaxs = SCRATCH + K_const*output_H_const*output_W_const*N + ti(k, h, w, n, pooled_H_const, pooled_W_const, N);

//                     int outputs_index = h_output*output_W_const + w_output;
//                     for (y = 0; y < pooling_radius_const; y++){
//                         for (x = 0; x < pooling_radius_const; x++){
//                             if (outputs_pointer[0 : BLOCK] > pooled_outputs_pointer[0 : BLOCK]){
//                                 pooled_outputs_pointer[0 : BLOCK] = outputs_pointer[0 : BLOCK];
//                                 argmaxs_pointer[0 : BLOCK] = outputs_index;
//                             }
//                             outputs_index++;
//                             outputs_pointer += BLOCK;
//                         }
//                         outputs_index += output_W_const - pooling_radius_const;
//                         outputs_pointer += (output_W_const - pooling_radius_const)*BLOCK;
//                     }

//                 }
//             }
//         } //nk

//     } // pragma_offload
// }

// Convolution after changing to [N_BLOCK, C, H, W, BLOCK] data structure, before the intrinsics
// int *convolution_layer1(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){

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
//         int n_block, n, k, i, j, h, w, c, y, x;
//         int nk, hw, ij, nkhw;

//         // computation of constants
//         int XWN = (-X_const + W_const)*N,
//             HYWN = (H_const-Y_const)*W_const*N;        

//         // SCRATCH[0:K_const*N*output_H_const*output_W_const] = 0.f;

//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(nk, hw, ij, n_block, n, k, h, w, c, y, x, i, j) \
//             shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XWN, HYWN)

//         // #pragma vector aligned
        
//         // ~=~=~=~=~=~=~=~= CONVOLUTION ~=~=~=~=~=~=~=~= 
//         for (nk = 0; nk < N/N_BLOCK*K_const; nk++){
            
//             n_block = nk / K_const;
//             n = n_block*N_BLOCK;
//             k = md(nk, K_const);

//             SCRATCH[omp_get_thread_num()*output_H_const*output_W_const*N_BLOCK : output_H_const*output_W_const*N_BLOCK] = 0.f;
//             // float *restrict convolutions = SCRATCH + ti(k, 0, 0, n, output_H_const, output_W_const, N);
//             // for (hw = 0; hw < output_H_const*output_W_const; hw++) convolutions[hw*N : N_BLOCK] = 0.f;

//             for (c = 0; c < C_const; c++){
//                 for (h = 0; h < output_H_const; h++){
//                     for (w = 0; w < output_W_const; w++){

//                         // float *restrict convolutions = SCRATCH + ti(k, h, w, n, output_H_const, output_W_const, N);
//                         float *restrict convolutions = SCRATCH + ti(omp_get_thread_num(), h, w, 0, output_H_const, output_W_const, N_BLOCK);

//                         int kcyx_shift = (k*C_const + c)*Y_const*X_const - 1; // filters
                                
//                         float *restrict filters_pointer = FILTERS + kcyx_shift;
                       
//                         // if we're not on boundary (i.e not affected by padding)
//                         if (w - padding_const >= 0 &&
//                             h - padding_const >= 0 &&
//                             output_W_const - 1 - w >= padding_const  &&
//                             output_H_const - 1 - h >= padding_const){

//                             float *restrict inputs_pointer = INPUTS + ti5(n_block, c, h - padding_const, w - padding_const, 0, C_const, H_const, W_const, N_BLOCK);
                              
//                             for (y = 0; y < Y_const; ++y){                                
//                                 for (x = 0; x < X_const; ++x){
//                                     convolutions[0 : N_BLOCK] += inputs_pointer[0 : N_BLOCK] * (*(++filters_pointer));
//                                     inputs_pointer += N_BLOCK;
//                                 } // x

//                                 inputs_pointer += (-X_const + W_const)*N_BLOCK;
//                             } // y

//                         }

//                         else{
//                             float *restrict inputs_pointer = INPUTS + ti5(n_block, c, mx(mn(h-padding_const, H_const-1), 0), mx(mn(w-padding_const, W_const-1), 0), 0, C_const, H_const, W_const, N_BLOCK);
                            
//                             for (y = 0; y < Y_const; ++y){
                                
//                                 float *restrict inputs_pointer_y = inputs_pointer; // start-of-line pointer
                                
//                                 if ((y + h - padding_const >= 0) && (y + h - padding_const < H_const)){ // i.e, are there any elements in this row that overlap with the image?
//                                     for (x = 0; x < X_const; ++x){
//                                         filters_pointer++;
                                        
//                                         if ((x + w - padding_const >= 0) && (x + w - padding_const < W_const)){
//                                             convolutions[0 : N_BLOCK] += inputs_pointer[0 : N_BLOCK] * (*filters_pointer);
//                                             inputs_pointer += N_BLOCK;
//                                         }
//                                     } // x

//                                     inputs_pointer = inputs_pointer_y + W_const*N_BLOCK; 
//                                 }

//                                 else filters_pointer += X_const;
//                             } // y

//                         } // if-else
//                     } // w
//                 } // h
//             } // c


//             // ~=~=~=~=~=~=~=~= POOLING ~=~=~=~=~=~=~=~= 
//             for (h = 0; h < pooled_H_const; h++){
//                 for (w = 0; w < pooled_W_const; w++){

//                     int h_output = h*pooling_stride_const;
//                     int w_output = w*pooling_stride_const;

//                     // float *restrict outputs_pointer = SCRATCH + ti(k, h_output, w_output, n, output_H_const, output_W_const, N);
//                     float *restrict outputs_pointer = SCRATCH + ti(omp_get_thread_num(), h_output, w_output, 0, output_H_const, output_W_const, N_BLOCK);
                    
//                     // int *restrict argmaxs_pointer = ARGMAXS + ti(k, h_output, w_output, n, output_H_const, output_W_const, N);
//                     int *restrict argmaxs_pointer = ARGMAXS + ti5(n_block, k, h, w, 0, K_const, pooled_H_const, pooled_W_const, N_BLOCK);
//                     float *restrict pooled_outputs_pointer = OUTPUTS + ti5(n_block, k, h, w, 0, K_const, pooled_H_const, pooled_W_const, N_BLOCK);
//                     pooled_outputs_pointer[0 : N_BLOCK] = -1.0e6;
                    
//                     // float *restrict argmaxs = SCRATCH + K_const*output_H_const*output_W_const*N + ti(k, h, w, n, pooled_H_const, pooled_W_const, N);

//                     int outputs_index = h_output*output_W_const + w_output;
//                     for (y = 0; y < pooling_radius_const; y++){
//                         for (x = 0; x < pooling_radius_const; x++){
//                             if (outputs_pointer[0 : N_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
//                                 pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK];
//                                 argmaxs_pointer[0 : N_BLOCK] = outputs_index;
//                             }
//                             outputs_index++;
//                             outputs_pointer += N_BLOCK;
//                         }
//                         outputs_index += output_W_const - pooling_radius_const;
//                         outputs_pointer += (output_W_const - pooling_radius_const)*N_BLOCK;
//                     }

//                 }
//             }
//         } //nk

//     } // pragma_offload
// }

// Convolution after changing to [N_BLOCK, C, H, W, BLOCK] data structure, after the intrinsics
// int *convolution_layer1(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){

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
//         int n_block, n, k, i, j, h, w, c, y, x;
//         int nk, hw, ij, nkhw;

//         // computation of constants
//         int XWN = (-X_const + W_const)*N,
//             HYWN = (H_const-Y_const)*W_const*N;        

//         // SCRATCH[0:K_const*N*output_H_const*output_W_const] = 0.f;
//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(nk, hw, ij, n_block, n, k, h, w, c, y, x, i, j) \
//             shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XWN, HYWN)
        
//         // ~=~=~=~=~=~=~=~= CONVOLUTION ~=~=~=~=~=~=~=~= 
//         for (nk = 0; nk < N/BLOCK*K_const; nk++){
            
//             n_block = nk / K_const;
//             n = n_block*BLOCK;
//             k = md(nk, K_const);

//             SCRATCH[omp_get_thread_num()*output_H_const*output_W_const*BLOCK : output_H_const*output_W_const*BLOCK] = 0.f;

//             for (c = 0; c < C_const; c++){
//                 for (h = 0; h < output_H_const; h++){
//                     for (w = 0; w < output_W_const; w++){

//                         // float *restrict convolutions = SCRATCH + ti(k, h, w, n, output_H_const, output_W_const, N);
//                         float *restrict convolutions = SCRATCH + ti(omp_get_thread_num(), h, w, 0, output_H_const, output_W_const, BLOCK);
//                         float *restrict convolutions_next = SCRATCH + ti(omp_get_thread_num(), h, w+1, 0, output_H_const, output_W_const, BLOCK);
//                 __assume_aligned(convolutions, 64);
//             _mm_prefetch((char *)(convolutions_next), _MM_HINT_ET0);
//             _mm_prefetch((char *)(convolutions_next + 16), _MM_HINT_ET0);
//             _mm_prefetch((char *)(convolutions_next + 32), _MM_HINT_ET0);
//             _mm_prefetch((char *)(convolutions_next + 48), _MM_HINT_ET0);

//                         int kcyx_shift = (k*C_const + c)*Y_const*X_const - 1; // filters
                                
//                         float *restrict filters_pointer = FILTERS + kcyx_shift;
                       
//                         // if we're not on boundary (i.e not affected by padding)
//                         if (w - padding_const >= 0 &&
//                             h - padding_const >= 0 &&
//                             output_W_const - 1 - w >= padding_const  &&
//                             output_H_const - 1 - h >= padding_const){

// #if 1 && defined __MIC__
//             __m512 res_1 = _mm512_extload_ps(convolutions, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
//             __m512 res_2 = _mm512_extload_ps(convolutions + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
//             __m512 res_3 = _mm512_extload_ps(convolutions + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
//             __m512 res_4 = _mm512_extload_ps(convolutions + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
// #endif
//                             float *restrict inputs_pointer = INPUTS + ti5(n_block, c, h - padding_const, w - padding_const, 0, C_const, H_const, W_const, BLOCK);
                              
//                             for (y = 0; y < Y_const; ++y){                                
//                 _mm_prefetch((char *)(inputs_pointer + N), _MM_HINT_T0);
//                 _mm_prefetch((char *)(inputs_pointer + N + 16), _MM_HINT_T0);
//                 _mm_prefetch((char *)(inputs_pointer + N + 32), _MM_HINT_T0);
//                 _mm_prefetch((char *)(inputs_pointer + N + 48), _MM_HINT_T0);
//                                 for (x = 0; x < X_const; ++x)
//                 {
//                         __assume_aligned(inputs_pointer, 64);
//                     filters_pointer++;
//                     _mm_prefetch((char *)(inputs_pointer + N), _MM_HINT_T0);
//                     _mm_prefetch((char *)(inputs_pointer + N + 16), _MM_HINT_T0);
//                     _mm_prefetch((char *)(inputs_pointer + N + 32), _MM_HINT_T0);
//                     _mm_prefetch((char *)(inputs_pointer + N + 48), _MM_HINT_T0);
// #if 1 && defined __MIC__
//                     __m512 v_filters = _mm512_set1_ps(*filters_pointer);
//                     {
//                     res_1 = _mm512_fmadd_ps(_mm512_extload_ps(inputs_pointer +  0, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE), v_filters, res_1); 
//                     res_2 = _mm512_fmadd_ps(_mm512_extload_ps(inputs_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE), v_filters, res_2); 
//                     res_3 = _mm512_fmadd_ps(_mm512_extload_ps(inputs_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE), v_filters, res_3); 
//                     res_4 = _mm512_fmadd_ps(_mm512_extload_ps(inputs_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE), v_filters, res_4); 
//                     }
// #else
//                                     convolutions[0 : BLOCK] += inputs_pointer[0 : BLOCK] * (*(filters_pointer));
// #endif
//                                     inputs_pointer += BLOCK;
//                                 } // x

//                                 inputs_pointer += (-X_const + W_const)*BLOCK;
//                             } // y
// #if 1 && defined __MIC__
//             _mm512_extstore_ps((float *)(convolutions), res_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
//             _mm512_extstore_ps((float *)(convolutions + 16), res_2, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
//             _mm512_extstore_ps((float *)(convolutions + 32), res_3, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
//             _mm512_extstore_ps((float *)(convolutions + 48), res_4, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
// #endif

//                         }

//                         else{
// #if 1 && defined __MIC__
//             __m512 res_1 = _mm512_extload_ps(convolutions, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
//             __m512 res_2 = _mm512_extload_ps(convolutions + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
//             __m512 res_3 = _mm512_extload_ps(convolutions + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
//             __m512 res_4 = _mm512_extload_ps(convolutions + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
// #endif
//                             float *restrict inputs_pointer = INPUTS + ti5(n_block, c, mx(mn(h-padding_const, H_const-1), 0), mx(mn(w-padding_const, W_const-1), 0), 0, C_const, H_const, W_const, BLOCK);
                            
//             int min_x = mx(0, (padding_const - w)); 
//             int max_x = mn(X_const, (W_const + padding_const - w)); 
//             int min_y = mx(0, (padding_const - h)); 
//             int max_y = mn(Y_const, (H_const + padding_const - h)); 
// #if 1
//                 filters_pointer += min_y*X_const;
//                             for (y = min_y; y < max_y; ++y){
//                                 float *restrict inputs_pointer_y = inputs_pointer; // start-of-line pointer
//                 filters_pointer += min_x;
// #pragma unroll (X_const-padding_const)
// #pragma noprefetch
//                                 for (x = min_x; x < max_x; ++x)
//                 {
//                             __assume_aligned(inputs_pointer, 64);
//                         filters_pointer++;
//                             _mm_prefetch((char *)(inputs_pointer + N), _MM_HINT_T0);
//                             _mm_prefetch((char *)(inputs_pointer + N + 16), _MM_HINT_T0);
//                             _mm_prefetch((char *)(inputs_pointer + N + 32), _MM_HINT_T0);
//                             _mm_prefetch((char *)(inputs_pointer + N + 48), _MM_HINT_T0);
// #if 1 && defined __MIC__
//                             __m512 v_filters = _mm512_set1_ps(*filters_pointer);
//                             {
//                         res_1 = _mm512_fmadd_ps(_mm512_extload_ps(inputs_pointer +  0, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE), v_filters, res_1); 
//                         res_2 = _mm512_fmadd_ps(_mm512_extload_ps(inputs_pointer + 16, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE), v_filters, res_2); 
//                         res_3 = _mm512_fmadd_ps(_mm512_extload_ps(inputs_pointer + 32, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE), v_filters, res_3); 
//                         res_4 = _mm512_fmadd_ps(_mm512_extload_ps(inputs_pointer + 48, _MM_UPCONV_PS_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE), v_filters, res_4); 
//                            }
// #else
//                                            convolutions[0 : BLOCK] += inputs_pointer[0 : BLOCK] * (*(filters_pointer));
// #endif
//                                            inputs_pointer += BLOCK;
//                 } // x
//                 filters_pointer += (X_const - max_x);
//                                 inputs_pointer = inputs_pointer_y + W_const*BLOCK; 
//                             } 
// #if 1 && defined __MIC__
//             _mm512_extstore_ps((float *)(convolutions), res_1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
//             _mm512_extstore_ps((float *)(convolutions + 16), res_2, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
//             _mm512_extstore_ps((float *)(convolutions + 32), res_3, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
//             _mm512_extstore_ps((float *)(convolutions + 48), res_4, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
// #endif
//                 filters_pointer += (Y_const - max_y)*X_const;
// #else
//                             for (y = 0; y < Y_const; ++y){
                                
//                                 float *restrict inputs_pointer_y = inputs_pointer; // start-of-line pointer
                                
//                                 if ((y + h - padding_const >= 0) && (y + h - padding_const < H_const)){ // i.e, are there any elements in this row that overlap with the image?
//                                     for (x = 0; x < X_const; ++x){
//                                         filters_pointer++;
                                        
//                                         if ((x + w - padding_const >= 0) && (x + w - padding_const < W_const)){
//                                             convolutions[0 : BLOCK] += inputs_pointer[0 : BLOCK] * (*filters_pointer);
//                                             inputs_pointer += BLOCK;
//                                         }
//                                     } // x

//                                     inputs_pointer = inputs_pointer_y + W_const*BLOCK; 
//                                 }

//                                 else filters_pointer += X_const;
//                             } // y
// #endif 

//                         } // if-else
//                     } // w
//                 } // h
//             } // c


//             // ~=~=~=~=~=~=~=~= POOLING ~=~=~=~=~=~=~=~= 
//             for (h = 0; h < pooled_H_const; h++){
//                 for (w = 0; w < pooled_W_const; w++){

//                     int h_output = h*pooling_stride_const;
//                     int w_output = w*pooling_stride_const;

//                     // float *restrict outputs_pointer = SCRATCH + ti(k, h_output, w_output, n, output_H_const, output_W_const, N);
//                     float *restrict outputs_pointer = SCRATCH + ti(omp_get_thread_num(), h_output, w_output, 0, output_H_const, output_W_const, BLOCK);
                    
//                     // int *restrict argmaxs_pointer = ARGMAXS + ti(k, h_output, w_output, n, output_H_const, output_W_const, N);
//                     int *restrict argmaxs_pointer = ARGMAXS + ti5(n_block, k, h, w, 0, K_const, pooled_H_const, pooled_W_const, BLOCK);
//                     float *restrict pooled_outputs_pointer = OUTPUTS + ti5(n_block, k, h, w, 0, K_const, pooled_H_const, pooled_W_const, BLOCK);
//                     pooled_outputs_pointer[0 : BLOCK] = -1.0e6;
                    
//                     // float *restrict argmaxs = SCRATCH + K_const*output_H_const*output_W_const*N + ti(k, h, w, n, pooled_H_const, pooled_W_const, N);

//                     int outputs_index = h_output*output_W_const + w_output;
//                     for (y = 0; y < pooling_radius_const; y++){
//                         for (x = 0; x < pooling_radius_const; x++){
//                             if (outputs_pointer[0 : BLOCK] > pooled_outputs_pointer[0 : BLOCK]){
//                                 pooled_outputs_pointer[0 : BLOCK] = outputs_pointer[0 : BLOCK];
//                                 argmaxs_pointer[0 : BLOCK] = outputs_index;
//                             }
//                             outputs_index++;
//                             outputs_pointer += BLOCK;
//                         }
//                         outputs_index += output_W_const - pooling_radius_const;
//                         outputs_pointer += (output_W_const - pooling_radius_const)*BLOCK;
//                     }

//                 }
//             }
//         } //nk

//     } // pragma_offload
// }

// convolution after interleaving FILTERS and INPUTS and OUTPUTS
// int *convolution_layer1(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){

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
//         int n_block, n, k_block, k, i, j, h, w, c, y, x;
//         int nk, hw, ij, nkhw;

//         // computation of constants
//         int XWN = (-X_const + W_const)*N,
//             HYWN = (H_const-Y_const)*W_const*N;        

//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(nk, hw, ij, n_block, n, k, k_block, h, w, c, y, x, i, j) \
//             shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XWN, HYWN)

//         // #pragma vector aligned
        
//         // ~=~=~=~=~=~=~=~= CONVOLUTION ~=~=~=~=~=~=~=~= 
//         for (nk = 0; nk < N/N_BLOCK*K_const/K_BLOCK; nk++){
            
//             n_block = nk / (K_const/K_BLOCK);
//             n = n_block*N_BLOCK;
//             k_block = md(nk, K_const/K_BLOCK);
//             k = k_block*K_BLOCK;

//             SCRATCH[omp_get_thread_num()*output_H_const*output_W_const*N_BLOCK*K_BLOCK : output_H_const*output_W_const*N_BLOCK*K_BLOCK] = 0.f;

//             for (c = 0; c < C_const; c++){
//                 for (h = 0; h < output_H_const; h++){
//                     for (w = 0; w < output_W_const; w++){

//                         float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const, output_W_const, K_BLOCK, N_BLOCK);
                                
//                         float *restrict filters_pointer = FILTERS + ti5(k_block, c, 0, 0, 0, C_const, Y_const, X_const, K_BLOCK); //(k*C_const + c)*Y_const*X_const - 1;
                       
//                         // if we're not on boundary (i.e not affected by padding)
//                         if (w - padding_const >= 0 &&
//                             h - padding_const >= 0 &&
//                             output_W_const - 1 - w >= padding_const  &&
//                             output_H_const - 1 - h >= padding_const){

//                             float *restrict inputs_pointer = INPUTS + ti5(n_block, c, h - padding_const, w - padding_const, 0, C_const, H_const, W_const, N_BLOCK);
                              
//                             for (y = 0; y < Y_const; ++y){                                
//                                 for (x = 0; x < X_const; ++x){
//                                     for (int kk = 0; kk < K_BLOCK; kk++){
//                                         convolutions[kk*N_BLOCK : N_BLOCK] += inputs_pointer[0 : N_BLOCK] * (*filters_pointer);
//                                         filters_pointer++;
//                                     }
//                                     inputs_pointer += N_BLOCK;
//                                 } // x

//                                 inputs_pointer += (-X_const + W_const)*N_BLOCK;
//                             } // y

//                         }

//                         else{
//                             float *restrict inputs_pointer = INPUTS + ti5(n_block, c, mx(mn(h-padding_const, H_const-1), 0), mx(mn(w-padding_const, W_const-1), 0), 0, C_const, H_const, W_const, N_BLOCK);
                            
//                             for (y = 0; y < Y_const; ++y){
                                
//                                 float *restrict inputs_pointer_y = inputs_pointer; // start-of-line pointer
                                
//                                 if ((y + h - padding_const >= 0) && (y + h - padding_const < H_const)){ // i.e, are there any elements in this row that overlap with the image?
//                                     for (x = 0; x < X_const; ++x){
                                        
//                                         if ((x + w - padding_const >= 0) && (x + w - padding_const < W_const)){
//                                             for (int kk = 0; kk < K_BLOCK; kk++){
//                                                 convolutions[kk*N_BLOCK : N_BLOCK] += inputs_pointer[0 : N_BLOCK] * (*filters_pointer);
//                                                 filters_pointer++;
//                                             }
//                                             inputs_pointer += N_BLOCK;
//                                         }
//                                         else filters_pointer += K_BLOCK;
//                                     } // x

//                                     inputs_pointer = inputs_pointer_y + W_const*N_BLOCK; 
//                                 }

//                                 else filters_pointer += X_const*K_BLOCK;
//                             } // y

//                         } // if-else
//                     } // w
//                 } // h
//             } // c


//             // ~=~=~=~=~=~=~=~= POOLING ~=~=~=~=~=~=~=~= 
//             for (h = 0; h < pooled_H_const; h++){
//                 for (w = 0; w < pooled_W_const; w++){

//                     int h_output = h*pooling_stride_const;
//                     int w_output = w*pooling_stride_const;
                    
//                     for (int kk = 0; kk < K_BLOCK; kk++){
//                         float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, kk, 0, output_H_const, output_W_const, K_BLOCK, N_BLOCK);
//                         int *restrict argmaxs_pointer = ARGMAXS + ti5(n_block, k + kk, h, w, 0, K_const, pooled_H_const, pooled_W_const, N_BLOCK);
//                         float *restrict pooled_outputs_pointer = OUTPUTS + ti5(n_block, k + kk, h, w, 0, K_const, pooled_H_const, pooled_W_const, N_BLOCK);
//                         pooled_outputs_pointer[0 : N_BLOCK] = -1.0e6;
                        
//                         // float *restrict argmaxs = SCRATCH + K_const*output_H_const*output_W_const*N + ti(k, h, w, n, pooled_H_const, pooled_W_const, N);

//                         int outputs_index = h_output*output_W_const + w_output;
//                         for (y = 0; y < pooling_radius_const; y++){
//                             for (x = 0; x < pooling_radius_const; x++){
//                                 if (outputs_pointer[0 : N_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
//                                     pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK];
//                                     argmaxs_pointer[0 : N_BLOCK] = outputs_index;
//                                 }
//                                 outputs_index++;
//                                 outputs_pointer += K_BLOCK*N_BLOCK;
//                             }
//                             outputs_index += output_W_const - pooling_radius_const;
//                             outputs_pointer += (output_W_const - pooling_radius_const)*K_BLOCK*N_BLOCK;
//                         }
//                     }

//                 }
//             }
//         } //nk

//     } // pragma_offload
// }

// convolution after interleaving N and K and blocking C, before intrinsics
// int *convolution_layer1(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){

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
//     assert(pooled_H_const == ceil((output_H_const - pooling_radius_const + 1.f)/pooling_stride_const));
//     assert(pooled_W_const == ceil((output_W_const - pooling_radius_const + 1.f)/pooling_stride_const));

//     #pragma offload target(mic:MIC_DEV) if(offloaded == 1) \ 
//     in(INPUTS:length(0) REUSE) \ 
//     in(FILTERS:length(0) REUSE) \ 
//     in(OUTPUTS:length(0) REUSE) \
//     in(ARGMAXS:length(0) REUSE) \
//     in(SCRATCH:length(0) REUSE)
//     {
//         int n_block, n, k_block, k, i, j, h, w, c, c_block, y, x;
//         int nk, hw, ij, nkhw;

//         int XWN = (-X_const + W_const)*N,
//             HYWN = (H_const-Y_const)*W_const*N;        

//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(nk, hw, ij, n_block, n, k, k_block, h, w, c, c_block, y, x, i, j) \
//             shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XWN, HYWN)
        
//         // ~=~=~=~=~=~=~=~= CONVOLUTION ~=~=~=~=~=~=~=~= 
//         for (nk = 0; nk < N/N_BLOCK*K_const/K_BLOCK; nk++){
            
//             n_block = nk / (K_const/K_BLOCK);
//             n = n_block*N_BLOCK;
//             k_block = md(nk, K_const/K_BLOCK);
//             k = k_block*K_BLOCK;

//             SCRATCH[omp_get_thread_num()*output_H_const*output_W_const*N_BLOCK*K_BLOCK : output_H_const*output_W_const*N_BLOCK*K_BLOCK] = 0.f;

//             for (c_block = 0; c_block < C_const/C_BLOCK; c_block++){
//                 c = c_block*C_BLOCK;

//                 for (h = 0; h < output_H_const; h++){
//                     for (w = 0; w < output_W_const; w++){

//                         float *restrict convolutions = SCRATCH + ti5(omp_get_thread_num(), h, w, 0, 0, output_H_const, output_W_const, K_BLOCK, N_BLOCK);
                              
//                         for (y = 0; y < Y_const; ++y){                                
//                             for (x = 0; x < X_const; ++x){
                                
//                                 if ((h + y - padding_const >= 0) && (h + y - padding_const < H_const) && (w + x - padding_const >= 0) && (w + x - padding_const < W_const)){
//                                     float *restrict filters_pointer = FILTERS + ti5(k_block, c, y, x, 0, C_const, Y_const, X_const, K_BLOCK);
//                                     float *restrict inputs_pointer = INPUTS + ti5(n_block, c, h + y - padding_const, w + x - padding_const, 0, C_const, H_const, W_const, N_BLOCK);

//                                     for (int cc = 0; cc < C_BLOCK; cc++){
//                                         for (int kk = 0; kk < K_BLOCK; kk++){
//                                             convolutions[kk*N_BLOCK : N_BLOCK] += inputs_pointer[0 : N_BLOCK] * (*filters_pointer);
//                                             filters_pointer++;
//                                         } //kk

//                                         filters_pointer += Y_const*X_const*K_BLOCK - K_BLOCK;
//                                         inputs_pointer += H_const*W_const*N_BLOCK;
//                                     } // cc
//                                 } // if

//                             } // x
//                         } // y
//                     } // w
//                 } // h
//             } // c_block


//             // ~=~=~=~=~=~=~=~= POOLING ~=~=~=~=~=~=~=~= 
//             for (h = 0; h < pooled_H_const; h++){
//                 for (w = 0; w < pooled_W_const; w++){

//                     int h_output = h*pooling_stride_const;
//                     int w_output = w*pooling_stride_const;

//                     int window_width = pooling_radius_const - mx(w_output + pooling_radius_const - output_W_const, 0);
//                     int window_height = pooling_radius_const - mx(h_output + pooling_radius_const - output_H_const, 0);

//                     for (int kk = 0; kk < K_BLOCK; kk++){
//                         float *restrict outputs_pointer = SCRATCH + ti5(omp_get_thread_num(), h_output, w_output, kk, 0, output_H_const, output_W_const, K_BLOCK, N_BLOCK);
//                         int *restrict argmaxs_pointer = ARGMAXS + ti5(n_block, k + kk, h, w, 0, K_const, pooled_H_const, pooled_W_const, N_BLOCK);
//                         float *restrict pooled_outputs_pointer = OUTPUTS + ti5(n_block, k + kk, h, w, 0, K_const, pooled_H_const, pooled_W_const, N_BLOCK);
                        
//                         pooled_outputs_pointer[0 : N_BLOCK] = -1.0e6;

//                         int outputs_index = h_output*output_W_const + w_output;
//                         for (y = 0; y < window_height; y++){
//                             for (x = 0; x < window_width; x++){
//                                 if (outputs_pointer[0 : N_BLOCK] > pooled_outputs_pointer[0 : N_BLOCK]){
//                                     pooled_outputs_pointer[0 : N_BLOCK] = outputs_pointer[0 : N_BLOCK];
//                                     argmaxs_pointer[0 : N_BLOCK] = outputs_index;
//                                 }
//                                 outputs_index++;
//                                 outputs_pointer += K_BLOCK*N_BLOCK;
//                             }
//                             outputs_index += output_W_const - window_width;
//                             outputs_pointer += (output_W_const - window_width)*K_BLOCK*N_BLOCK;
//                         }
//                     }

//                 }
//             }
//         } //nk

//     } // pragma_offload
// }

// convolution for [C, H, W, N] data structure
// int *convolution_layer1(int N, int C, int H, int W, float *restrict INPUTS, int K, int Y, int X, float *restrict FILTERS, float *restrict OUTPUTS, int *restrict ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH){

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
//         int n_block, n, k, i, j, h, w, c, y, x;
//         int nk, hw, ij, nkhw;

//         // computation of constants
//         int XWN = (-X_const + W_const)*N,
//             HYWN = (H_const-Y_const)*W_const*N;        

//         // SCRATCH[0:K_const*N*output_H_const*output_W_const] = 0.f;

//         #pragma omp parallel for \
//             schedule(dynamic) \
//             default(none) \
//             private(nk, hw, ij, n_block, n, k, h, w, c, y, x, i, j) \
//             shared(N, INPUTS, OUTPUTS, FILTERS, ARGMAXS, SCRATCH, XWN, HYWN)

//         // #pragma vector aligned
        
//         // ~=~=~=~=~=~=~=~= CONVOLUTION ~=~=~=~=~=~=~=~= 
//         for (nk = 0; nk < N/BLOCK*K_const; nk++){
            
//             n_block = nk / K_const;
//             n = n_block*BLOCK;
//             k = md(nk, K_const);

//             SCRATCH[omp_get_thread_num()*output_H_const*output_W_const*BLOCK : output_H_const*output_W_const*BLOCK] = 0.f;
//             // float *restrict convolutions = SCRATCH + ti(k, 0, 0, n, output_H_const, output_W_const, N);
//             // for (hw = 0; hw < output_H_const*output_W_const; hw++) convolutions[hw*N : BLOCK] = 0.f;

//             for (c = 0; c < C_const; c++){
//                 for (h = 0; h < output_H_const; h++){
//                     for (w = 0; w < output_W_const; w++){

//                         // float *restrict convolutions = SCRATCH + ti(k, h, w, n, output_H_const, output_W_const, N);
//                         float *restrict convolutions = SCRATCH + ti(omp_get_thread_num(), h, w, 0, output_H_const, output_W_const, BLOCK);

//                         int kcyx_shift = (k*C_const + c)*Y_const*X_const - 1; // filters
                                
//                         float *restrict filters_pointer = FILTERS + kcyx_shift;
                       
//                         // if we're not on boundary (i.e not affected by padding)
//                         if (w - padding_const >= 0 &&
//                             h - padding_const >= 0 &&
//                             output_W_const - 1 - w >= padding_const  &&
//                             output_H_const - 1 - h >= padding_const){

//                             float *restrict inputs_pointer = INPUTS + ti(c, h - padding_const, w - padding_const, n, H_const, W_const, N);
                              
//                             for (y = 0; y < Y_const; ++y){                                
//                                 for (x = 0; x < X_const; ++x){
//                                     convolutions[0 : BLOCK] += inputs_pointer[0 : BLOCK] * (*(++filters_pointer));
//                                     inputs_pointer += N;
//                                 } // x

//                                 inputs_pointer += XWN;
//                             } // y

//                         }

//                         else{
//                             float *restrict inputs_pointer = INPUTS + ti(c, mx(mn(h-padding_const, H_const-1), 0), mx(mn(w-padding_const, W_const-1), 0), n, H_const, W_const, N);
                            
//                             for (y = 0; y < Y_const; ++y){
                                
//                                 float *restrict inputs_pointer_y = inputs_pointer; // start-of-line pointer
                                
//                                 if ((y + h - padding_const >= 0) && (y + h - padding_const < H_const)){ // i.e, are there any elements in this row that overlap with the image?
//                                     for (x = 0; x < X_const; ++x){
//                                         filters_pointer++;
                                        
//                                         if ((x + w - padding_const >= 0) && (x + w - padding_const < W_const)){
//                                             convolutions[0 : BLOCK] += inputs_pointer[0 : BLOCK] * (*filters_pointer);
//                                             inputs_pointer += N;
//                                         }
//                                     } // x

//                                     inputs_pointer = inputs_pointer_y + W_const*N; 
//                                 }

//                                 else filters_pointer += X_const;
//                             } // y

//                         } // if-else
//                     } // w
//                 } // h
//             } // c


//             // ~=~=~=~=~=~=~=~= POOLING ~=~=~=~=~=~=~=~= 
//             for (h = 0; h < pooled_H_const; h++){
//                 for (w = 0; w < pooled_W_const; w++){

//                     int h_output = h*pooling_stride_const;
//                     int w_output = w*pooling_stride_const;

//                     // float *restrict outputs_pointer = SCRATCH + ti(k, h_output, w_output, n, output_H_const, output_W_const, N);
//                     float *restrict outputs_pointer = SCRATCH + ti(omp_get_thread_num(), h_output, w_output, 0, output_H_const, output_W_const, BLOCK);
                    
//                     // int *restrict argmaxs_pointer = ARGMAXS + ti(k, h_output, w_output, n, output_H_const, output_W_const, N);
//                     int *restrict argmaxs_pointer = ARGMAXS + ti(k, h, w, n, pooled_H_const, pooled_W_const, N);
//                     float *restrict pooled_outputs_pointer = OUTPUTS + ti(k, h, w, n, pooled_H_const, pooled_W_const, N);
//                     pooled_outputs_pointer[0 : BLOCK] = -1.0e6;
                    
//                     // float *restrict argmaxs = SCRATCH + K_const*output_H_const*output_W_const*N + ti(k, h, w, n, pooled_H_const, pooled_W_const, N);

//                     int outputs_index = h_output*output_W_const + w_output;
//                     for (y = 0; y < pooling_radius_const; y++){
//                         for (x = 0; x < pooling_radius_const; x++){
//                             if (outputs_pointer[0 : BLOCK] > pooled_outputs_pointer[0 : BLOCK]){
//                                 pooled_outputs_pointer[0 : BLOCK] = outputs_pointer[0 : BLOCK];
//                                 argmaxs_pointer[0 : BLOCK] = outputs_index;
//                             }
//                             outputs_index++;
//                             outputs_pointer += BLOCK;
//                         }
//                         outputs_index += output_W_const - pooling_radius_const;
//                         outputs_pointer += (output_W_const - pooling_radius_const)*BLOCK;
//                     }

//                 }
//             }
//         } //nk

//     } // pragma_offload
// }

// gradient before any optimization
// void convolution_gradient_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){

//     #pragma offload target(mic:MIC_DEV) \ 
//     in(INPUTS:length(0) REUSE) \ 
//     in(FILTERS:length(0) REUSE) \ 
//     in(ARGMAXS:length(0) REUSE) \
//     in(D_POOLED_OUTPUTS:length(0) REUSE) \
//     in(D_FILTERS:length(0) REUSE) \
//     in(SCRATCH:length(0) REUSE)
//     {
//         int khw, h, w, k, c, lin_index, y, x, h_arg, w_arg, n, tmp;
//         int XWN = (-X_const + W_const)*N,
//             HYWN = (H_const-Y_const)*W_const*N;

//         #pragma omp parallel for \
//         default(none) \
//         schedule(dynamic) \
//         private(khw, h, w, k, c, lin_index, y, x, h_arg, w_arg, n, tmp) \
//         shared(N, XWN, HYWN, INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, SCRATCH)

//         for (khw = 0; khw < K_const*pooled_H_const*pooled_W_const; khw++){            

//             k = khw / (pooled_H_const*pooled_W_const);
//             h = md(khw, pooled_H_const*pooled_W_const) / pooled_W_const;
//             w = md(khw, pooled_W_const);

//             float *d_filters_tmp = SCRATCH + omp_get_thread_num()*C_const*Y_const*X_const;
//             d_filters_tmp[0 : C_const*Y_const*X_const] = 0.f;

//             float *restrict d_filters_pointer = d_filters_tmp; // D_FILTERS + k*C_const*Y_const*X_const;
            
//             for (n = 0; n < N; n++){

//                 float *restrict d_pooled_outputs_pointer = D_POOLED_OUTPUTS + ti(k, h, w, n, pooled_H_const, pooled_W_const, N);
//                 int *restrict argmaxs_pointer = ARGMAXS + ti(k, h, w, n, pooled_H_const, pooled_W_const, N);

//                 lin_index = *argmaxs_pointer;
//                 h_arg = lin_index/output_W_const;
//                 w_arg = lin_index - h_arg*output_W_const;

//                 if ((w_arg - padding_const >= 0) &&
//                     (h_arg - padding_const >= 0) &&
//                     (output_W_const - 1 - w_arg >= padding_const) &&
//                     (output_H_const - 1 - h_arg >= padding_const)){
                    
//                     float *restrict inputs_pointer = INPUTS + ti(0, h_arg - padding_const, w_arg - padding_const, n, H_const, W_const, N);

//                     for (c = 0; c < C_const; ++c){                                
//                         for (y = 0; y < Y_const; ++y){                                
//                             for (x = 0; x < X_const; ++x){
//                                 *d_filters_pointer += (*inputs_pointer) * (*d_pooled_outputs_pointer);
//                                 // *d_inputs_pointer += (*d_pooled_outputs_pointer) * (*filters_pointer);
                                
//                                 d_filters_pointer++;
//                                 inputs_pointer += N;
//                             } // x

//                             inputs_pointer += XWN;
//                         } // y
//                         inputs_pointer += HYWN;
//                     } //c
//                 }

//                 else{
//                     float *restrict inputs_pointer = INPUTS + ti(0, mx(mn(h_arg-padding_const, H_const-1), 0), mx(mn(w_arg-padding_const, W_const-1), 0), n, H_const, W_const, N);

//                     for (c = 0; c < C_const; ++c){
//                         for (y = 0; y < Y_const; ++y){
//                             float *restrict inputs_pointer_y = inputs_pointer;

//                             if ((y + h_arg - padding_const >= 0) && (y + h_arg - padding_const < H_const)){
//                                 for (x = 0; x < X_const; ++x){
//                                     if ((x + w_arg - padding_const >= 0) && (x + w_arg - padding_const < W_const)){
//                                         *d_filters_pointer += (*d_pooled_outputs_pointer) * (*inputs_pointer); 
//                                         inputs_pointer += N;
//                                     }
//                                     d_filters_pointer++;
//                                 } // x
//                             }

//                             else d_filters_pointer += X_const; // advance pointer without going into loop
                            
//                             inputs_pointer = inputs_pointer_y + W_const*N; 
//                         } // y

//                         inputs_pointer += HYWN;
//                     } // c
//                 }
            
//             d_filters_pointer = D_FILTERS + k*C_const*Y_const*X_const;
//             d_filters_pointer[0 : C_const*Y_const*X_const] += d_filters_tmp[0 : C_const*Y_const*X_const];

//             } // n
//         } // khw
//     } // pragma offload
// }

// void convolution_gradient_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){

//   #pragma offload target(mic:MIC_DEV) \ 
//   in(INPUTS:length(0) REUSE) \ 
//   in(FILTERS:length(0) REUSE) \ 
//   in(ARGMAXS:length(0) REUSE) \
//   in(D_POOLED_OUTPUTS:length(0) REUSE) \
//   in(D_FILTERS:length(0) REUSE) \
//   in(SCRATCH:length(0) REUSE)
//   {
//         int nkhw, n_block, h, w, k, c, lin_index, y, x, h_arg, w_arg, n, tmp;
//         int XWN = (-X_const + W_const)*N,
//                     HYWN = (H_const-Y_const)*W_const*N;

//         SCRATCH[0 : K_const*output_H_const*output_W_const*N] = 0.f;

//         #pragma omp parallel for \
//         default(none) \
//         schedule(dynamic) \
//         private(n) \
//         shared(N, ARGMAXS, SCRATCH)
//         for (n = 0; n < K_const*pooled_H_const*pooled_W_const*N; n++){
//             int argmax = ARGMAXS[n];
//             SCRATCH[argmax] += 1.f;
//         }

//         #pragma omp parallel for \
//         default(none) \
//         schedule(dynamic) \
//         private(nkhw, n_block, h, w, k, c, lin_index, y, x, h_arg, w_arg, n, tmp) \
//         shared(N, XWN, HYWN, INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, SCRATCH)

//         for (nkhw = 0; nkhw < K_const*output_H_const*output_W_const*N/BLOCK; nkhw++){            

//             n_block = nkhw / (K_const*output_H_const*output_W_const);
//             n = n_block*BLOCK;

//             k = md(nkhw, K_const*output_H_const*output_W_const) / (output_H_const*output_W_const);
//             h = md(nkhw, output_H_const*output_W_const) / output_W_const;
//             w = md(nkhw, output_W_const);

//             float *restrict d_filters_pointer = D_FILTERS + k*C_const*Y_const*X_const;
            
//             int h_pool = h / pooling_stride_const;
//             int w_pool = w / pooling_stride_const;
//             float *restrict d_pooled_outputs_pointer = D_POOLED_OUTPUTS + ti(k, h_pool, w_pool, n, pooled_H_const, pooled_W_const, N);
            
//             // need to modify this to real argmaxs array, which is over the outputs and for each element has the count of the number of times it's the argmax
//             float *restrict argmaxs_pointer = SCRATCH + ti(k, h, w, n, output_H_const, output_W_const, N);

//             if ((w - padding_const >= 0) &&
//                 (h - padding_const >= 0) &&
//                 (output_W_const - 1 - w >= padding_const) &&
//                 (output_H_const - 1 - h >= padding_const)){
                
//                 float *restrict inputs_pointer = INPUTS + ti(0, h, w, n, H_const, W_const, N);

//                 for (c = 0; c < C_const; c++){
//                     for (y = 0; y < Y_const; y++){
//                         for (x = 0; x < X_const; x++){
//                             *d_filters_pointer += __sec_reduce_add(inputs_pointer[0 : BLOCK] * argmaxs_pointer[0 : BLOCK] * d_pooled_outputs_pointer[0 : BLOCK]);
//                             d_filters_pointer++;
//                             inputs_pointer += N;
//                         } // x

//                         inputs_pointer += XWN;
//                     } // y
//                     inputs_pointer += HYWN;
//                 }
//             }

//             // else{
//             //     float *restrict inputs_pointer = INPUTS + cHWn - 1 + mx(mn(h_arg-padding_const, H_const-1), 0)*W_const + mx(mn(w_arg-padding_const, W_const-1), 0);

//             //     for (y = 0; y < Y_const; ++y){
//             //         float *restrict inputs_pointer_ncyX = inputs_pointer; // start-of-line pointer
//             //         if ((y + h_arg - padding_const >= 0) && (y + h_arg - padding_const < H_const)){
//             //             for (x = 0; x < X_const; ++x){
//             //                 d_filters_pointer++;
//             //                 if ((x + w_arg - padding_const >= 0) && (x + w_arg - padding_const < W_const))
//             //                     *d_filters_pointer += (*d_pooled_outputs_pointer) * (*(++inputs_pointer)); 
//             //             } // x
//             //         }

//             //         else d_filters_pointer += X_const; // advance pointer without going into loop
//             //         inputs_pointer = inputs_pointer_ncyX + W_const; 
//             //     } // y
//             // }

//         } // nkhw
  
//   }
// }

// // gradient for inputs data structure [N, H, W, C]
// void convolution_gradient_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){

//     #pragma offload target(mic:MIC_DEV) \ 
//     in(INPUTS:length(0) REUSE) \ 
//     in(FILTERS:length(0) REUSE) \ 
//     in(ARGMAXS:length(0) REUSE) \
//     in(D_POOLED_OUTPUTS:length(0) REUSE) \
//     in(D_FILTERS:length(0) REUSE) \
//     in(SCRATCH:length(0) REUSE)
//     {
//         int khw, h, w, k, c, lin_index, y, x, h_arg, w_arg, n, tmp;
//         int XWN = (-X_const + W_const)*N,
//             HYWN = (H_const-Y_const)*W_const*N;

//         #pragma omp parallel for \
//         default(none) \
//         schedule(dynamic) \
//         private(khw, h, w, k, c, lin_index, y, x, h_arg, w_arg, n, tmp) \
//         shared(N, XWN, HYWN, INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, SCRATCH)

//         for (khw = 0; khw < K_const*pooled_H_const*pooled_W_const; khw++){            

//             k = khw / (pooled_H_const*pooled_W_const);
//             h = md(khw, pooled_H_const*pooled_W_const) / pooled_W_const;
//             w = md(khw, pooled_W_const);

//             float *restrict d_filters_tmp = SCRATCH + omp_get_thread_num()*Y_const*X_const*C_const;
//             d_filters_tmp[0 : C_const*Y_const*X_const] = 0.f;

//             float *restrict d_filters_pointer = d_filters_tmp; // D_FILTERS + k*C_const*Y_const*X_const;
            
//             for (n = 0; n < N; n++){

//                 float *restrict d_pooled_outputs_pointer = D_POOLED_OUTPUTS + ti(k, h, w, n, pooled_H_const, pooled_W_const, N);
//                 int *restrict argmaxs_pointer = ARGMAXS + ti(k, h, w, n, pooled_H_const, pooled_W_const, N);

//                 lin_index = *argmaxs_pointer;
//                 h_arg = lin_index/output_W_const;
//                 w_arg = lin_index - h_arg*output_W_const;

//                 // float *restrict inputs_pointer = INPUTS + ti(n, h_arg - padding_const, w_arg - padding_const, 0, H_const, W_const, C_const);
//                 float *restrict inputs_pointer = INPUTS + ti(n, mx(mn(h_arg-padding_const, H_const-1), 0), mx(mn(w_arg-padding_const, W_const-1), 0), 0, H_const, W_const, C_const);

//                 const int x_invalid_left = mx(padding_const - w_arg, 0);
//                 const int x_invalid_right = mx(w_arg - padding_const + X_const - W_const, 0);
//                 const int x_len = X_const - x_invalid_left - x_invalid_right;

//                 for (y = 0; y < Y_const; ++y){
//                     if ((y + h_arg - padding_const >= 0) && (y + h_arg - padding_const < H_const)){
//                             d_filters_pointer[x_invalid_left*C_const : x_len*C_const] += inputs_pointer[0 : x_len*C_const] * (*d_pooled_outputs_pointer);
                            
//                             d_filters_pointer += X_const*C_const;
//                             inputs_pointer += W_const*C_const;
//                     } // if
//                 } // y
            
//             d_filters_pointer = D_FILTERS + k*Y_const*X_const*C_const;
//             d_filters_pointer[0 : Y_const*X_const*C_const] += d_filters_tmp[0 : Y_const*X_const*C_const];

//             } // n
//         } // khw
//     } // pragma offload
// }

// void *initialize_locks(){

//     void *writelock[C_BLOCK_GRAD*Y_BLOCK_GRAD*16];

//     #pragma offload target(mic:MIC_DEV) \
//     in(writelock:length(0))
//     { 
//         omp_lock_t *writelock_casted = (omp_lock_t *) writelock;
//         for(int i = 0; i < C_BLOCK_GRAD*Y_BLOCK_GRAD; i++)      
//             omp_init_lock(&writelock_casted[16*i]);

//     }

//     return (void *)writelock;
// }


// computes FFT/iFFT of N contiguous HxW maps
void fft(int N, int H, int W, int IS_FORWARD, float *restrict SPATIALS, float complex *restrict FREQUENCIES){
    #pragma offload target(mic:MIC_DEV) \ 
        in(SPATIALS:length(0) REUSE) \
        in(FREQUENCIES:length(0) REUSE)
    {

        DFTI_DESCRIPTOR_HANDLE hand; MKL_LONG status; 
        
        MKL_LONG transform_dimensions[] = {H, W};
        MKL_LONG spatial_strides[] = {0, W, 1}; // spatial shape: HxW
        MKL_LONG frequency_strides[] = {0, W/2 + 1, 1}; // frequency shape: H*(floor(W/2) + 1)
        // printf("Starting.\n");

        status = DftiCreateDescriptor(&hand, DFTI_SINGLE, DFTI_REAL, 2, transform_dimensions);
        // printf("Create %d\n", status);

        status = DftiSetValue(hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        // printf("place %d\n", status);
        float scale = pow((float) H*W, -0.5);
        status = DftiSetValue(hand, DFTI_FORWARD_SCALE, scale);
        // printf("forward %d\n", status);
        status = DftiSetValue(hand, DFTI_BACKWARD_SCALE, scale);
        // printf("backward %d\n", status);
        status = DftiSetValue(hand, DFTI_NUMBER_OF_TRANSFORMS, N);
        // printf("num %d\n", status);

        status = DftiSetValue(hand, DFTI_THREAD_LIMIT, 236);

        status = DftiSetValue(hand, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        // printf("storage %d\n", status);

        if (IS_FORWARD == 1){
            status = DftiSetValue(hand, DFTI_INPUT_STRIDES, spatial_strides);
            // printf("in strides %d\n", status);
            status = DftiSetValue(hand, DFTI_OUTPUT_STRIDES, frequency_strides);
            // printf("out strides %d\n", status);

            status = DftiSetValue(hand, DFTI_INPUT_DISTANCE, H*W);
            // printf("input %d\n", status);
            status = DftiSetValue(hand, DFTI_OUTPUT_DISTANCE, H*(W/2 + 1));
            // printf("output %d\n", status);
        }
        else{
            status = DftiSetValue(hand, DFTI_INPUT_STRIDES, frequency_strides);
            status = DftiSetValue(hand, DFTI_OUTPUT_STRIDES, spatial_strides);

            status = DftiSetValue(hand, DFTI_INPUT_DISTANCE, H*(W/2 + 1));
            status = DftiSetValue(hand, DFTI_OUTPUT_DISTANCE, H*W);
        }

        status = DftiCommitDescriptor(hand);

        // printf("commit %d\n", status);
        if (IS_FORWARD == 1) status = DftiComputeForward(hand, SPATIALS, FREQUENCIES);
        else              status = DftiComputeBackward(hand, FREQUENCIES, SPATIALS);
        
        // printf("fft %d\n", status);
        status = DftiFreeDescriptor(&hand);
        // printf("free %d\n", status);

    }
}

void fft_full(int N, int H, int W, int IS_FORWARD, float *restrict SPATIALS, float complex *restrict FREQUENCIES){
    #pragma offload target(mic:MIC_DEV) \ 
        in(SPATIALS:length(0) REUSE) \
        in(FREQUENCIES:length(0) REUSE)
    {

        DFTI_DESCRIPTOR_HANDLE hand; MKL_LONG status; 
        
        MKL_LONG transform_dimensions[] = {H, W};
        MKL_LONG spatial_strides[] = {0, W, 1}; // spatial shape: HxW
        MKL_LONG frequency_strides[] = {0, W, 1}; // frequency shape: H*(floor(W/2) + 1)
        // printf("Starting.\n");

        status = DftiCreateDescriptor(&hand, DFTI_SINGLE, DFTI_COMPLEX, 2, transform_dimensions);
        // printf("Create %d\n", status);

        status = DftiSetValue(hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        // printf("place %d\n", status);
        float scale = pow((float) H*W, -0.5);
        status = DftiSetValue(hand, DFTI_FORWARD_SCALE, scale);
        // printf("forward %d\n", status);
        status = DftiSetValue(hand, DFTI_BACKWARD_SCALE, scale);
        // printf("backward %d\n", status);
        status = DftiSetValue(hand, DFTI_NUMBER_OF_TRANSFORMS, N);
        // printf("num %d\n", status);

        status = DftiSetValue(hand, DFTI_THREAD_LIMIT, 236);

        status = DftiSetValue(hand, DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX);
        // printf("storage %d\n", status);
        status = DftiSetValue(hand, DFTI_INPUT_DISTANCE, H*W);
            // printf("input %d\n", status);
        status = DftiSetValue(hand, DFTI_OUTPUT_DISTANCE, H*W);

        status = DftiSetValue(hand, DFTI_INPUT_STRIDES, frequency_strides);
        status = DftiSetValue(hand, DFTI_OUTPUT_STRIDES, spatial_strides);
        // printf("input %d\n", status);

        status = DftiCommitDescriptor(hand);

        // printf("commit %d\n", status);
        if (IS_FORWARD == 1) status = DftiComputeForward(hand, SPATIALS, FREQUENCIES);
        else                 status = DftiComputeBackward(hand, FREQUENCIES, SPATIALS);
        
        // printf("fft %d\n", status);
        status = DftiFreeDescriptor(&hand);
        // printf("free %d\n", status);

    }
}

void cmult(int N, int K, int C, int HW, float complex *restrict INPUTS, float complex *restrict FILTERS, float complex *restrict OUTPUTS){
    #pragma offload target(mic:MIC_DEV) \ 
        in(INPUTS:length(0) REUSE) \
        in(FILTERS:length(0) REUSE) \ 
        in(OUTPUTS:length(0) REUSE)
    {
        int nk;
        #pragma omp parallel for private(nk) shared(INPUTS, FILTERS, OUTPUTS, N, K, C, HW)
        for (nk = 0; nk < N*K; nk++){
            int n = nk/K;
            int k = md(nk, K);
            assert(n*K + k == nk);

            OUTPUTS[nk*HW : HW] = (float complex) (0.f + I*0.f);
            for (int c = 0; c < C; c++){
                // for (int hw = 0; hw < HW; hw++){
                //     OUTPUTS[ti3(n, k, hw, K, HW)] += FILTERS[ti3(k, c, hw, C, HW)] * INPUTS[ti3(n, c, hw, C, HW)];
                // }
                OUTPUTS[nk*HW : HW] += FILTERS[(k*C + c)*HW : HW] * INPUTS[(n*C + c)*HW : HW];
            }
        }
    }
}

void cmult_gradient(int N, int K, int C, int HW, float complex *restrict INPUTS, float complex *restrict GRADIENT_INPUTS, float complex *restrict FILTERS, float complex *restrict GRADIENT_FILTERS, float complex *restrict GRADIENT_OUTPUTS){
    #pragma offload target(mic:MIC_DEV) \ 
        in(INPUTS:length(0) REUSE) \
        in(GRADIENT_INPUTS:length(0) REUSE) \
        in(FILTERS:length(0) REUSE) \ 
        in(GRADIENT_FILTERS:length(0) REUSE) \
        in(GRADIENT_OUTPUTS:length(0) REUSE)
    {
        // int nk;
        int kc;
        #pragma omp parallel for private(kc) shared(INPUTS, GRADIENT_INPUTS, FILTERS, GRADIENT_FILTERS, GRADIENT_OUTPUTS, N, K, C, HW)
        // for (nk = 0; nk < N*K; nk++){
        for (kc = 0; kc < K*C; kc++){
            int k = kc/C;
            int c = md(kc, C);
            assert(k*C + c == kc);

            // GRADIENT_FILTERS[kc*HW : HW] = 0.f;
            for (int n = 0; n < N; n++){
                // for (int hw = 0; hw < HW; hw++){
                //     GRADIENT_FILTERS[ti3(k, c, hw, C, HW)] += GRADIENT_OUTPUTS[ti3(n, k, hw, K, HW)] * conjf(INPUTS[ti3(n, c, hw, C, HW)]);
                // }
                GRADIENT_FILTERS[kc*HW : HW] += GRADIENT_OUTPUTS[(n*K + k)*HW : HW] * conjf(INPUTS[(n*C + c)*HW : HW]);
            }
        }
    }
}

void wipe_out_irrelevant_entries(int N, int H, int W, float complex *restrict INPUTS){
    #pragma offload target(mic:MIC_DEV) \ 
        in(INPUTS:length(0) REUSE)
    {
        int n;
        #pragma omp parallel for private(n) shared(INPUTS, N, H, W)
        for (n = 0; n < N; n++){
            // for (int h = 0; h < H; h++){
            //     for (int w = 1; w < W-1; w++){
            //         INPUTS[ti3(n, h, w, H, W)] -= I*cimagf(INPUTS[ti3(n, h, w, H, W)]);
            //     }
            // }

            int w = W-1;
            for (int h = H/2+1; h < H; h++) INPUTS[ti3(n, h, w, H, W)] = (float complex) 0.f + 0.f*I;
            // for (int h = 1; h < H/2; h++) INPUTS[ti3(n, h, w, H, W)] = (float complex) 0.f + 0.f*I;
                // for (int h = 1; h < H/2; h++) INPUTS[ti3(n, h, w, H, W)] -= I*cimagf(INPUTS[ti3(n, h, w, H, W)]);

            w = 0;
            for (int h = H/2+1; h < H; h++) INPUTS[ti3(n, h, w, H, W)] = (float complex) 0.f + 0.f*I;
            // for (int h = 1; h < H/2; h++) INPUTS[ti3(n, h, w, H, W)] = (float complex) 0.f + 0.f*I;
                // for (int h = 1; h < H/2; h++) INPUTS[ti3(n, h, w, H, W)] -= I*cimagf(INPUTS[ti3(n, h, w, H, W)]);

            INPUTS[ti3(n, 0, 0, H, W)] -= I*cimagf(INPUTS[ti3(n, 0, 0, H, W)]);
            INPUTS[ti3(n, H/2, 0, H, W)] -= I*cimagf(INPUTS[ti3(n, H/2, 0, H, W)]);
            INPUTS[ti3(n, 0, W-1, H, W)] -= I*cimagf(INPUTS[ti3(n, 0, W-1, H, W)]);
            INPUTS[ti3(n, H/2, W-1, H, W)] -= I*cimagf(INPUTS[ti3(n, H/2, W-1, H, W)]);
        }
    }
}

void wipe_out_irrelevant_entries_full(int N, int H, int W, float complex *restrict INPUTS){
    #pragma offload target(mic:MIC_DEV) \ 
        in(INPUTS:length(0) REUSE)
    {
        int n;
        #pragma omp parallel for private(n) shared(INPUTS, N, H, W)
        for (n = 0; n < N; n++){
            
            for (int h = 0; h < H; h++){
                for (int w = W/2 + 1; w < W; w++){
                    INPUTS[ti3(n, h, w, H, W)] = (float complex) 0.f + 0.f*I;
                }
            }

            int w = W/2;
            for (int h = H/2+1; h < H; h++) INPUTS[ti3(n, h, w, H, W)] = (float complex) 0.f + 0.f*I;

            w = 0;
            for (int h = H/2+1; h < H; h++) INPUTS[ti3(n, h, w, H, W)] = (float complex) 0.f + 0.f*I;

            INPUTS[ti3(n, 0, 0, H, W)] -= I*cimagf(INPUTS[ti3(n, 0, 0, H, W)]);
            INPUTS[ti3(n, H/2, 0, H, W)] -= I*cimagf(INPUTS[ti3(n, H/2, 0, H, W)]);
            INPUTS[ti3(n, 0, W/2, H, W)] -= I*cimagf(INPUTS[ti3(n, 0, W/2, H, W)]);
            INPUTS[ti3(n, H/2, W/2, H, W)] -= I*cimagf(INPUTS[ti3(n, H/2, W/2, H, W)]);
        }
    }
}

void fft_conjugate_symmetry_scaling(int N, int K, int H, int W_SPATIAL, float complex *restrict GRADIENT, float complex *restrict SCRATCH){
    
    #pragma offload target(mic:MIC_DEV) \ 
        in(GRADIENT:length(0) REUSE) \
        in(SCRATCH:length(0) REUSE)
    {   
        int W_HALF = W_SPATIAL/2 + 1;
        int IS_EVEN = W_SPATIAL/2 - (W_SPATIAL - 1)/2; // 1 if even, 0 if offloaded
        int nk, h, w;
        // SCRATCH[0 : N*K*H*W_HALF] = GRADIENT[0 : N*K*H*W_HALF];
       
        #pragma omp parallel for private(nk, h, w) shared(GRADIENT, N, K, H, W_SPATIAL, W_HALF, SCRATCH)
        for (nk = 0; nk < N*K; nk++){
            int n = nk/K;
            int k = md(nk, K);
            assert(n*K + k == nk);

            for (h = 0; h < H; h++){
                for (w = 0; w < W_HALF; w++){ // scan over all elements that have been removed due to conjugate symmetry
                    // GRADIENT[ti(n, k, h, w, K, H, W_HALF)] += conjf(SCRATCH[ti(n, k, h, w, K, H, W_HALF)]);
                    
                    GRADIENT[ti(n, k, h, w, K, H, W_HALF)] *= (double complex) 2.f + 0.f*I;
                    
                    // GRADIENT[ti(n, k, h, w, K, H, W_HALF)] += -2.*I*cimagf(GRADIENT[ti(n, k, h, w, K, H, W_HALF)]);
                    // GRADIENT[ti(n, k, h, w, K, H, W_HALF)] += -I*crealf(GRADIENT[ti(n, k, h, w, K, H, W_HALF)]);
                }
                // for (w = 0; w < W_SPATIAL - W_HALF; w++){ // scan over all elements that have been removed due to conjugate symmetry
                //     GRADIENT[ti(n, k, h, W_HALF - 1 - w - IS_EVEN, K, H, W_HALF)] += conjf(SCRATCH[ti(n, k, md(H-h, H), W_HALF - 1 - w, K, H, W_HALF)]);
                // }
            }

            GRADIENT[ti(n, k, 0, 0, K, H, W_HALF)] *= 0.5;
            GRADIENT[ti(n, k, H/2, 0, K, H, W_HALF)] *= 0.5;
            GRADIENT[ti(n, k, 0, W_HALF-1, K, H, W_HALF)] *= 0.5;
            GRADIENT[ti(n, k, H/2, W_HALF-1, K, H, W_HALF)] *= 0.5;

            // int w = W_HALF-1;
            // for (int h = H/2+1; h < H; h++) INPUTS[ti(n, k, h, w, K, H, W_HALF)] *= 2.;

            // w = 0;
            // for (int h = H/2+1; h < H; h++) INPUTS[ti(n, k, h, w, K, H, W_HALF)] *= 2.;

            // for k2=N2/2+1…N2-1: Z{k1,k2,m} = conj(AZ[os[0]+(N1-k1)%N1*os[1]+(N2-k2)%N2*os[2]+m*odist])
        }
    }
}

void conjugate(int N, float *restrict A){
    #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE)
    {   
        A[1:N:2] *= -1.f;
    }
}

void real_and_imaginary(int N, float *restrict A, float *restrict REAL, float *restrict IMAGINARY){
    #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        in(REAL:length(0) REUSE) \
        in(IMAGINARY:length(0) REUSE)
    {   
        REAL[0:N] = A[0:N:2];
        IMAGINARY[0:N] = A[1:N:2];
    }
}

void amplitude_and_phase(int N, float complex *restrict A, float *restrict AMPLITUDE, float *restrict PHASE){
    #pragma offload target(mic:MIC_DEV) \ 
        in(A:length(0) REUSE) \
        in(AMPLITUDE:length(0) REUSE) \
        in(PHASE:length(0) REUSE)
    {   
        // int n;
        // #pragma omp parallel for private(n) shared(AMPLITUDE, PHASE, A)
        // for (n = 0; n < N; n++){
        //     AMPLITUDE[n] = cabsf(A[n]);
        //     PHASE[n] = cargf(A[n]);
        // }
        AMPLITUDE[0:N] = crealf(A[0:N])*crealf(A[0:N]) + cimagf(A[0:N])*cimagf(A[0:N]);
        PHASE[0:N] = cargf(A[0:N]);
    }
}