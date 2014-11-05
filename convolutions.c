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


// gradient for intense blocking/data sturctures

// INPUTS data structure [N, C/C_BLOCK, H, W, C_BLOCK]
// D_FILTERS/FILTERS data structure [C/C_BLOCK, Y/Y_BLOCK, K, Y_BLOCK, X, C_BLOCK]
// ARGMAXS/OUTPUTS/D_POOLED_OUTPUTS data structure [N, pooled_H, pooled_W, K]
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
    
    omp_lock_t writelock[C_BLOCK_GRAD*Y_BLOCK_GRAD*16];

        for(int i = 0; i < C_BLOCK_GRAD*Y_BLOCK_GRAD; i++)      
            omp_init_lock(&writelock[16*i]);
    
    double st = omp_get_wtime();
        #pragma omp parallel for \
        default(none) \
        schedule(dynamic) \
        shared(N, INPUTS, ARGMAXS, FILTERS, D_POOLED_OUTPUTS, D_FILTERS, SCRATCH, C_blocks, Y_blocks, N_blocks, H_blocks, W_blocks, writelock)
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
        D_FILTERS[c_block * y_block * K_const*Y_BLOCK_GRAD*X_const*C_BLOCK_GRAD : K_const*Y_BLOCK_GRAD*X_const*C_BLOCK_GRAD] += local_scratch[ 0 : K_const*Y_BLOCK_GRAD*X_const*C_BLOCK_GRAD];
        }
        omp_unset_lock(&writelock[16*c_block*y_block]);
        } // outer
    double end = omp_get_wtime();
    printf("Time first loop = %.5lf\n", (end - st));

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


// INPUTS data structure [N, C/C_BLOCK, H, W, C_BLOCK]
// D_FILTERS/FILTERS data structure [C/C_BLOCK, Y/Y_BLOCK, K, Y_BLOCK, X, C_BLOCK]
// ARGMAXS/OUTPUTS/D_POOLED_OUTPUTS data structure [N, pooled_H, pooled_W, K]
void convolution_gradient_layer2(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_POOLED_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH){
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
    
    omp_lock_t writelock[C_BLOCK_GRAD*Y_BLOCK_GRAD*16];

        for(int i = 0; i < C_BLOCK_GRAD*Y_BLOCK_GRAD; i++)      
            omp_init_lock(&writelock[16*i]);
    
    double st = omp_get_wtime();
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
        D_FILTERS[c_block * y_block * K_const*Y_BLOCK_GRAD*X_const*C_BLOCK_GRAD : K_const*Y_BLOCK_GRAD*X_const*C_BLOCK_GRAD] += local_scratch[ 0 : K_const*Y_BLOCK_GRAD*X_const*C_BLOCK_GRAD];
        }
        omp_unset_lock(&writelock[16*c_block*y_block]);
        } // outer
    double end = omp_get_wtime();
    printf("Time first loop = %.5lf\n", (end - st));

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

void permute_dimensions(int D1, int D2, int D3, int D4, int perm1, int perm2, int perm3, int perm4, float *restrict TENSOR, float *restrict SCRATCH){

    #pragma offload target(mic:MIC_DEV) \ 
        in(TENSOR:length(0) REUSE) \
        in(SCRATCH:length(0) REUSE)
    {   
        int i1, i2, i3, i4;
        SCRATCH[0 : D1*D2*D3*D4] = TENSOR[0 : D1*D2*D3*D4];

        int dimensions[4] = {D1, D2, D3, D4};
        int P1 = dimensions[perm1];
        int P2 = dimensions[perm2];
        int P3 = dimensions[perm3];
        int P4 = dimensions[perm4];

        // printf("(%d, %d, %d, %d)\n (%d, %d, %d, %d)\n", D1, D2, D3, D4, P1, P2, P3, P4);

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(i1, i2, i3, i4) \
            shared(TENSOR, SCRATCH, D1, D2, D3, D4, P1, P2, P3, P4, perm1, perm2, perm3, perm4)

        for (int i1i2 = 0; i1i2 < D1*D2; i1i2++){
            i1 = i1i2 / D2;
            i2 = md(i1i2, D2);

            int indices[4];
            for (i3 = 0; i3 < D3; i3++){
                for (i4 = 0; i4 < D4; i4++){
                    indices[0] = i1;
                    indices[1] = i2;
                    indices[2] = i3;
                    indices[3] = i4;
                    int p1 = indices[perm1];
                    int p2 = indices[perm2];
                    int p3 = indices[perm3];
                    int p4 = indices[perm4];
                    TENSOR[ti(p1, p2, p3, p4, P2, P3, P4)] = SCRATCH[ti(i1, i2, i3, i4, D2, D3, D4)];
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
        SCRATCH[0 : D1*D2*D3*D4] = (float) TENSOR[0 : D1*D2*D3*D4];

        int dimensions[4] = {D1, D2, D3, D4};
        int P1 = dimensions[perm1];
        int P2 = dimensions[perm2];
        int P3 = dimensions[perm3];
        int P4 = dimensions[perm4];

        // printf("(%d, %d, %d, %d)\n (%d, %d, %d, %d)\n", D1, D2, D3, D4, P1, P2, P3, P4);

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(i1, i2, i3, i4) \
            shared(TENSOR, SCRATCH, D1, D2, D3, D4, P1, P2, P3, P4, perm1, perm2, perm3, perm4)

        for (int i1i2 = 0; i1i2 < D1*D2; i1i2++){
            i1 = i1i2 / D2;
            i2 = md(i1i2, D2);

            int indices[4];
            for (i3 = 0; i3 < D3; i3++){
                for (i4 = 0; i4 < D4; i4++){
                    indices[0] = i1;
                    indices[1] = i2;
                    indices[2] = i3;
                    indices[3] = i4;
                    int p1 = indices[perm1];
                    int p2 = indices[perm2];
                    int p3 = indices[perm3];
                    int p4 = indices[perm4];
                    TENSOR[ti(p1, p2, p3, p4, P2, P3, P4)] = (int) SCRATCH[ti(i1, i2, i3, i4, D2, D3, D4)];
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
        
        // mkl_somatcopy('R', 'T', N, C, 1.0, TENSOR, C, SCRATCH, N);
        int n, c;
        SCRATCH[0 : N*C] = TENSOR[0 : N*C];

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(n, c) \
            shared(N, TENSOR, SCRATCH, C)

        for (int c = 0; c < C; c++){
            TENSOR[c*N : N] = SCRATCH[c : N : C];
        }            

    } // pragma offload
}

void transpose_replace_int(int N, int C, int *restrict TENSOR, float *restrict SCRATCH){

    #pragma offload target(mic:MIC_DEV) \ 
        in(TENSOR:length(0) REUSE) \
        in(SCRATCH:length(0) REUSE)
    {   
        
        // mkl_somatcopy('R', 'T', N, C, 1.0, TENSOR, C, SCRATCH, N);
        int n, c;
        SCRATCH[0 : N*C] = (float) TENSOR[0 : N*C];

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(n, c) \
            shared(N, TENSOR, SCRATCH, C)

        for (int c = 0; c < C; c++){
            TENSOR[c*N : N] = (int) SCRATCH[c : N : C];
        }            

    } // pragma offload
}

void interleave_for_gradient(const int N, const int C, const int H, const int W, const int BLOCKSIZE, float *restrict TENSOR, float *restrict SCRATCH){

    #pragma offload target(mic:MIC_DEV) \ 
        in(TENSOR:length(0) REUSE) \
        in(SCRATCH:length(0) REUSE)
    {
        int c_block, cc, n, c, h, w;
        int C_BLOCKSIZE = C/BLOCKSIZE;
        
        SCRATCH[0 : N*C*H*W] = TENSOR[0 : N*C*H*W];

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

            
            for (h = 0; h < H; h++){
                for (w = 0; w < W; w++){
                    TENSOR[ti5(n, c_block, h, w, cc, C_BLOCKSIZE, H, W, BLOCKSIZE)] = SCRATCH[ti(n, c, h, w, C, H, W)];
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
        
        SCRATCH[0 : N*C*H*W] = TENSOR[0 : N*C*H*W];

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

            
            for (h = 0; h < H; h++){
                for (w = 0; w < W; w++){
                    TENSOR[ti(n, c, h, w, C, H, W)] = SCRATCH[ti5(n, c_block, h, w, cc, C_BLOCKSIZE, H, W, BLOCKSIZE)];
                }
            }

        } // nc
    } // pragma offload
}

void interleave_block(const int N, const int C, const int BLOCKSIZE, float *restrict TENSOR, float *restrict SCRATCH){

    #pragma offload target(mic:MIC_DEV) \ 
        in(TENSOR:length(0) REUSE) \
        in(SCRATCH:length(0) REUSE)
    {
        int n_block, n, c;
        
        SCRATCH[0 : N*C] = TENSOR[0 : N*C];

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(n_block, n, c) \
            shared(N, TENSOR, SCRATCH, C, BLOCKSIZE)

        for (int c = 0; c < C; c++){
            // float *restrict tensor_pointer = TENSOR + ti3(n_block, c, 0, C, BLOCKSIZE);
            // float *restrict scratch_pointer = SCRATCH + ti2(c, n, N);

            for (n_block = 0; n_block < N/BLOCKSIZE; n_block++){
                TENSOR[(n_block*C + c)*BLOCKSIZE : BLOCKSIZE] = SCRATCH[c*N + n_block*BLOCKSIZE : BLOCKSIZE];    
            }
            
        } // nc
    } // pragma offload
}

void interleave_block_int(const int N, const int C, const int BLOCKSIZE, int *restrict TENSOR, float *restrict SCRATCH){

    #pragma offload target(mic:MIC_DEV) \ 
        in(TENSOR:length(0) REUSE) \
        in(SCRATCH:length(0) REUSE)
    {
        int n_block, n, c;
        
        SCRATCH[0 : N*C] = (float) TENSOR[0 : N*C];

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(n_block, n, c) \
            shared(N, TENSOR, SCRATCH, C, BLOCKSIZE)

        for (int nc = 0; nc < N/BLOCKSIZE*C; nc++){
            n_block = nc / C;
            n = n_block*BLOCKSIZE;
            c = md(nc, C);
            
            int *restrict tensor_pointer = TENSOR + ti3(n_block, c, 0, C, BLOCKSIZE);
            float *restrict scratch_pointer = SCRATCH + ti2(c, n, N);

            tensor_pointer[0 : BLOCKSIZE] = (int) scratch_pointer[0 : BLOCKSIZE];
        } // nc
    } // pragma offload
}

void uninterleave_block(const int N, const int C, const int BLOCKSIZE, float *restrict TENSOR, float *restrict SCRATCH){

    #pragma offload target(mic:MIC_DEV) \ 
        in(TENSOR:length(0) REUSE) \
        in(SCRATCH:length(0) REUSE)
    {
        int n_block, n, c;
        
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
    } // pragma offload
}

void uninterleave_block_int(const int N, const int C, const int BLOCKSIZE, int *restrict TENSOR, float *restrict SCRATCH){

    #pragma offload target(mic:MIC_DEV) \ 
        in(TENSOR:length(0) REUSE) \
        in(SCRATCH:length(0) REUSE)
    {
        int n_block, n, c;
        
        SCRATCH[0 : N*C] = (float) TENSOR[0 : N*C];

        #pragma omp parallel for \
            schedule(dynamic) \
            default(none) \
            private(n_block, n, c) \
            shared(N, TENSOR, SCRATCH, C, BLOCKSIZE)

        for (int nc = 0; nc < N/BLOCKSIZE*C; nc++){
            n_block = nc / C;
            n = n_block*BLOCKSIZE;
            c = md(nc, C);
            
            TENSOR[ti2(c, n, N) : BLOCKSIZE] = (int) SCRATCH[ti3(n_block, c, 0, C, BLOCKSIZE) : BLOCKSIZE];
        } // nc
    } // pragma offload
}

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