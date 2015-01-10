#Copyright (c) 2014, Oren Rippel and Ryan P. Adams
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cdef extern from "micmat.h":
    void tester()

    # ctypedef void *OmpLock

    # OmpLock *initialize_locks()

    void speed_tester(int N, float *INPUTS, float *OUTPUTS)

    float *allocate_host(int N)

    void zeros(int N, float *A)

    void fill_randn(int skip_num, int N, float *A, float mu, float sigma)

    void fill_uniform(int skip_num, int N, float *A)

    void fill_bernoulli(int skip_num, int N, int *A, float p)

    void fill_uniform_int(int skip_num, int N, int *A, int i_start, int i_end)

    void fill_geometric(int skip_num, int N, float *A, float p)

    void fill_nested(int N, int K, float *A, float *GEOMETRIC_SAMPLES)

    void apply_dropout(int N, int K, float *A, int *MASK)

    void fill_zeros(int N, float *A, int offloaded)

    void fill_zeros_int(int N, int *A, int offloaded)

    void fill_ones(int N, float *A, int offloaded)

    void fill_ones_int(int N, int *A, int offloaded)

    void fill_const(int N, float *A, float c, int offloaded)

    void fill_const_int(int N, int *A, int c, int offloaded)

    float *ones_mic(int N)

    void expo(int N, float *A)

    void clip(int N, float *A, float LOWER, float UPPER)

    void clip_low(int N, float *A, float LOWER)

    void flooro(int N, float *A)

    void sign(int N, float *A)

    float *equal(int N, float *A, float *B)

    float *elementwise_or(int N, float *A, float *B)

    float *leq(int N, float *A, float B)

    float *geq(int N, float *A, float B)

    float *greater(int N, float *A, float B)

    void greater_replace(int N, float *A, float B)

    void labels_to_vectors(int N, int K, float *A, float *targets)

    void lg(int N, float *A)

    void abso(int N, float *A)

    void sqrto(int N, float *A)

    float frac_nan(int N, float *A)

    void treat_nan(int N, float *A)

    void powo(int N, float *A, float b)

    float normo(int N, float *A)

    float *unalign_host(int N, float *A)

    float output_float(float *A)

    void copy(int N, float *A, float *B, int offloaded)

    void copy_int(int N, int *A, int *B, int offloaded)

    void shift_int(int N, int shift, int *A)

    void replace_host(int N, float *A, float *B)

    void replace_mic(int N, float *A, float *B)

    void replace_host_int(int N, int *A, int *B)

    void replace_mic_int(int N, int *A, int *B)

    void replace_partial_host(int N_A, int SHIFT_A, float *A, int N_B, int SHIFT_B, float *B)

    void replace_partial_mic(int N_A, int SHIFT_A, float *A, int N_B, int SHIFT_B, float *B)

    float *get_partial(int N, int SHIFT, float *A)

    void T(int ROWS, int COLS, float *A)

    void scale(int N, float *A, float c)

    void mult(int ROWS_A, int COLS_A, float *A, int ROWS_X, int COLS_X, float *X)

    void invert(int N, float *A)

    void divide(int ROWS_A, int COLS_A, float *A, int ROWS_X, int COLS_X, float *X)

    void print_slice(int ROWS, int COLS, float *A, int offloaded)

    void print_slice_mic(int ROWS, int COLS, float *A)

    float *slice_inds(int N, int *indices, float *A, int indices_offloaded, int offloaded)

    void slice_inds_replace(int N, int *indices, float *A, int indices_offloaded, int offloaded, float *scratch)

    float *slice_cols(int N, int *indices, int ROWS, int COLS, float *A, int indices_offloaded, int offloaded)

    float *slice_rows(int N, int *indices, int ROWS, int COLS, float *A, int indices_offloaded, int offloaded)

    void offload_mic(int N, float *A)

    void offload_mic_int(int N, int *A)

    void pull_mic(int N, float *A)

    void push_mic(int N, float *A)

    float *cast_float(int N, int *A, int offloaded)

    int *cast_int(int N, float *A, int offloaded)

    void cast_int_replace(int N, float *A, int *OUTPUT, int offloaded)

    void free_host(int N, float *A)

    void free_host_int(int N, int *A)

    void free_mic(int N, float *A)

    void free_mic_int(int N, int *A)

    # void update(int ROWS_A, int COLS_A, float *A, int ROWS_X, int COLS_X, float *X, float ALPHA)

    # void update(int a1, int a2, int a3, int a4, float *A, int b1, int b2, int b3, int b4, float *B, float ALPHA, int offloaded)
    void update(int a1, int a2, int a3, int a4, int a5, int a6, float *A, int b1, int b2, int b3, int b4, int b5, int b6, float *B, float ALPHA, int offloaded)

    void update_const(int N, float *A, float c)

    float *dot(int ROWS_A, int COLS_A, int T_A, float *A, int COLS_B, int T_B, float *B)

    void dot_replace(int ROWS_A, int COLS_A, int T_A, float *A, int ROWS_B, int COLS_B, int T_B, float *B, float BETA, float *C)

    float *dot_vec(int N, float *A, float *B)

    float *sum_axis(int ROWS_A, int COLS_A, float *A, int AXIS)

    float *sum_axis_replace(int ROWS_A, int COLS_A, float *A, int AXIS, float *S)
    # float *sum_axis(int a1, int a2, int a3, int a4, float *A, int AXIS, int offloaded)

    int *max_axis(int ROWS_A, int COLS_A, float *A, int AXIS)

    void max_axis_replace(int ROWS_A, int COLS_A, float *A, int AXIS, float *S)

    void index_global_to_local(int ROWS_A, int COLS_A, int *A, int AXIS)

    float sumo(int N, float *A)

    void convolve(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int tight)

    void horizontal_reflection(int N, int C, int H, int W, float *INPUTS, int *SHOULD_FLIP, float *scratch)
    void vertical_reflection(int N, int C, int H, int W, float *INPUTS, int *SHOULD_FLIP, float *scratch)

    void crop(int N, int C, int H, int W, int output_W, int output_H, float *INPUTS, int *CROP_INFO, float *OUTPUTS, int offloaded)

    void rgb_to_hsv(int N, int H, int W, float *INPUTS, float *OUTPUTS, float *SCRATCH, int offloaded)
    void hsv_to_rgb(int N, int H, int W, float *INPUTS, float *OUTPUTS, float *SCRATCH, int offloaded)

    void perturb_hue(int N, int C, int H, int W, float *INPUTS, float *PERTURBATIONS, int offloaded)
    void perturb_saturation(int N, int C, int H, int W, float *INPUTS, float *SCALE_PERTURBATIONS, float *SHIFT_PERTURBATIONS, int offloaded)
    void perturb_value(int N, int C, int H, int W, float *INPUTS, float *SCALE_PERTURBATIONS, float *SHIFT_PERTURBATIONS, int offloaded)

    void response_normalization(int N, int K, int H, int W, float *INPUTS, float *OUTPUTS, float ALPHA, float BETA, int local_radius)

    void response_normalization_gradient(int N, int K, int H, int W, float *INPUTS, float *OUTPUTS, float *D_INPUTS, float *D_OUTPUTS, float ALPHA, float BETA, int LOCAL_RADIUS)

    void permute_dimensions(int D1, int D2, int D3, int D4, int perm1, int perm2, int perm3, int perm4, float *TENSOR, float *SCRATCH)
    void permute_dimensions_int(int D1, int D2, int D3, int D4, int perm1, int perm2, int perm3, int perm4, int *TENSOR, float *SCRATCH)

    void interleave_block(int N, int C, int BLOCKSIZE, float *TENSOR, float *SCRATCH)
    void interleave_block_int(int N, int C, int BLOCKSIZE, int *TENSOR, float *SCRATCH)
    void uninterleave_block(int N, int C, int BLOCKSIZE, float *TENSOR, float *SCRATCH)
    void uninterleave_block_int(int N, int C, int BLOCKSIZE, int *TENSOR, float *SCRATCH)
    void interleave_for_gradient(int N, int C, int H, int W, int BLOCKSIZE, float *TENSOR, float *SCRATCH)
    void uninterleave_for_gradient(int N, int C, int H, int W, int BLOCKSIZE, float *TENSOR, float *SCRATCH)

    void transpose_replace(int N, int C, float *INPUT, float *OUTPUT)
    void transpose_replace_int(int N, int C, int *INPUT, float *OUTPUT)
    
    void convolution_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int *ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH)
    void convolution_layer2(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int *ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH)
    void convolution_layer3(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int *ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH)
    void convolution_layer4(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int *ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH)
    void convolution_layer5(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int *ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH)
    void convolution_layer6(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int *ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH)
    void convolution_layer7(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int *ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH)
    void convolution_layer8(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int *ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH)
    void convolution_layer9(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int *ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH)
    void convolution_layer10(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int *ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH)

    void get_argmaxs(int N, int C, int H, int W, float *INPUTS, float *OUTPUTS, int *ARGMAXS)

    void convolution_gradient_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH)
    void convolution_gradient_layer2(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH)
    void convolution_gradient_layer3(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH)
    void convolution_gradient_layer4(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH)
    void convolution_gradient_layer5(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH)
    void convolution_gradient_layer6(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH)
    void convolution_gradient_layer7(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH)
    void convolution_gradient_layer8(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH)
    void convolution_gradient_layer9(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH)
    void convolution_gradient_layer10(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH)



    void local_filtering_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int *ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH)

    int *convolve_and_pool_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int *ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH)

    int *convolve_and_pool_shadow_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int *ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH)

    int *convolve_argmaxs_fixed_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int *ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded)

    int *convolve_argmaxs_fixed_layer2(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int *ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded)
    
    int *convolve_and_pool_layer2(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int *ARGMAXS, int stride, int padding, int pooling_radius, int pooling_stride, int offloaded, float *SCRATCH)

    void convolve_gradient_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH)

    void convolve_gradient_shadow_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH)

    void convolve_gradient_layer2(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, int *ARGMAXS, float *D_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH)

    void local_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int stride, int padding, int offloaded, float *SCRATCH)

    void local_gradient_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, float *D_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH)

    void local_layer2(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int stride, int padding, int offloaded, float *SCRATCH)

    void local_gradient_layer2(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, int padding, float *FILTERS, float *D_OUTPUTS, float *D_INPUTS, float *D_FILTERS, float *SCRATCH)

    # int *convolve_and_pool_layer1(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int pool_radius, int stride, int *ARGMAXS, int argmaxs_fixed, int offloaded)

    # int *convolve_and_pool_layer2(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int pool_radius, int stride, int *ARGMAXS, int argmaxs_fixed, int offloaded)

    void fft(int N, int H, int W, int IS_FORWARD, float *SPATIAL, float *FREQUENCIES)

    void fft_full(int N, int H, int W, int IS_FORWARD, float *SPATIAL, float *FREQUENCIES)

    void wipe_out_irrelevant_entries(int N, int H, int W, float *INPUTS)

    void wipe_out_irrelevant_entries_full(int N, int H, int W, float *INPUTS)

    void cmult(int N, int K, int C, int HW, float *INPUTS, float *FILTERS, float *OUTPUTS)

    void cmult_gradient(int N, int K, int C, int HW, float *INPUTS, float *GRADIENT_INPUTS, float *FILTERS, float *GRADIENT_FILTERS, float *GRADIENT_OUTPUTS)

    void low_pass_filter(int N, int H, int W, int DIAMETER_H, int DIAMETER_W, float *INPUTS, float *OUTPUTS)
    void low_pass_filter_gradient(int N, int H, int W, int DIAMETER_H, int DIAMETER_W, float *INPUTS, float *OUTPUTS)

    void conjugate(int N, float *A)

    void fft_conjugate_symmetry_scaling(int N, int K, int H, int FORWARD, int W, float *A, float *scratch)

    void real_and_imaginary(int N, float *A, float *REAL, float *IMAGINARY)

    void amplitude_and_phase(int N, float *A, float *AMPLITUDE, float *PHASE)

    void gram_schmidt(int N, int C, float *A, float *SCRATCH)

    void check_mic_status()

    void ping_each_core()

    int main()

