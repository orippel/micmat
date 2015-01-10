#Copyright (c) 2014, Oren Rippel and Ryan P. Adams
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


C = None
import numpy as np
import timeit
import time
from functools import partial
import os
from sys import *
import subprocess

def main():

    np.set_printoptions(precision = 4, suppress = True)

    examine_convolution = True
    examine_local = False

    time_and_dont_test = False
    time_and_dont_test_grad = False
    test_gradient = True
    offload = True
    
    if time_and_dont_test:
        if examine_convolution:
            N_block = 1
            K_block = 32
            C_block = 1

            N_block_grad = None
            C_block_grad = 16
            H_arg_block_grad = 1
            W_arg_block_grad = None # will set to be output_W
            Y_block_grad = 1

            N = 128 # faster if N is a multiple of 236 (stems from needing N/N_block*K/K_block to be divisible by 236)
            K = 64
            c = 64
            H = 13
            W = 13
            X = 5
            Y = 5
            stride = 1
            padding = 2
            pooling_radius = 3
            pooling_stride = 2

        elif examine_local:
            N_block = 16
            K_block = 4
            C_block = 4

            N = 128
            K = 64
            c = 64
            H = 13
            W = 13
            X = 5
            Y = 5
            stride = 1
            padding = 0
            pooling_radius = 1
            pooling_stride = 1
    
    else:
        N_block_grad = None
        C_block_grad = 16
        H_arg_block_grad = 1
        W_arg_block_grad = None # will set to be output_W
        Y_block_grad = 1

        N_block = 1
        K_block = 32
        C_block = 1
        N = 128
        K = 32
        c = 16
        H = 7
        W = 7
        X = 3
        Y = 3
        stride = 1
        padding = 2
        pooling_radius = 1
        pooling_stride = 1


    shadow = False
    K_preshadow = K
    if shadow:
        K *= 2
    
    global C, stream, timer
    timer = Timer()

    recompile_MICMat(N, K_preshadow, c, H, W, X, Y, stride, padding, pooling_radius, pooling_stride, N_block, K_block, C_block, N_block_grad, C_block_grad, H_arg_block_grad, W_arg_block_grad, Y_block_grad)
    import micmat_wrap as C

    stream = C.RandGen()
    
    scratch = C.Scratch((1e8, 1))
    scratch.offload_mic()
    scratch.fill_zeros()

    C.check_mic_status()

    # scratch.frac_nan()
    # test_fft(scratch)


    # inputs.shape = (1, np.product(shape))
    # print inputs
    # inputs.shape = shape

    # inputs.rgb_to_hsv(outputs, scratch)
    # outputs.shape = (1, np.product(shape))
    # print outputs
    # outputs.shape = shape

    # inputs_copy = inputs.deepcopy().fill_zeros()
    # outputs.hsv_to_rgb(inputs_copy, scratch)
    # inputs_copy.shape = (1, np.product(shape))
    # print inputs_copy
    # inputs_copy.shape = shape

    ########
    # shape = (2000, 3, 1, 1)
    # inputs = C.MICMat(shape).offload_mic().fill_zeros()
    # outputs = inputs.deepcopy()
    # inputs.fill_randn(stream, 180.0, 60.0)
    # inputs.clip_low(0.0)
    # inputs.scale(-1.).clip_low(-255.0).scale(-1.)
    # # print inputs
    # inputs.shape = (1, np.product(shape))
    # print inputs
    # inputs.shape = shape

    # inputs.rgb_to_hsv(outputs, scratch)
    # outputs.shape = (1, np.product(shape))
    # print outputs
    # outputs.shape = shape
    
    # inputs_copy = inputs.deepcopy().fill_zeros()
    # inputs_copy.clip_low(0.0)
    # inputs_copy.scale(-1.).clip_low(-255.0).scale(-1.)
    # outputs.hsv_to_rgb(inputs_copy, scratch)
    # print (inputs - inputs_copy).abs().mean()
    # inputs_copy.shape = (1, np.product(shape))
    # print inputs_copy
    # inputs_copy.shape = shape

    # inputs_np = inputs.ndarray()
    # inputs_copy_np = inputs_copy.ndarray()
    # diff = abs(inputs_np - inputs_copy_np).flatten()
    # arg = diff.argmax()
    # arg = (arg/3) * 3

    # print inputs_np.flatten()[arg:arg+3]
    # print outputs.ndarray().flatten()[arg:arg+3]
    # print inputs_copy_np.flatten()[arg:arg+3]

    # C.initialize_locks()
    if examine_convolution:
        test_convolution(time_and_dont_test, time_and_dont_test_grad, test_gradient, offload, N, K_preshadow, c, H, W, X, Y, stride, padding, pooling_radius, pooling_stride, scratch, shadow, N_block, K_block, C_block, N_block_grad, C_block_grad, H_arg_block_grad, W_arg_block_grad, Y_block_grad)

    if examine_local:
        test_local(time_and_dont_test, test_gradient, offload, N, K_preshadow, c, H, W, X, Y, stride, padding, pooling_radius, pooling_stride, scratch, shadow, N_block, K_block, C_block)


    # inputs = C.MICMat((N, K, H, W)).offload_mic().fill_zeros()
    # outputs = inputs.deepcopy()
    # d_inputs = inputs.deepcopy()
    # d_outputs = outputs.deepcopy()

    # scratch.reset(inputs)

    # alpha = 0.0001
    # beta = 0.75
    # local_radius = 4

    # timer.tic()
    # outputs.response_normalization(inputs, alpha, beta, local_radius)
    # test_time = timer.toc()

    # timer.tic()
    # d_inputs.response_normalization_gradient(inputs, outputs, d_inputs, alpha, beta, local_radius)
    # grad_time = timer.toc()

    # num_operations = N*K*H*W*(2*local_radius + 1)
    # print '\n \n Response normalization time: %f seconds.' % test_time
    # print 'Speed: %f Gflops.' % (num_operations/test_time*1e-9)

    # print '\n \n Response normalization gradient time: %f seconds.' % grad_time
    # print 'Speed: %f Gflops.' % (num_operations/grad_time*1e-9)
    

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def tic(self):
        self.start_time = time.time()

    def toc(self):
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time

        return self.total_time

    def elapsed(self):
        print 'Elapsed: %s.' % (time.time() - self.start_time)


def recompile_MICMat(N, K, c, H, W, X, Y, stride, padding, pooling_radius, pooling_stride, N_block, K_block, C_block, N_block_grad, C_block_grad, H_arg_block_grad, W_arg_block_grad, Y_block_grad):
    OUTPUT_PATH = '/global/homes/r/rippel/sota/output/default/'
    SPECIFIC_MICMAT_PATH = OUTPUT_PATH + 'micmat/'
    MICMAT_PATH = '/global/homes/r/rippel/sota/micmat/'
    # OUTPUT_PATH = '/project/projectdirs/mantissa/climate/dbn/micmat/'
    # SPECIFIC_MICMAT_PATH = OUTPUT_PATH + 'compilation/'
    # MICMAT_PATH = '/project/projectdirs/mantissa/climate/dbn/micmat/'
    
    print 'Modifying macros file. \n'
    output_H = (H + 2*padding - Y + 1)/stride
    output_W = (W + 2*padding - X + 1)/stride
    pooled_H = int(np.ceil((output_H - pooling_radius + 1.)/pooling_stride))
    pooled_W = int(np.ceil((output_W - pooling_radius + 1.)/pooling_stride))

    N_block_grad = N
    W_arg_block_grad = output_W
    
    macros_file = open(SPECIFIC_MICMAT_PATH + 'generated_macros.h', 'w')
    
    contents = '#define N_BLOCK ' + `N_block` + '\n' + \
            '#define K_BLOCK ' + `K_block` + '\n' + \
            '#define C_BLOCK ' + `C_block` + '\n\n'

    contents += '#define N_BLOCK_GRAD ' + `N_block_grad` + '\n' + \
            '#define C_BLOCK_GRAD ' + `C_block_grad` + '\n' + \
            '#define H_ARG_BLOCK_GRAD ' + `H_arg_block_grad` + '\n' + \
            '#define W_ARG_BLOCK_GRAD ' + `W_arg_block_grad` + '\n' + \
            '#define Y_BLOCK_GRAD ' + `Y_block_grad` + '\n\n'
            #define H_ARG_BLOCK 1
#define W_ARG_BLOCK output_W_const
#define Y_BLOCK 1

    contents += '#define C_const ' + `c` + '\n' + \
            '#define H_const ' + `H` + '\n' + \
            '#define W_const ' + `W` + '\n' + \
            '#define K_const ' + `K` + '\n' + \
            '#define stride_const ' + `stride` + '\n' + \
            '#define padding_const ' + `padding` + '\n' + \
            '#define pooling_radius_const ' + `pooling_radius` + '\n' + \
            '#define pooling_stride_const ' + `pooling_stride` + '\n' + \
            '#define Y_const ' + `Y` + '\n' + \
            '#define X_const ' + `X` + '\n' \
            '#define output_H_const ' + `output_H` + '\n' \
            '#define output_W_const ' + `output_W` + '\n' \
            '#define pooled_H_const ' + `pooled_H` + '\n' \
            '#define pooled_W_const ' + `pooled_W` + '\n'

    contents += '\n\n' + '#define C_constL ' + `c` + '\n' + \
            '#define H_constL ' + `H` + '\n' + \
            '#define W_constL ' + `W` + '\n' + \
            '#define K_constL ' + `K` + '\n' + \
            '#define stride_constL ' + `stride` + '\n' + \
            '#define padding_constL ' + `padding` + '\n' + \
            '#define Y_constL ' + `Y` + '\n' + \
            '#define X_constL ' + `X` + '\n' \
            '#define output_H_constL ' + `output_H` + '\n' \
            '#define output_W_constL ' + `output_W` + '\n'

    contents += '\n\n' + '#define C_const2 1\n' + \
            '#define H_const2 1' + '\n' + \
            '#define W_const2 1' + '\n' + \
            '#define K_const2 1' + '\n' + \
            '#define stride_const2 1' + '\n' + \
            '#define padding_const2 1' + '\n' + \
            '#define pooling_radius_const2 1' + '\n' + \
            '#define pooling_stride_const2 1' + '\n' + \
            '#define Y_const2 1' + '\n' + \
            '#define X_const2 1' + '\n' \
            '#define output_H_const2 1'  + '\n' \
            '#define output_W_const2 1'  + '\n' \
            '#define pooled_H_const2 1' + '\n' \
            '#define pooled_W_const2 1' + '\n'

    contents += '#define N_BLOCK_GRAD2 1' + '\n' + \
            '#define C_BLOCK_GRAD2 1' + '\n' + \
            '#define H_ARG_BLOCK_GRAD2 1' + '\n' + \
            '#define W_ARG_BLOCK_GRAD2 1' + '\n' + \
            '#define Y_BLOCK_GRAD2 1' + '\n\n'

    contents += '\n\n' + '#define C_const3 1\n' + \
            '#define H_const3 1' + '\n' + \
            '#define W_const3 1' + '\n' + \
            '#define K_const3 1' + '\n' + \
            '#define stride_const3 1' + '\n' + \
            '#define padding_const3 1' + '\n' + \
            '#define pooling_radius_const3 1' + '\n' + \
            '#define pooling_stride_const3 1' + '\n' + \
            '#define Y_const3 1' + '\n' + \
            '#define X_const3 1' + '\n' \
            '#define output_H_const3 1'  + '\n' \
            '#define output_W_const3 1'  + '\n' \
            '#define pooled_H_const3 1' + '\n' \
            '#define pooled_W_const3 1' + '\n'

    contents += '#define N_BLOCK_GRAD3 1' + '\n' + \
            '#define C_BLOCK_GRAD3 1' + '\n' + \
            '#define H_ARG_BLOCK_GRAD3 1' + '\n' + \
            '#define W_ARG_BLOCK_GRAD3 1' + '\n' + \
            '#define Y_BLOCK_GRAD3 1' + '\n\n'

    contents += '\n\n' + '#define C_const4 1\n' + \
            '#define H_const4 1' + '\n' + \
            '#define W_const4 1' + '\n' + \
            '#define K_const4 1' + '\n' + \
            '#define stride_const4 1' + '\n' + \
            '#define padding_const4 1' + '\n' + \
            '#define pooling_radius_const4 1' + '\n' + \
            '#define pooling_stride_const4 1' + '\n' + \
            '#define Y_const4 1' + '\n' + \
            '#define X_const4 1' + '\n' \
            '#define output_H_const4 1'  + '\n' \
            '#define output_W_const4 1'  + '\n' \
            '#define pooled_H_const4 1' + '\n' \
            '#define pooled_W_const4 1' + '\n'

    contents += '#define N_BLOCK_GRAD4 1' + '\n' + \
            '#define C_BLOCK_GRAD4 1' + '\n' + \
            '#define H_ARG_BLOCK_GRAD4 1' + '\n' + \
            '#define W_ARG_BLOCK_GRAD4 1' + '\n' + \
            '#define Y_BLOCK_GRAD4 1' + '\n\n'

    contents += '\n\n' + '#define C_const5 1\n' + \
            '#define H_const5 1' + '\n' + \
            '#define W_const5 1' + '\n' + \
            '#define K_const5 1' + '\n' + \
            '#define stride_const5 1' + '\n' + \
            '#define padding_const5 1' + '\n' + \
            '#define pooling_radius_const5 1' + '\n' + \
            '#define pooling_stride_const5 1' + '\n' + \
            '#define Y_const5 1' + '\n' + \
            '#define X_const5 1' + '\n' \
            '#define output_H_const5 1'  + '\n' \
            '#define output_W_const5 1'  + '\n' \
            '#define pooled_H_const5 1' + '\n' \
            '#define pooled_W_const5 1' + '\n'

    contents += '#define N_BLOCK_GRAD5 1' + '\n' + \
            '#define C_BLOCK_GRAD5 1' + '\n' + \
            '#define H_ARG_BLOCK_GRAD5 1' + '\n' + \
            '#define W_ARG_BLOCK_GRAD5 1' + '\n' + \
            '#define Y_BLOCK_GRAD5 1' + '\n\n'

    contents += '\n\n' + '#define C_const6 1\n' + \
            '#define H_const6 1' + '\n' + \
            '#define W_const6 1' + '\n' + \
            '#define K_const6 1' + '\n' + \
            '#define stride_const6 1' + '\n' + \
            '#define padding_const6 1' + '\n' + \
            '#define pooling_radius_const6 1' + '\n' + \
            '#define pooling_stride_const6 1' + '\n' + \
            '#define Y_const6 1' + '\n' + \
            '#define X_const6 1' + '\n' \
            '#define output_H_const6 1'  + '\n' \
            '#define output_W_const6 1'  + '\n' \
            '#define pooled_H_const6 1' + '\n' \
            '#define pooled_W_const6 1' + '\n'

    contents += '#define N_BLOCK_GRAD6 1' + '\n' + \
            '#define C_BLOCK_GRAD6 1' + '\n' + \
            '#define H_ARG_BLOCK_GRAD6 1' + '\n' + \
            '#define W_ARG_BLOCK_GRAD6 1' + '\n' + \
            '#define Y_BLOCK_GRAD6 1' + '\n\n'

    contents += '\n\n' + '#define C_const7 1\n' + \
            '#define H_const7 1' + '\n' + \
            '#define W_const7 1' + '\n' + \
            '#define K_const7 1' + '\n' + \
            '#define stride_const7 1' + '\n' + \
            '#define padding_const7 1' + '\n' + \
            '#define pooling_radius_const7 1' + '\n' + \
            '#define pooling_stride_const7 1' + '\n' + \
            '#define Y_const7 1' + '\n' + \
            '#define X_const7 1' + '\n' \
            '#define output_H_const7 1'  + '\n' \
            '#define output_W_const7 1'  + '\n' \
            '#define pooled_H_const7 1' + '\n' \
            '#define pooled_W_const7 1' + '\n'

    contents += '#define N_BLOCK_GRAD7 1' + '\n' + \
            '#define C_BLOCK_GRAD7 1' + '\n' + \
            '#define H_ARG_BLOCK_GRAD7 1' + '\n' + \
            '#define W_ARG_BLOCK_GRAD7 1' + '\n' + \
            '#define Y_BLOCK_GRAD7 1' + '\n\n'

    contents += '\n\n' + '#define C_const8 1\n' + \
            '#define H_const8 1' + '\n' + \
            '#define W_const8 1' + '\n' + \
            '#define K_const8 1' + '\n' + \
            '#define stride_const8 1' + '\n' + \
            '#define padding_const8 1' + '\n' + \
            '#define pooling_radius_const8 1' + '\n' + \
            '#define pooling_stride_const8 1' + '\n' + \
            '#define Y_const8 1' + '\n' + \
            '#define X_const8 1' + '\n' \
            '#define output_H_const8 1'  + '\n' \
            '#define output_W_const8 1'  + '\n' \
            '#define pooled_H_const8 1' + '\n' \
            '#define pooled_W_const8 1' + '\n'

    contents += '#define N_BLOCK_GRAD8 1' + '\n' + \
            '#define C_BLOCK_GRAD8 1' + '\n' + \
            '#define H_ARG_BLOCK_GRAD8 1' + '\n' + \
            '#define W_ARG_BLOCK_GRAD8 1' + '\n' + \
            '#define Y_BLOCK_GRAD8 1' + '\n\n'

    contents += '\n\n' + '#define C_const9 1\n' + \
            '#define H_const9 1' + '\n' + \
            '#define W_const9 1' + '\n' + \
            '#define K_const9 1' + '\n' + \
            '#define stride_const9 1' + '\n' + \
            '#define padding_const9 1' + '\n' + \
            '#define pooling_radius_const9 1' + '\n' + \
            '#define pooling_stride_const9 1' + '\n' + \
            '#define Y_const9 1' + '\n' + \
            '#define X_const9 1' + '\n' \
            '#define output_H_const9 1'  + '\n' \
            '#define output_W_const9 1'  + '\n' \
            '#define pooled_H_const9 1' + '\n' \
            '#define pooled_W_const9 1' + '\n'

    contents += '#define N_BLOCK_GRAD9 1' + '\n' + \
            '#define C_BLOCK_GRAD9 1' + '\n' + \
            '#define H_ARG_BLOCK_GRAD9 1' + '\n' + \
            '#define W_ARG_BLOCK_GRAD9 1' + '\n' + \
            '#define Y_BLOCK_GRAD9 1' + '\n\n'

    contents += '\n\n' + '#define C_const10 1\n' + \
            '#define H_const10 1' + '\n' + \
            '#define W_const10 1' + '\n' + \
            '#define K_const10 1' + '\n' + \
            '#define stride_const10 1' + '\n' + \
            '#define padding_const10 1' + '\n' + \
            '#define pooling_radius_const10 1' + '\n' + \
            '#define pooling_stride_const10 1' + '\n' + \
            '#define Y_const10 1' + '\n' + \
            '#define X_const10 1' + '\n' \
            '#define output_H_const10 1'  + '\n' \
            '#define output_W_const10 1'  + '\n' \
            '#define pooled_H_const10 1' + '\n' \
            '#define pooled_W_const10 1' + '\n'

    contents += '#define N_BLOCK_GRAD10 1' + '\n' + \
            '#define C_BLOCK_GRAD10 1' + '\n' + \
            '#define H_ARG_BLOCK_GRAD10 1' + '\n' + \
            '#define W_ARG_BLOCK_GRAD10 1' + '\n' + \
            '#define Y_BLOCK_GRAD10 1' + '\n\n'

    contents += '\n\n' + '#define C_constL2 ' + `c` + '\n' + \
            '#define H_constL2 ' + `H` + '\n' + \
            '#define W_constL2 ' + `W` + '\n' + \
            '#define K_constL2 ' + `K` + '\n' + \
            '#define stride_constL2 ' + `stride` + '\n' + \
            '#define padding_constL2 ' + `padding` + '\n' + \
            '#define Y_constL2 ' + `Y` + '\n' + \
            '#define X_constL2 ' + `X` + '\n' \
            '#define output_H_constL2 ' + `output_H` + '\n' \
            '#define output_W_constL2 ' + `output_W` + '\n'

    print contents
    macros_file.write(contents)
    macros_file.close()

    print 'Recompiling MICMat for the given set of hyperparameters at ' + SPECIFIC_MICMAT_PATH + '. \n'
    
    os.environ['SPECIFIC_MICMAT_PATH'] = SPECIFIC_MICMAT_PATH
    os.environ['MICMAT_PATH'] = MICMAT_PATH
    path.append(SPECIFIC_MICMAT_PATH)
    subprocess.call(MICMAT_PATH + 'micmat_build_specific.sh', cwd = MICMAT_PATH, env = os.environ)


def test_fft(scratch):
    H, W = 15, 15
    band_H, band_W = 6, 6
    inputs = C.MICMat((1, 1, band_H, band_W)).offload_mic().fill_randn(stream, 0., 1.)
    outputs = C.MICMat((1, 1, H, W)).offload_mic().fill_zeros()
    frequencies = C.MICMat((1, 1, H, 2*(W/2 + 1))).offload_mic().fill_zeros()
    pooled_frequencies = C.MICMat((1, 1, band_H, 2*(band_W/2 + 1))).offload_mic().fill_zeros()

    inputs.fft(pooled_frequencies)
    pooled_frequencies.wipe_out_irrelevant_entries()
    frequencies.low_pass_filter_gradient(pooled_frequencies, band_H, band_W)
    real, imaginary = pooled_frequencies.real_and_imaginary()
    print real
    print imaginary
    
    real, imaginary = frequencies.real_and_imaginary()
    print real
    print imaginary

    pooled_frequencies.ifft(outputs)
    assert 2 == 1

    # outputs = C.MICMat((1, 1, H, W)).offload_mic().fill_zeros()
    # inputs = C.MICMat((1, 1, H, 2*(W/2 + 1))).offload_mic().fill_randn(stream, 0., 1.)
    # inputs.wipe_out_irrelevant_entries()

    # real, imaginary = inputs.real_and_imaginary()
    # print real
    # print imaginary

    # inputs.ifft(outputs)

    # inputs_copy = outputs.fft(inputs.deepcopy()).wipe_out_irrelevant_entries()

    # print (inputs - inputs_copy).abs().mean()
    # real, imaginary = inputs_copy.real_and_imaginary()
    # print real
    # print imaginary


    # outputs = C.MICMat((1, 1, H, 2*W)).offload_mic().fill_zeros()
    # inputs = C.MICMat((1, 1, H, 2*W)).offload_mic().fill_randn(stream, 0., 1.)
    # inputs.update(inputs.deepcopy().conjugate())

    # inputs.fft_full(outputs)
    # outputs_ifft = inputs.ifft_full(outputs.deepcopy())
    # inputs_copy = outputs.ifft_full(inputs.deepcopy().fill_zeros())

    # real, imaginary = outputs.real_and_imaginary()
    # print real
    # print imaginary
    # real, imaginary = outputs_ifft.real_and_imaginary()
    # print real
    # print imaginary

    # print (outputs - outputs_ifft.conjugate()).abs().sum()

  

    outputs = C.MICMat((1, 1, H, W)).offload_mic().fill_zeros()
    outputs_full = C.MICMat((1, 1, H, 2*W)).offload_mic().fill_zeros()
    consts_outputs = C.MICMat((1, 1, H, W)).offload_mic().fill_randn(stream, 0., 1.)
    consts_outputs_full = C.MICMat((1, 1, H, 2*W)).offload_mic().fill_randn(stream, 0., 1.)
    consts_outputs_full.update(consts_outputs_full.deepcopy().conjugate()).scale(0.5)
    inputs = C.MICMat((1, 1, H, 2*(W/2 + 1))).offload_mic().fill_randn(stream, 0., 1.) #.wipe_out_irrelevant_entries()
    inputs_full = C.MICMat((1, 1, H, 2*W)).offload_mic().fill_randn(stream, 0., 1.)
    
    forward_pass = lambda inp: inp.ifft(outputs.deepcopy()).multiply(consts_outputs).sum()
    R = forward_pass(inputs)

    grad_R = consts_outputs.deepcopy()
    # grad_inputs = grad_R.ifft_full(inputs_full.deepcopy().fill_zeros())
    grad_inputs = grad_R.fft(inputs.deepcopy().fill_zeros())
    # grad_inputs.conjugate()
    grad_inputs.wipe_out_irrelevant_entries()
    
    real, imaginary = grad_inputs.real_and_imaginary()
    print real
    print imaginary
    
    # grad_inputs.conjugate()
    # grad_inputs.wipe_out_irrelevant_entries()
    grad_inputs.fft_conjugate_symmetry_scaling(W, scratch)

    real, imaginary = grad_inputs.real_and_imaginary()
    print real
    print imaginary

    
    print grad_inputs.deepcopy().pow(2.).sum()

    epsilon = 0.0001
    inputs_plus_d = inputs.deepcopy().update(grad_inputs, epsilon)
    inputs_minus_d = inputs.deepcopy().update(grad_inputs, -epsilon)
    R_plus = forward_pass(inputs_plus_d)
    R_minus = forward_pass(inputs_minus_d)

    R_diff = (R_plus - R_minus) / (2.*epsilon)
    print R_diff

    

    

    
    # ifft_d = d_inputs.ifft(outputs.deepcopy())
    # computed_gradient = ifft_d
    # # computed_gradient.scale(2.)

    # print computed_gradient.divide(fd_gradient)

    # outputs = C.MICMat((1, 1, 4, 4)).offload_mic().fill_zeros()
    # inputs = C.MICMat((1, 1, 4, 2*3)).offload_mic().fill_randn(stream, 0., 1.).wipe_out_irrelevant_entries()
    # d_inputs = C.MICMat((1, 1, 4, 2*3)).offload_mic().fill_randn(stream, 0., 0.00001).wipe_out_irrelevant_entries()
    # inputs_plus_d = inputs.deepcopy().update(d_inputs)
    # inputs_minus_d = inputs.deepcopy().update(d_inputs, -1.)

    # outputs_plus_d = inputs_plus_d.ifft(outputs.deepcopy())
    # outputs_minus_d = inputs_minus_d.ifft(outputs.deepcopy())
    # fd_gradient = (outputs_plus_d - outputs_minus_d).scale(0.5)
    
    # ifft_d = d_inputs.ifft(outputs.deepcopy())
    # computed_gradient = ifft_d
    # # computed_gradient.scale(2.)

    # print computed_gradient.divide(fd_gradient)


    # inputs.wipe_out_irrelevant_entries()
    # # inputs = outputs.fft()
    # timer.elapsed()
    # # outputs.fill_zeros()
    # extra.fill_randn(stream, 0., 1.)
    # timer.tic()
    # inputs.ifft(outputs)
    
    # outputs.fft(inputs)
    # inputs.ifft(outputs)
    # outputs.fft(inputs)
    # inputs.ifft(outputs)
    # timer.elapsed()


    # spatials = C.MICMat((128, 64, 32, 32)).offload_mic().fill_randn(stream, 0., 1.)
    # frequencies = C.MICMat((128, 64, 32, 2*17)).offload_mic().fill_randn(stream, 0., 1.)
    # spatials = C.MICMat((1, 1, 6, 6)).offload_mic().fill_randn(stream, 0., 1.)
    # frequencies = C.MICMat((1, 1, 6, 2*4)).offload_mic().fill_randn(stream, 0., 1.)
    # extra = C.MICMat((10, 3, 32, 17)).offload_mic()

    # spatials_original = spatials.deepcopy()

    # timer.tic()
    # spatials.fft(frequencies)
    # frequencies.wipe_out_irrelevant_entries()
    # # frequencies = spatials.fft()
    # timer.elapsed()
    # # spatials.fill_zeros()
    # extra.fill_randn(stream, 0., 1.)
    # timer.tic()
    # frequencies.ifft(spatials)
    
    # spatials.fft(frequencies)
    # frequencies.ifft(spatials)
    # spatials.fft(frequencies)
    # frequencies.ifft(spatials)
    # timer.elapsed()

    # print (spatials - spatials_original).abs().mean()

    # real, imaginary = frequencies.real_and_imaginary()
    # print real
    # print imaginary

    # frequencies.conjugate()
    # real, imaginary = frequencies.real_and_imaginary()
    # print real
    # print imaginary

def test_convolution(time_and_dont_test, time_and_dont_test_grad, test_gradient, offload, N, K, c, H, W, X, Y, stride, padding, pooling_radius, pooling_stride, scratch, shadow, N_block, K_block, C_block, N_block_grad, C_block_grad, H_arg_block_grad, W_arg_block_grad, Y_block_grad):
    output_H = (H + 2*padding - Y + 1)/stride
    output_W = (W + 2*padding - X + 1)/stride
    pooled_H = int(np.ceil((output_H - pooling_radius + 1.)/pooling_stride))
    pooled_W = int(np.ceil((output_W - pooling_radius + 1.)/pooling_stride))

    K_preshadow = K
    if shadow:
        K *= 2

    num_operations = N*K*c*output_H*output_W*X*Y*2
    num_operations_argmax = N*K*c*pooled_H*pooled_W*Y*X*2
    num_operations_gradient = N*K*c*pooled_H*pooled_W*Y*X*2

    inputs = C.MICMat((N, c, H, W))
    inputs.fill_zeros()
    
    filters = C.MICMat((K_preshadow, c, Y, X))
    filters.fill_zeros()
    outputs = C.MICMat((N, K, pooled_H, pooled_W))
    argmaxs = outputs.deepcopy().offload_mic().astype('int')
    
    if offload:
        inputs.offload_mic()
        inputs.fill_randn(stream, 0., 1.)

        outputs.offload_mic()
        
        filters.offload_mic()
        filters.fill_randn(stream, 0., 1.)

    print 'Computing convolution now.'
    filters_rotated = scratch.reset(filters.shape)
    filters_interleaved = scratch.append(filters.shape)
    inputs_rotated = scratch.append(inputs.shape)
    inputs_interleaved = scratch.append(inputs.shape)
    outputs_interleaved = scratch.append(outputs.shape)
    outputs_rotated = scratch.append(outputs.shape)
                
    scratch.update_end()

    outputs_interleaved.shape = outputs_interleaved.shape[1:] + (outputs_interleaved.shape[0],)
    outputs_interleaved.shape = (outputs_interleaved.shape[-1]/N_block,) + outputs_interleaved.shape[0:-1] + (N_block,)

    argmaxs.shape = argmaxs.shape[1:] + (argmaxs.shape[0],)
    argmaxs.shape = (argmaxs.shape[-1]/N_block,) + argmaxs.shape[0:-1] + (N_block,)

    inputs.rotate_dimensions('forward', inputs_rotated)
    inputs_rotated.interleave_block(N_block, inputs_interleaved)

    filters.rotate_dimensions('forward', filters_rotated)
    filters_rotated.interleave_block(K_block, filters_interleaved)    

    print 'Done data conversions.'
    timer.tic()
    outputs_interleaved.convolution(inputs_interleaved, filters_interleaved, argmaxs, stride, padding, pooling_radius, pooling_stride, 1, False, scratch.end)
    # outputs.convolve_and_pool_replace(inputs, filters, argmaxs, stride, padding, pooling_radius, pooling_stride, 1, False, scratch, shadow)
    test_time = timer.toc()
    print 'Done convolution.'

    outputs_interleaved.uninterleave_block(outputs_rotated)
    outputs_rotated.rotate_dimensions('backward', outputs)

    argmaxs.uninterleave_block(scratch.end)
    argmaxs.rotate_dimensions('backward', scratch.end)

    print '\n \nConvolution time: %f seconds.' % test_time
    print 'Speed: %f Gflops.' % (num_operations/test_time*1e-9)

    # timer.tic()
    # outputs.convolve_and_pool_replace(inputs, filters, argmaxs, stride, padding, pooling_radius, pooling_stride, 1, True, scratch, shadow)
    # fixed_time = timer.toc()

    # print outputs

    if not time_and_dont_test:
        print 'Running convolution test. '
        inputs_np = np.lib.pad(inputs_rotated.ndarray(), ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant', constant_values = (0, 0))

        filters_np = filters.ndarray()
        outputs_np = np.zeros((K, output_H, output_W, N)) - 1e10        
        pooled_outputs_np = np.zeros((K, pooled_H, pooled_W, N))

        for k in range(K):
            for h in range(output_H):
                for w in range(output_W):
                    for n in range(N):
                        outputs_np[k, h, w, n] = np.sum(np.multiply(inputs_np[:, h:h+Y, w:w+X, n], filters_np[k, :, :, :]))

        for k in range(K):
            for h in range(pooled_H):
                for w in range(pooled_W):
                    for n in range(N):
                        h_start = h*pooling_stride
                        h_end = min(h_start + pooling_radius, output_H)
                        w_start = w*pooling_stride
                        w_end = min(w_start + pooling_radius, output_W)
                        pooled_outputs_np[k, h, w, n] = np.amax(outputs_np[k, h_start:h_end, w_start:w_end, n])

        difference = np.mean(np.abs(outputs_rotated.ndarray() - pooled_outputs_np))
        if difference < 1.e-6:
            print 'Convolution test passed. '

        else:
            print 'Convolution test failed with difference %f. ' % difference
            # print (outputs.ndarray() - pooled_outputs_np)[0, 0, :, :]
            # print ''
            
            # for k in range(K):
            #     # for n in range(N):
            #         # print (k, n)
            #     print outputs.ndarray()[k, :, :, n]
            #     print ''
            #     print pooled_outputs_np[k, :, :, n]
            #     print ''
            #     print (outputs.ndarray() - pooled_outputs_np)[k, :, :, n]
            #     print ''

            # print outputs
            # print np.reshape(pooled_outputs_np[0, 0, :, :], pooled_outputs_np.shape[2:])

    if test_gradient:
        print '\n\n\n'
        print '='*20
        print 'Computing convolution gradient now.' 
        gradient_filters = filters.deepcopy().fill_zeros()
        gradient_inputs = inputs.deepcopy().fill_zeros()
        gradient_outputs = outputs.deepcopy().fill_randn(stream, 0., 1.)

        # filters_rotated = scratch.reset(filters.shape)
        # filters_interleaved = scratch.append(filters.shape)
        # inputs_rotated = scratch.append(inputs.shape)
        # inputs_interleaved = scratch.append(inputs.shape)
        # outputs_interleaved = scratch.append(outputs.shape)
        # outputs_rotated = scratch.append(outputs.shape)
                    
        # scratch.update_end()

        # outputs_interleaved.shape = outputs_interleaved.shape[1:] + (outputs_interleaved.shape[0],)
        # outputs_interleaved.shape = (outputs_interleaved.shape[-1]/N_block,) + outputs_interleaved.shape[0:-1] + (N_block,)

        # inputs.rotate_dimensions('forward', inputs_rotated)
        # inputs_rotated.interleave_block(N_block, inputs_interleaved)

        # filters.rotate_dimensions('forward', filters_rotated)
        # filters_rotated.interleave_block(K_block, filters_interleaved)    

        # timer.tic()
        # # Satish, the routine you're testing goes here
        # gradient_filters.convolution_gradient(inputs, filters, argmaxs,
        #     gradient_pooled_outputs, gradient_inputs, stride, padding, pooling_radius, pooling_stride, 1, scratch)
        # convolution_gradient_time = timer.toc()

        # print '\n \n Time: %f seconds.' % convolution_gradient_time
        # print 'Speed: %f Gflops.' % (num_operations_gradient/convolution_gradient_time*1e-9)

        if not time_and_dont_test_grad:
            # compute gradient as already validated
            inputs_interleaved = scratch.reset(inputs.shape)
            filters_permuted = scratch.append(filters.shape)
            filters_interleaved = scratch.append(filters.shape)
            gradient_permuted = scratch.append(gradient_filters.shape)
            gradient_interleaved = scratch.append(gradient_filters.shape, fill_zeros = True)
            gradient_inputs_interleaved = scratch.append(gradient_inputs.shape, fill_zeros = True)
            gradient_outputs_interleaved = scratch.append(gradient_outputs.shape)

            scratch.update_end()

            inputs.interleave_for_gradient(C_block_grad, inputs_interleaved)

            filters.permute_dimensions((2, 0, 3, 1), filters_permuted)
            filters_permuted.interleave_block(C_block_grad, filters_interleaved)

            gradient_outputs.permute_dimensions((0, 2, 3, 1), gradient_outputs_interleaved) 

            N, c, H, W = gradient_inputs_interleaved.shape    
            gradient_inputs_interleaved.shape = (N, c/C_block_grad, H, W, C_block_grad)
            
            K, c, Y, X = gradient_filters.shape
            gradient_interleaved.shape = (Y, K, X, c)
            gradient_interleaved.shape = (gradient_interleaved.shape[-1]/C_block_grad,) + gradient_interleaved.shape[0:-1] + (C_block_grad,)

            argmaxs.permute_dimensions((0, 2, 3, 1), scratch.end)

            gradient_interleaved.convolution_gradient(inputs_interleaved, filters_interleaved, argmaxs,
                gradient_outputs_interleaved, gradient_inputs_interleaved, stride, padding, 
                pooling_radius, pooling_stride, 
                1, scratch.end)

            gradient_interleaved.uninterleave_block(gradient_permuted)
            gradient_permuted.permute_dimensions((1, 3, 0, 2), gradient_filters)

            difference = np.mean(np.abs(gradient_filters.ndarray() - gradient_filters_tested.ndarray()))

            print '\n\n'
            if difference < 1.e-4:
                print 'Gradient test passed. '
            else:
                print 'Gradient test failed. '

# def test_convolution(time_and_dont_test, time_and_dont_test_grad, test_gradient, offload, N, K, c, H, W, X, Y, stride, padding, pooling_radius, pooling_stride, scratch, shadow, N_block, K_block, C_block, N_block_grad, C_block_grad, H_arg_block_grad, W_arg_block_grad, Y_block_grad):
#     output_H = (H + 2*padding - Y + 1)/stride
#     output_W = (W + 2*padding - X + 1)/stride
#     pooled_H = int(np.ceil((output_H - pooling_radius + 1.)/pooling_stride))
#     pooled_W = int(np.ceil((output_W - pooling_radius + 1.)/pooling_stride))

#     K_preshadow = K
#     if shadow:
#         K *= 2

#     num_operations = N*K*c*output_H*output_W*X*Y*2
#     num_operations_argmax = N*K*c*pooled_H*pooled_W*Y*X*2
#     num_operations_gradient = N*K*c*pooled_H*pooled_W*Y*X*2

#     inputs = C.MICMat((c, H, W, N))
#     inputs.fill_zeros()
    
#     filters = C.MICMat((K_preshadow, c, Y, X))
#     filters.fill_zeros()
#     outputs = C.MICMat((K, pooled_H, pooled_W, N))
#     argmaxs = outputs.deepcopy().offload_mic().astype('int')
    
#     if offload:
#         inputs.offload_mic()
#         inputs.fill_randn(stream, 0., 1.)

#         outputs.offload_mic()
        
#         filters.offload_mic()
#         filters.fill_randn(stream, 0., 1.)

#     print 'Computing convolution now.'

#     timer.tic()
#     inputs.interleave_block(N_block, scratch)
#     input_interleave_time = timer.toc()
#     print 'Done input interleaving.'

#     timer.tic()
#     outputs.interleave_block(N_block, scratch)
#     output_interleave_time = timer.toc()
#     print 'Done output interleaving.'

#     timer.tic()
#     argmaxs.interleave_block(N_block, scratch)
#     argmax_interleave_time = timer.toc()
#     print 'Done argmax interleaving.'
    
#     timer.tic()
#     filters.rotate_dimensions('forward', scratch)
#     rotation_time = timer.toc()
#     print 'Done dimension rotation.'

#     timer.tic()
#     filters.interleave_block(K_block, scratch)
#     filter_interleave_time = timer.toc()
#     print 'Done filter interleaving.'




#     print 'Done data conversions.'
#     timer.tic()
#     outputs.convolution(inputs, filters, argmaxs, stride, padding, pooling_radius, pooling_stride, 1, False, scratch)
#     # outputs.convolve_and_pool_replace(inputs, filters, argmaxs, stride, padding, pooling_radius, pooling_stride, 1, False, scratch, shadow)
#     test_time = timer.toc()
#     print 'Done convolution.'
#     timer.tic()
#     inputs.uninterleave_block(scratch)
#     input_uninterleave_time = timer.toc()

#     timer.tic()
#     outputs.uninterleave_block(scratch)
#     output_uninterleave_time = timer.toc()

#     timer.tic()
#     argmaxs.uninterleave_block(scratch)
#     argmax_uninterleave_time = timer.toc()
    
#     timer.tic()
#     filters.uninterleave_block(scratch)
#     filter_uninterleave_time = timer.toc()

#     timer.tic()
#     filters.rotate_dimensions('backward', scratch)
#     rotation_backward_time = timer.toc()
#     # print outputs

#     print 'Input interleaving time: %f seconds.' % input_interleave_time
#     print 'Output interleaving time: %f seconds.' % output_interleave_time
#     print 'Argmax interleaving time: %f seconds.' % argmax_interleave_time
#     print 'Filter interleaving time: %f seconds.\n' % filter_interleave_time

#     print 'Input un-interleaving time: %f seconds.' % input_uninterleave_time
#     print 'Output un-interleaving time: %f seconds.' % output_uninterleave_time
#     print 'Argmax un-interleaving time: %f seconds.' % argmax_uninterleave_time
#     print 'Filter un-interleaving time: %f seconds.\n' % filter_uninterleave_time

#     print 'Rotation time: %f seconds.' % rotation_time
#     print 'Rotation backward time: %f seconds.' % rotation_backward_time
#     print '\n \nConvolution time: %f seconds.' % test_time
#     print 'Speed: %f Gflops.' % (num_operations/test_time*1e-9)

#     # timer.tic()
#     # outputs.convolve_and_pool_replace(inputs, filters, argmaxs, stride, padding, pooling_radius, pooling_stride, 1, True, scratch, shadow)
#     # fixed_time = timer.toc()

#     # print outputs

#     if not time_and_dont_test:
#         print 'Running convolution test. '
#         inputs_np = np.lib.pad(inputs.ndarray(), ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant', constant_values = (0, 0))

#         filters_np = filters.ndarray()
#         outputs_np = np.zeros((K, output_H, output_W, N)) - 1e10        
#         pooled_outputs_np = np.zeros((K, pooled_H, pooled_W, N))

#         for k in range(K):
#             for h in range(output_H):
#                 for w in range(output_W):
#                     for n in range(N):
#                         outputs_np[k, h, w, n] = np.sum(np.multiply(inputs_np[:, h:h+Y, w:w+X, n], filters_np[k, :, :, :]))

#         for k in range(K):
#             for h in range(pooled_H):
#                 for w in range(pooled_W):
#                     for n in range(N):
#                         h_start = h*pooling_stride
#                         h_end = min(h_start + pooling_radius, output_H)
#                         w_start = w*pooling_stride
#                         w_end = min(w_start + pooling_radius, output_W)
#                         pooled_outputs_np[k, h, w, n] = np.amax(outputs_np[k, h_start:h_end, w_start:w_end, n])

#         difference = np.mean(np.abs(outputs.ndarray() - pooled_outputs_np))
#         if difference < 1.e-6:
#             print 'Convolution test passed. '

#         else:
#             print 'Convolution test failed with difference %f. ' % difference
#             print (outputs.ndarray() - pooled_outputs_np)[0, 0, :, :]
#             print ''
            
#             for k in range(K):
#                 # for n in range(N):
#                     # print (k, n)
#                 print outputs.ndarray()[k, :, :, n]
#                 print ''
#                 print pooled_outputs_np[k, :, :, n]
#                 print ''
#                 print (outputs.ndarray() - pooled_outputs_np)[k, :, :, n]
#                 print ''

#             print outputs
#             print np.reshape(pooled_outputs_np[0, 0, :, :], pooled_outputs_np.shape[2:])

#     if test_gradient:
#         print '\n\n\n'
#         print '='*20
#         print 'Computing convolution gradient now.' 
#         timer.tic()
#         inputs.rotate_dimensions('backward', scratch)
#         time = timer.toc()
#         print 'Gradient input rotation time: %f seconds.' % time            

#         timer.tic()
#         outputs.rotate_dimensions('backward', scratch)
#         time = timer.toc()
#         print 'Gradient output rotation time: %f seconds.' % time

#         timer.tic()
#         argmaxs.rotate_dimensions('backward', scratch)
#         time = timer.toc()
#         print 'Gradient argmax rotation time: %f seconds.' % time

#         timer.tic()
#         inputs.interleave_for_gradient(C_block_grad, scratch)
#         time = timer.toc()
#         print 'Gradient input interleaving time: %f seconds.' % time
#         assert Y_block_grad == 1

#         timer.tic()
#         filters.permute_dimensions((2, 0, 3, 1), scratch)
#         time = timer.toc()
#         print 'Gradient filter permutation time: %f seconds.' % time

#         # filters shape [C/C_BLOCK, Y, K, X, C_BLOCK]
#         timer.tic()
#         filters.interleave_block(C_block_grad, scratch)
#         time = timer.toc()
#         print 'Gradient filter interleaving time: %f seconds.' % time

#         timer.tic()
#         argmaxs.permute_dimensions((0, 2, 3, 1), scratch)
#         time = timer.toc()
#         print 'Gradient argmax permutation time: %f seconds.' % time

#         timer.tic()
#         outputs.permute_dimensions((0, 2, 3, 1), scratch) # d_pooled_outputs
#         time = timer.toc()
#         print 'Gradient output permutation time: %f seconds.' % time

#         gradient_pooled_outputs = outputs.deepcopy()
#         gradient_filters = filters.deepcopy()
#         gradient_inputs = inputs.deepcopy()

#         timer.tic()
#         gradient_filters.convolution_gradient(inputs, filters, argmaxs,
#             gradient_pooled_outputs, gradient_inputs, stride, padding, pooling_radius, pooling_stride, 1, scratch)
#         convolution_gradient_time = timer.toc()

#         print '\n \n Time: %f seconds.' % convolution_gradient_time
#         print 'Speed: %f Gflops.' % (num_operations_gradient/convolution_gradient_time*1e-9)

#         if not time_and_dont_test_grad:
#             # undo incorrect transformation
#             gradient_filters.uninterleave_block(scratch)
#             gradient_filters.permute_dimensions((1, 3, 0, 2), scratch)

#             gradient_filters_tested = gradient_filters.deepcopy()
            
#             #do correct transformation
#             gradient_filters.permute_dimensions((0, 2, 3, 1), scratch)
        
#             gradient_filters.convolution_gradient(inputs, filters, argmaxs,
#                 gradient_pooled_outputs, gradient_inputs, stride, padding, pooling_radius, pooling_stride, 1, scratch)
        
#             gradient_filters.permute_dimensions((0, 3, 1, 2), scratch)

#             difference = np.mean(np.abs(gradient_filters.ndarray() - gradient_filters_tested.ndarray()))

#             print '\n\n'
#             if difference < 1.e-6:
#                 print 'Gradient test passed. '
#             else:
#                 print 'Gradient test failed. '


def test_local(time_and_dont_test, test_gradient, offload, N, K, c, H, W, X, Y, stride, padding, pooling_radius, pooling_stride, scratch, shadow, N_block, K_block, C_block):
    output_H = (H + 2*padding - Y + 1)/stride
    output_W = (W + 2*padding - X + 1)/stride
    pooled_H = int(np.ceil((output_H - pooling_radius + 1.)/pooling_stride))
    pooled_W = int(np.ceil((output_W - pooling_radius + 1.)/pooling_stride))

    K_preshadow = K
    if shadow:
        K *= 2

    num_operations = N*K*c*output_H*output_W*X*Y*2
    num_operations_argmax = N*K*c*pooled_H*pooled_W*Y*X*2
    num_operations_gradient = N*K*c*pooled_H*pooled_W*Y*X*2

    inputs = C.MICMat((c, H, W, N))
    inputs.fill_zeros()
    
    filters = C.MICMat((K_preshadow, output_H, output_W, c, Y, X))
    filters.fill_zeros()
    outputs = C.MICMat((K, pooled_H, pooled_W, N))
    argmaxs = outputs.deepcopy().offload_mic().astype('int')
    
    if offload:
        inputs.offload_mic()
        inputs.fill_randn(stream, 0., 1.)

        outputs.offload_mic()
        
        filters.offload_mic()
        filters.fill_randn(stream, 0., 1.)

    print 'Computing local filtering now.'

    inputs.interleave_block(N_block, scratch)

    filters.rotate_dimensions('forward', scratch)
    filters.interleave_block(K_block, scratch)

    timer.tic()
    outputs.local_filtering(inputs, filters, argmaxs, stride, padding, pooling_radius, pooling_stride, 1, False, scratch)
    test_time = timer.toc()

    inputs.uninterleave_block(scratch)

    outputs.uninterleave_block(scratch)

    argmaxs.uninterleave_block(scratch)

    filters.uninterleave_block(scratch)
    filters.rotate_dimensions('backward', scratch)
    
    print '\n \n Local filtering time: %f seconds.' % test_time
    print 'Speed: %f Gflops.' % (num_operations/test_time*1e-9)

    # timer.tic()
    # outputs.convolve_and_pool_replace(inputs, filters, argmaxs, stride, padding, pooling_radius, pooling_stride, 1, True, scratch, shadow)
    # fixed_time = timer.toc()

    # print outputs

    if not time_and_dont_test:
        print 'Running local filtering test. '
        inputs_np = np.lib.pad(inputs.ndarray(), ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant', constant_values = (0, 0))

        filters_np = filters.ndarray()
        outputs_np = np.zeros((K, output_H, output_W, N)) - 1e10        
        pooled_outputs_np = np.zeros((K, pooled_H, pooled_W, N))

        for k in range(K):
            for h in range(output_H):
                for w in range(output_W):
                    for n in range(N):
                        outputs_np[k, h, w, n] = np.sum(np.multiply(inputs_np[:, h:h+Y, w:w+X, n], filters_np[k, h, w, :, :, :]))

        for k in range(K):
            for h in range(pooled_H):
                for w in range(pooled_W):
                    for n in range(N):
                        h_start = h*pooling_stride
                        h_end = min(h_start + pooling_radius, output_H)
                        w_start = w*pooling_stride
                        w_end = min(w_start + pooling_radius, output_W)
                        pooled_outputs_np[k, h, w, n] = np.amax(outputs_np[k, h_start:h_end, w_start:w_end, n])

        difference = np.mean(np.abs(outputs.ndarray() - pooled_outputs_np))
        if difference < 1.e-6:
            print 'Local filtering test passed. '

        else:
            print 'Local filtering test failed with difference %f. ' % difference
            print (outputs.ndarray() - pooled_outputs_np)[0, 0, :, :]
            print ''
            
            for k in range(K):
                # for n in range(N):
                    # print (k, n)
                print outputs.ndarray()[k, :, :, n]
                print ''
                print pooled_outputs_np[k, :, :, n]
                print ''
                print (outputs.ndarray() - pooled_outputs_np)[k, :, :, n]
                print ''

            print outputs
            print np.reshape(pooled_outputs_np[0, 0, :, :], pooled_outputs_np.shape[2:])

    else:
        if test_gradient:
            gradient_pooled_outputs = outputs.deepcopy()
            gradient_filters = filters.deepcopy()
            gradient_inputs = inputs.deepcopy()

            print 'Computing local filtering gradient now.'
            timer.tic()
            gradient_filters.convolution_gradient(inputs, filters, argmaxs,
                gradient_pooled_outputs, gradient_inputs, stride, padding, pooling_radius, pooling_stride, 1, scratch)
            convolution_gradient_time = timer.toc()

            print '\n \n Time: %f seconds.' % convolution_gradient_time
            print 'Speed: %f Gflops.' % (num_operations_gradient/convolution_gradient_time*1e-9)


# def test_local(time_and_dont_test, offload, N, K, c, H, W, X, Y, stride, padding, scratch):
#     output_H = (H + 2*padding - Y + 1)/stride
#     output_W = (W + 2*padding - X + 1)/stride

#     num_operations = N*K*c*output_H*output_W*X*Y*2
#     num_operations_gradient = N*K*c*output_H*output_W*Y*X*4

#     inputs = C.MICMat((N, c, H, W))
#     inputs.fill_zeros()
    
#     filters = C.MICMat((K, output_H, output_W, c, Y, X))
#     filters.fill_zeros()
#     outputs = C.MICMat((N, K, output_H, output_W))
    
#     if offload:
#         inputs.offload_mic()
#         inputs.fill_randn(stream, 0., 1.)

#         outputs.offload_mic()
        
#         filters.offload_mic()
#         filters.fill_randn(stream, 0., 1.)

#     print 'Computing local now.'
#     timer.tic()
#     outputs.local_replace(inputs, filters, stride, padding, 1, scratch)
#     test_time = timer.toc()

#     # print outputs
#     # outputs.fill_zeros()

#     print '\n \n Local time: %f seconds.' % test_time
#     print 'Speed: %f Gflops.' % (num_operations/test_time*1e-9)

#     if not time_and_dont_test:
#         print 'Running convolution test. '
#         # pure_python = np.zeros((N, K, pooled_H, pooled_W)) - 1e10
#         # for n in range(N):
#         #     for k in range(K):
#         #         for h in range(H - Y + 1):
#         #             for w in range(W - X + 1):
#         #                 pure_python[n, k, h/pooling_stride, w/pooling_stride] = max(np.sum(np.multiply(inputs.ndarray()[n, :, h:h+Y, w:w+X], filters.ndarray()[k, :, :, :])), pure_python[n, k, h/pooling_stride, w/pooling_stride])

#         # difference = np.mean(np.abs(outputs.ndarray() - pure_python))
#         # if difference < 1.e-6:
#         #     print 'Convolution test passed. '

#         # else:
#         #     print 'Convolution test failed with difference %f. ' % difference,
#         #     print outputs
#         #     print pure_python[0, 0, :, :]

#     else:
#         gradient_outputs = outputs.deepcopy()
#         gradient_filters = filters.deepcopy()
#         gradient_inputs = inputs.deepcopy()

#         print 'Computing local gradient now.'
#         timer.tic()
#         gradient_filters.local_gradient(inputs, filters, gradient_outputs, gradient_inputs, stride, padding, 1, scratch)
#         local_gradient_time = timer.toc()

#         print '\n \n Time: %f seconds.' % local_gradient_time
#         print 'Speed: %f Gflops.' % (num_operations_gradient/local_gradient_time*1e-9)


def test_update(time_and_dont_test, offload):
    
    if time_and_dont_test:
        shape = (30, 20, 100, 100)
    else:
        shape = (4, 3, 6, 9)

    A = C.MICMat(shape)
    B = C.MICMat((1, 1) + A.shape[2:])
    num_operations = 2*A.size

    if offload:
        A.offload_mic()
        A.fill_randn(stream, 0., 1.)

        B.offload_mic()
        B.fill_randn(stream, 0., 1.)
    
    output = A.deepcopy()
    print 'Computing update now.'
    timer.tic()
    output.update(B, 0.3)
    test_time = timer.toc()

    if not time_and_dont_test:
        pure_python = A.ndarray() + 0.3*B.ndarray()
        
        difference = np.mean(np.abs(output.ndarray() - pure_python))
        if difference < 1.e-6:
            print 'Update test passed. ',

        else:
            print 'Update test failed with difference %f. ' % difference,
            print A
            print pure_python[0, 0, :, :]

    else:
        print 'Time: %f seconds.' % test_time
        print 'Speed: %f Gflops.' % (num_operations/test_time*1e-9)

def test_sum(time_and_dont_test, offload):
    
    if time_and_dont_test:
        shape = (30, 20, 100, 100)
    else:
        shape = (4, 3, 6, 9)

    A = C.MICMat(shape)
    num_operations = A.size

    if offload:
        A.offload_mic()
        A.fill_randn(stream, 0., 1.)
    
    print 'Computing sum now.'
    timer.tic()
    S3 = A.sum(2) 
    test_time = timer.toc()

    S = A.sum()

    if not time_and_dont_test:
        pure_python = A.ndarray().sum()
        pure_python3 = A.ndarray().sum(axis = 3, keepdims = True)
        
        difference = np.mean(np.abs(S.float() - pure_python) + np.mean(np.abs(pure_python3 - S3.ndarray())))
        if difference < 1.e-4:
            print 'Update test passed. ',

        else:
            print 'Update test failed with difference %f. ' % difference,
            print S
            print pure_python
            print S3
            print pure_python3[0, 0, :, :]

    else:
        print 'Time: %f seconds.' % test_time
        print 'Speed: %f Gflops.' % (num_operations/test_time*1e-9)

stream = None
timer = None

if __name__ == '__main__':
    main()
