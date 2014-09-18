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


def recompile_MICMat(N, K, c, H, W, X, Y, stride, padding, pooling_radius, pooling_stride):
    OUTPUT_PATH = '/global/homes/r/rippel/sota/output/default/'
    SPECIFIC_MICMAT_PATH = OUTPUT_PATH + 'micmat/'
    MICMAT_PATH = '/global/homes/r/rippel/sota/micmat/'
    
    print 'Modifying macros file. \n'
    output_H = (H + 2*padding - Y + 1)/stride
    output_W = (W + 2*padding - X + 1)/stride
    pooled_H = (output_H - pooling_radius + 1)/pooling_stride
    pooled_W = (output_W - pooling_radius + 1)/pooling_stride
    
    macros_file = open(SPECIFIC_MICMAT_PATH + 'generated_macros.h', 'w')
    contents = '#define C_const ' + `c` + '\n' + \
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

def main():

    time_only = True
    offload = True
      
    N = 512
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
    
    global C, stream, timer
    timer = Timer()

    recompile_MICMat(N, K, c, H, W, X, Y, stride, padding, pooling_radius, pooling_stride)
    import micmat_wrap as C

    stream = C.RandGen()
    
    scratch = C.Scratch((1e8, 1))
    scratch.offload_mic()
    scratch.fill_zeros()

    C.check_mic_status()

    test_convolution(time_only, offload, N, K, c, H, W, X, Y, stride, padding, pooling_radius, pooling_stride, scratch)
    

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


def test_convolution(time_only, offload, N, K, c, H, W, X, Y, stride, padding, pooling_radius, pooling_stride, scratch):
    output_H = (H + 2*padding - Y + 1)/stride
    output_W = (W + 2*padding - X + 1)/stride
    pooled_H = (output_H - pooling_radius + 1)/pooling_stride
    pooled_W = (output_W - pooling_radius + 1)/pooling_stride

    num_operations = N*K*c*output_H*output_W*X*Y*2
    num_operations_argmax = N*K*c*pooled_H*pooled_W*Y*X*2
    num_operations_gradient = N*K*c*pooled_H*pooled_W*Y*X*4

    inputs = C.MICMat((N, c, H, W))
    inputs.fill_zeros()
    
    filters = C.MICMat((K, c, Y, X))
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
    timer.tic()
    outputs.convolve_and_pool_replace(inputs, filters, argmaxs, stride, padding, pooling_radius, pooling_stride, 1, False, scratch)
    test_time = timer.toc()

    # print outputs

    print '\n \n Convolution time: %f seconds.' % test_time
    print 'Speed: %f Gflops.' % (num_operations/test_time*2e-9)

    timer.tic()
    outputs.convolve_and_pool_replace(inputs, filters, argmaxs, stride, padding, pooling_radius, pooling_stride, 1, True, scratch)
    fixed_time = timer.toc()

    # print outputs

    if not time_only:
        print 'Running convolution test. '
        pure_python = np.zeros((N, K, pooled_H, pooled_W)) - 1e10
        for n in range(N):
            for k in range(K):
                for h in range(H - Y + 1):
                    for w in range(W - X + 1):
                        pure_python[n, k, h/pooling_stride, w/pooling_stride] = max(np.sum(np.multiply(inputs.ndarray()[n, :, h:h+Y, w:w+X], filters.ndarray()[k, :, :, :])), pure_python[n, k, h/pooling_stride, w/pooling_stride])

        difference = np.mean(np.abs(outputs.ndarray() - pure_python))
        if difference < 1.e-6:
            print 'Convolution test passed. '

        else:
            print 'Convolution test failed with difference %f. ' % difference,
            print outputs
            print pure_python[0, 0, :, :]

    else:
        gradient_pooled_outputs = outputs.deepcopy()
        gradient_filters = filters.deepcopy()
        gradient_inputs = inputs.deepcopy()

        print 'Computing convolution gradient now.'
        timer.tic()
        gradient_filters.convolve_gradient(inputs, filters, argmaxs,
            gradient_pooled_outputs, gradient_inputs, stride, padding, pooling_radius, pooling_stride, 1, scratch)
        convolution_gradient_time = timer.toc()

        print '\n \n Time: %f seconds.' % convolution_gradient_time
        print 'Speed: %f Gflops.' % (num_operations_gradient/convolution_gradient_time*2e-9)


def test_local(time_only, offload, N, K, c, H, W, X, Y, stride, padding, scratch):
    output_H = (H + 2*padding - Y + 1)/stride
    output_W = (W + 2*padding - X + 1)/stride

    num_operations = N*K*c*output_H*output_W*X*Y*2
    num_operations_gradient = N*K*c*output_H*output_W*Y*X*4

    inputs = C.MICMat((N, c, H, W))
    inputs.fill_zeros()
    
    filters = C.MICMat((K, output_H, output_W, c, Y, X))
    filters.fill_zeros()
    outputs = C.MICMat((N, K, output_H, output_W))
    
    if offload:
        inputs.offload_mic()
        inputs.fill_randn(stream, 0., 1.)

        outputs.offload_mic()
        
        filters.offload_mic()
        filters.fill_randn(stream, 0., 1.)

    print 'Computing local now.'
    timer.tic()
    outputs.local_replace(inputs, filters, stride, padding, 1, scratch)
    test_time = timer.toc()

    # print outputs
    # outputs.fill_zeros()

    print '\n \n Local time: %f seconds.' % test_time
    print 'Speed: %f Gflops.' % (num_operations/test_time*2e-9)

    if not time_only:
        print 'Running convolution test. '
        # pure_python = np.zeros((N, K, pooled_H, pooled_W)) - 1e10
        # for n in range(N):
        #     for k in range(K):
        #         for h in range(H - Y + 1):
        #             for w in range(W - X + 1):
        #                 pure_python[n, k, h/pooling_stride, w/pooling_stride] = max(np.sum(np.multiply(inputs.ndarray()[n, :, h:h+Y, w:w+X], filters.ndarray()[k, :, :, :])), pure_python[n, k, h/pooling_stride, w/pooling_stride])

        # difference = np.mean(np.abs(outputs.ndarray() - pure_python))
        # if difference < 1.e-6:
        #     print 'Convolution test passed. '

        # else:
        #     print 'Convolution test failed with difference %f. ' % difference,
        #     print outputs
        #     print pure_python[0, 0, :, :]

    else:
        gradient_outputs = outputs.deepcopy()
        gradient_filters = filters.deepcopy()
        gradient_inputs = inputs.deepcopy()

        print 'Computing local gradient now.'
        timer.tic()
        gradient_filters.local_gradient(inputs, filters, gradient_outputs, gradient_inputs, stride, padding, 1, scratch)
        local_gradient_time = timer.toc()

        print '\n \n Time: %f seconds.' % local_gradient_time
        print 'Speed: %f Gflops.' % (num_operations_gradient/local_gradient_time*2e-9)


def test_update(time_only, offload):
    
    if time_only:
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

    if not time_only:
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
        print 'Speed: %f Gflops.' % (num_operations/test_time*2e-9)

def test_sum(time_only, offload):
    
    if time_only:
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

    if not time_only:
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
        print 'Speed: %f Gflops.' % (num_operations/test_time*2e-9)

stream = None
timer = None

if __name__ == '__main__':
    main()
