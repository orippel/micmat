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

    time_and_dont_test = True
    time_and_dont_test_grad = False
    test_gradient = True
    offload = True
    
    if time_and_dont_test:
        if examine_convolution:
            N_block = 16
            K_block = 4
            C_block = 1

            N_block_grad = None
            C_block_grad = 3
            H_arg_block_grad = 1
            W_arg_block_grad = None # will set to be output_W
            Y_block_grad = 1

            N = 128 # faster if N is a multiple of 236 (stems from needing N/N_block*K/K_block to be divisible by 236)
            K = 64
            c = 3
            H = 13
            W = 13
            X = 5
            Y = 5
            stride = 1
            padding = 0
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

        N_block = 16
        K_block = 4
        C_block = 1
        N = 128
        K = 16
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

    # A = C.MICMat((N, K, H, W)).offload_mic().fill_randn(stream, 0., 1.)

    # B = A.deepcopy()
    # B.permute_dimensions((0, 2, 3, 1), scratch)
    
    # A_np = A.ndarray()
    # B_np = np.transpose(A_np, (0, 2, 3, 1))

    # print np.abs(B_np - B.ndarray()).sum()

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

    inputs = C.MICMat((c, H, W, N))
    inputs.fill_zeros()
    
    filters = C.MICMat((K_preshadow, c, Y, X))
    filters.fill_zeros()
    outputs = C.MICMat((K, pooled_H, pooled_W, N))
    argmaxs = outputs.deepcopy().offload_mic().astype('int')
    
    if offload:
        inputs.offload_mic()
        inputs.fill_randn(stream, 0., 1.)

        outputs.offload_mic()
        
        filters.offload_mic()
        filters.fill_randn(stream, 0., 1.)

    print 'Computing convolution now.'

    timer.tic()
    inputs.interleave_block(N_block, scratch)
    input_interleave_time = timer.toc()

    timer.tic()
    outputs.interleave_block(N_block, scratch)
    output_interleave_time = timer.toc()

    timer.tic()
    argmaxs.interleave_block(N_block, scratch)
    argmax_interleave_time = timer.toc()
    
    timer.tic()
    filters.rotate_dimensions('forward', scratch)
    rotation_time = timer.toc()

    timer.tic()
    filters.interleave_block(K_block, scratch)
    filter_interleave_time = timer.toc()




    
    timer.tic()
    outputs.convolution(inputs, filters, argmaxs, stride, padding, pooling_radius, pooling_stride, 1, False, scratch)
    # outputs.convolve_and_pool_replace(inputs, filters, argmaxs, stride, padding, pooling_radius, pooling_stride, 1, False, scratch, shadow)
    test_time = timer.toc()

    timer.tic()
    inputs.uninterleave_block(scratch)
    input_uninterleave_time = timer.toc()

    timer.tic()
    outputs.uninterleave_block(scratch)
    output_uninterleave_time = timer.toc()

    timer.tic()
    argmaxs.uninterleave_block(scratch)
    argmax_uninterleave_time = timer.toc()
    
    timer.tic()
    filters.uninterleave_block(scratch)
    filter_uninterleave_time = timer.toc()

    timer.tic()
    filters.rotate_dimensions('backward', scratch)
    rotation_backward_time = timer.toc()
    # print outputs

    print 'Input interleaving time: %f seconds.' % input_interleave_time
    print 'Output interleaving time: %f seconds.' % output_interleave_time
    print 'Argmax interleaving time: %f seconds.' % argmax_interleave_time
    print 'Filter interleaving time: %f seconds.\n' % filter_interleave_time

    print 'Input un-interleaving time: %f seconds.' % input_uninterleave_time
    print 'Output un-interleaving time: %f seconds.' % output_uninterleave_time
    print 'Argmax un-interleaving time: %f seconds.' % argmax_uninterleave_time
    print 'Filter un-interleaving time: %f seconds.\n' % filter_uninterleave_time

    print 'Rotation time: %f seconds.' % rotation_time
    print 'Rotation backward time: %f seconds.' % rotation_backward_time
    print '\n \nConvolution time: %f seconds.' % test_time
    print 'Speed: %f Gflops.' % (num_operations/test_time*1e-9)

    # timer.tic()
    # outputs.convolve_and_pool_replace(inputs, filters, argmaxs, stride, padding, pooling_radius, pooling_stride, 1, True, scratch, shadow)
    # fixed_time = timer.toc()

    # print outputs

    if not time_and_dont_test:
        print 'Running convolution test. '
        inputs_np = np.lib.pad(inputs.ndarray(), ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant', constant_values = (0, 0))

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

        difference = np.mean(np.abs(outputs.ndarray() - pooled_outputs_np))
        if difference < 1.e-6:
            print 'Convolution test passed. '

        else:
            print 'Convolution test failed with difference %f. ' % difference
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

    if test_gradient:
        print '\n\n\n'
        print '='*20
        print 'Computing convolution gradient now.' 
        timer.tic()
        inputs.rotate_dimensions('backward', scratch)
        time = timer.toc()
        print 'Gradient input rotation time: %f seconds.' % time            

        timer.tic()
        outputs.rotate_dimensions('backward', scratch)
        time = timer.toc()
        print 'Gradient output rotation time: %f seconds.' % time

        timer.tic()
        argmaxs.rotate_dimensions('backward', scratch)
        time = timer.toc()
        print 'Gradient argmax rotation time: %f seconds.' % time

        timer.tic()
        inputs.interleave_for_gradient(C_block_grad, scratch)
        time = timer.toc()
        print 'Gradient input interleaving time: %f seconds.' % time
        assert Y_block_grad == 1

        timer.tic()
        filters.permute_dimensions((2, 0, 3, 1), scratch)
        time = timer.toc()
        print 'Gradient filter permutation time: %f seconds.' % time

        # filters shape [C/C_BLOCK, Y, K, X, C_BLOCK]
        timer.tic()
        filters.interleave_block(C_block_grad, scratch)
        time = timer.toc()
        print 'Gradient filter interleaving time: %f seconds.' % time

        timer.tic()
        argmaxs.permute_dimensions((0, 2, 3, 1), scratch)
        time = timer.toc()
        print 'Gradient argmax permutation time: %f seconds.' % time

        timer.tic()
        outputs.permute_dimensions((0, 2, 3, 1), scratch) # d_pooled_outputs
        time = timer.toc()
        print 'Gradient output permutation time: %f seconds.' % time

        gradient_pooled_outputs = outputs.deepcopy()
        gradient_filters = filters.deepcopy()
        gradient_inputs = inputs.deepcopy()

        timer.tic()
        gradient_filters.convolution_gradient(inputs, filters, argmaxs,
            gradient_pooled_outputs, gradient_inputs, stride, padding, pooling_radius, pooling_stride, 2, scratch)
        convolution_gradient_time = timer.toc()

        print '\n \n Time: %f seconds.' % convolution_gradient_time
        print 'Speed: %f Gflops.' % (num_operations_gradient/convolution_gradient_time*1e-9)

        if not time_and_dont_test_grad:
            # undo incorrect transformation
            gradient_filters.uninterleave_block(scratch)
            gradient_filters.permute_dimensions((1, 3, 0, 2), scratch)

            gradient_filters_tested = gradient_filters.deepcopy()
            
            #do correct transformation
            gradient_filters.permute_dimensions((0, 2, 3, 1), scratch)
        
            gradient_filters.convolution_gradient(inputs, filters, argmaxs,
                gradient_pooled_outputs, gradient_inputs, stride, padding, pooling_radius, pooling_stride, 1, scratch)
        
            gradient_filters.permute_dimensions((0, 3, 1, 2), scratch)

            difference = np.mean(np.abs(gradient_filters.ndarray() - gradient_filters_tested.ndarray()))

            print '\n\n'
            if difference < 1.e-6:
                print 'Gradient test passed. '
            else:
                print 'Gradient test failed. '


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
