#Copyright (c) 2014, Oren Rippel and Ryan P. Adams
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import micmat_wrap as C
import numpy as np
import timeit
import time
from functools import partial

def main():

    C.check_mic_status()
    
    time_only = True
    offload = True
    test_convolution(time_only, offload)

    # test_update(time_only, offload)

    # test_sum(time_only, offload)

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


def test_convolution(time_only, offload):
    if time_only:
        N = 256
        K = 50
        c = 3
        H = 32
        W = 32
        X = 8
        Y = 8
        stride = 1
        pool_radius = 8
    else:
        N = 20
        K = 10
        c = 3
        H = 10
        W = 10
        X = 2
        Y = 2
        stride = 1
        pool_radius = 2

    num_operations = N*K*c*(H-Y+1)*(W-X+1)*X*Y*2

    pooled_H = np.ceil(((H - Y + 1.))/pool_radius)
    pooled_W = np.ceil(((W - X + 1.))/pool_radius)

    inputs = C.MICMat((N, c, H, W))
    inputs.fill_zeros()
    
    filters = C.MICMat((K, c, Y, X))
    filters.fill_zeros()

    outputs = C.MICMat((N, K, pooled_H, pooled_W))
    

    if offload:
        inputs.offload_mic()
        inputs.fill_randn(stream, 0., 1.)

        outputs.offload_mic()
        
        filters.offload_mic()
        filters.fill_randn(stream, 0., 1.)
    
    outputs.fill_zeros()

    print 'Computing convolution now.'
    timer.tic()
    argmaxs = outputs.convolve_and_pool_replace(inputs, None, filters, pool_radius, stride)
    test_time = timer.toc()

    if not time_only:
        pure_python = np.zeros((N, K, pooled_H, pooled_W)) - 1e10
        for n in range(N):
            for k in range(K):
                for h in range(H - Y + 1):
                    for w in range(W - X + 1):
                        pure_python[n, k, h/pool_radius, w/pool_radius] = max(np.sum(np.multiply(inputs.ndarray()[n, :, h:h+Y, w:w+X], filters.ndarray()[k, :, :, :])), pure_python[n, k, h/pool_radius, w/pool_radius])

        difference = np.mean(np.abs(outputs.ndarray() - pure_python))
        if difference < 1.e-6:
            print 'Convolution test passed. ',

        else:
            print 'Convolution test failed with difference %f. ' % difference,
            print outputs
            print pure_python[0, 0, :, :]

    else:
        print 'Time: %f seconds.' % test_time
        print 'Speed: %f Gflops.' % (num_operations/test_time*2e-9)

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

stream = C.RandGen()
timer = Timer()

if __name__ == '__main__':
    main()
