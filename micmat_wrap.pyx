#Copyright (c) 2014, Oren Rippel and Ryan P. Adams
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cimport cython
from libc.stdlib cimport malloc, free
cimport cmicmat
import numpy as np
cimport numpy as np
import time

cdef long memory_used_host = 0
cdef long memory_used_mic = 0

def get_memory_used_host():
    global memory_used_host
    return memory_used_host

def get_memory_used_mic():
    global memory_used_mic
    return memory_used_mic

cdef class RandGen:
    cdef public long skip_num

    def __cinit__(self):
        self.skip_num = 0


cdef class MICMat:
    #cdef int ROWS, COLS
    cdef tuple shape
    cdef int offloaded
    cdef int persist # if is set to 1, memory is NOT deallocated by garbage collector. Use with caution!
    cdef int dtype # 0 is float, 1 is int
    cdef float *A
    cdef int *A_int

    def __cinit__(self, *args):
        if not args:
            pass

        else:
            if type(args[0]) == tuple:
                shape = args[0]
                self.allocate_host(shape)

            elif type(args[0]) == np.ndarray:
                B = args[0]
                
                assert B.dtype == np.float32 or B.dtype == np.float64, 'Unrecognized data type given to MICMat: ' + `B.dtype` + '.'
                assert B.ndim == 2, 'Array given to MICMat has wrong number of dimensions: ' + B.ndim + '.'
                
                self.copy_host(B)

            elif type(args[0]) == MICMat:
                B = args[0]
                self.copy_host(B)

            else:
                raise NameError('Unrecognized type given to MICMat: ' + `type(args[0])` + '.')


######### allocation and deallocation functions
    def allocate_host(self, *args):
        if not args:
            assert self.ROWS * self.COLS > 0, 'MICMat shape not set. Cannot allocate.'
            self.A = cmicmat.allocate_host(self.size)
            global memory_used_host
            memory_used_host += 4*self.size

        else:
            shape = args[0]
            self.shape = shape
            self.A = cmicmat.allocate_host(self.size)
            global memory_used_host
            memory_used_host += 4*self.size

        self.dtype = 0
        return self

    def allocate(self, *args):
        if not args:
            self.allocate_host()

        else:
            self.allocate_host(args[0])

        self.offload_mic()
        return self

    def free_mic_float(self):
        cmicmat.free_mic(self.size, self.A)
        global memory_used_mic
        memory_used_mic -= 4*self.size
        self.offloaded = False
        
        return self

    def free_mic_int(self):
        cmicmat.free_mic_int(self.size, self.A_int)
        global memory_used_mic
        memory_used_mic -= 4*self.size
        self.offloaded = False
        
        return self

    def free_host_float(self):
        cmicmat.free_host(self.size, self.A)
        global memory_used_host
        memory_used_host -= 4*self.size
        self.shape = (0, 0)
        return self

    def free_host_int(self):
        cmicmat.free_host_int(self.size, self.A_int)
        global memory_used_host
        memory_used_host -= 4*self.size
        self.shape = (0, 0)
        return self

    def free_mic(self):
        if self.dtype == 0:
            self.free_mic_float()

        elif self.dtype == 1:
            self.free_mic_int()

        self.offloaded = False

    def free_float(self):
        if self.offloaded: 
            self.free_mic_float()

        self.free_host_float()

        return self

    def free_int(self):
        if self.offloaded: 
            self.free_mic_int()
        
        self.free_host_int()
        return self

    def free(self):
        if self.dtype == 0:
            self.free_float()

        elif self.dtype == 1:
            self.free_int()

    def __dealloc__(self):
        #print self.shape 
        if self.size > 0 and not self.persist:
            self.free()


######### offloading and pulling functions
    def offload_mic(self):
        if not self.offloaded:
            cmicmat.offload_mic(self.size, self.A)
            global memory_used_mic
            memory_used_mic += 4*self.size
            self.offloaded = True
        else:
            cmicmat.push_mic(self.size, self.A)    
        
        return self

    def offload_host(self):
        if self.offloaded:
            self.pull_mic()

        self.free_mic()
        return self

    #def push_mic(self):
    #    cmicmat.pull_mic(self.size, self.A)
    #    return self

    def pull_mic(self):
        cmicmat.pull_mic(self.size, self.A)
        return self


######### attribute getting, setting, casting
    def __getattr__(self, name):
        if name == 'shape':
            #return [self.ROWS, self.COLS]
            return self.shape

        if name == 'ROWS':
            assert self.ndim == 2, 'MICMat shape is ' + `self.shape` + ', so cannot request for ROWS attribute.'
            return self.shape[0]

        if name == 'COLS':
            assert self.ndim == 2, 'MICMat shape is ' + `self.shape` + ', so cannot request for COLS attribute.'
            return self.shape[1]

        elif name == 'size':
            #return self.ROWS*self.COLS
            return np.prod(self.shape)

        elif name == 'ndim':
            return len(self.shape)

        elif name == 'dtype':
            return self.dtype

        elif name == 'offloaded':
            return self.offloaded

        else:
            raise NameError('Cannot get attribute \'' + name + '\'.')

    def __setattr__(self, name, value):
        if name == 'shape':
            # in pure Python need command of the form object.__setattr__(self, 'ROWS', value[0])
            # but here we're using extension types, so direct assignment is used
            if self.size > 0:
                assert self.size == np.prod(value), 'Assigned shape ' + `value` + ' does not match current shape ' + `self.shape` + '.'

            #global memory_used_host
            #memory_used_host += np.product(self.shape) - self.size

            self.shape = value

            

            #self.ROWS = value[0]
            #self.COLS = value[1]

        #if name == 'ROWS':
        #    assert len(self.shape) == 2, 'MICMat shape is ' + `self.shape` + ', so cannot set ROWS attribute.'
        #    self.shape = (value, self.shape[1])

        #if name == 'COLS':
        #    assert len(self.shape) == 2, 'MICMat shape is ' + `self.shape` + ', so cannot set COLS attribute.'
        #    self.shape = (self.shape[0], value)

        elif name == 'offloaded':
            self.offloaded = value

        elif name == 'dtype':
            self.dtype = value

        else:
            raise NameError('Cannot set attribute \'' + name + '\'.')

    def astype(self, dtype):
        # we assume that there are only 2 data types... float and int

        if dtype == 'float':
            self.A = cmicmat.cast_float(self.size, self.A_int, self.offloaded)
            self.dtype = 0

        elif dtype == 'int':
            self.A_int = cmicmat.cast_int(self.size, self.A, self.offloaded)
            self.dtype = 1

        return self

    def flatten(self):
        self.shape = (self.size,)

        return self

    def reshape(self, shape):
        if self.size > 0:
                assert self.size == np.prod(shape), 'Assigned shape ' + `shape` + ' does not match current shape ' + `self.shape` + '.'

        self.shape = shape
        return self

    def flatten_to_vectors(self):
        return self.reshape((self.shape[0], np.product(self.shape[1:])))


######### copy and replacement
    def copy_host(self, B):
        if self.size > 0: 
            self.free()
            
        cdef MICMat B_MICMat
        cdef np.ndarray[np.float32_t, ndim = 1] B_float

        if type(B) == MICMat:
            B_MICMat = B
            self.allocate_host(B_MICMat.shape)
            cmicmat.copy(B_MICMat.size, self.A, B_MICMat.A, 0)
        
        elif type(B) == np.ndarray:
            self.allocate_host(B.shape)  
            B_float = B.ravel(order = 'C').astype(np.float32)
            cmicmat.copy(B_float.size, self.A, <float *> B_float.data, 0)

        return self

    def copy_mic(self, MICMat B):
        assert B.offloaded, 'MICMat object copied on MIC is not, in fact, on the MIC.'

        if self.offloaded: 
            self.free_mic()

        cmicmat.copy(B.size, self.A, B.A, 1)
        self.offloaded = True

        return self

    def deepcopy(self):
        cdef MICMat B = MICMat()

        if self.offloaded: 
            B.allocate(self.shape)

            if self.dtype == 0:
                cmicmat.replace_mic(self.size, B.A, self.A)

            elif self.dtype == 1:
                B.astype('int')
                cmicmat.replace_mic_int(self.size, B.A_int, self.A_int)
        
        else:
            B.copy_host(self)

        return B

    def replace_host(self, B):
        cdef MICMat B_MICMat
        cdef np.ndarray[np.float32_t, ndim = 1] B_float

        if type(B) == MICMat:
            B_MICMat = B
            assert self.shape == B_MICMat.shape, 'Matrix dimensions ' + `self.shape` + ' and ' + `B_MICMat.shape` + ' don\'t match in update.'
            if self.dtype == 0:
                cmicmat.replace_host(self.size, self.A, B_MICMat.A)

            elif self.dtype == 1:
                cmicmat.replace_host_int(self.size, self.A_int, B_MICMat.A_int)                

        elif type(B) == np.ndarray:
            assert self.shape == B.shape, 'Matrix dimensions ' + `self.shape` + ' and ' + `B.shape` + ' don\'t match in update.'    
            B_float = B.ravel(order = 'C')
            cmicmat.replace_host(self.size, self.A, <float *> B_float.data)

        return self

    def replace_mic(self, B):
        cdef MICMat B_MICMat
        cdef np.ndarray[np.float32_t, ndim = 1] B_float

        if type(B) == MICMat:
            B_MICMat = B
            assert self.shape == B_MICMat.shape, 'Matrix dimensions ' + `self.shape` + ' and ' + `B_MICMat.shape` + ' don\'t match in update.'
            if self.dtype == 0:
                cmicmat.replace_mic(self.size, self.A, B_MICMat.A)

            elif self.dtype == 1:
                cmicmat.replace_mic_int(self.size, self.A_int, B_MICMat.A_int)

        elif type(B) == np.ndarray:
            assert self.shape == B.shape, 'Matrix dimensions ' + `self.shape` + ' and ' + `B.shape` + ' don\'t match in update.'
            B_float = B.ravel(order = 'C')
            cmicmat.replace_mic(self.size, self.A, <float *> B_float.data)

        return self

    def replace(self, B):
        self.replace_host(B)

        if self.offloaded:
            self.replace_mic(B)

        return self


######### slicing
    def slice_inds(self, indices):
        cdef np.ndarray[np.int32_t, ndim = 1] indices_np
        cdef MICMat indices_MICMat, A_sliced

        if type(indices) == MICMat:
            A_sliced = MICMat()
            A_sliced.shape = indices.shape
            indices_MICMat = indices
            indices_MICMat.dtype = 1
            A_sliced.A = cmicmat.slice_inds(indices_MICMat.size, indices_MICMat.A_int, self.A, indices_MICMat.offloaded, self.offloaded)

        elif type(indices) == list or type(indices) == np.ndarray:
            A_sliced = MICMat()
            A_sliced.shape = (1, len(indices))
            indices_np = np.array(indices, dtype = np.int32)
            assert indices_np.max() < self.size and indices_np.min() >= 0, 'Indices specified are out of bounds for array of size ' + `self.size` + '.'
            A_sliced.A = cmicmat.slice_inds(len(indices), <int *> indices_np.data, self.A, 0, self.offloaded)

        A_sliced.offloaded = self.offloaded
        return A_sliced

    def __getslice__(self, i, j):
        return self.slice_inds(range(i, j)) 

    def slice_cols(self, indices):     
        cdef np.ndarray[np.int32_t, ndim = 1] indices_np
        cdef MICMat indices_MICMat, A_sliced

        if type(indices) == MICMat:
            A_sliced = MICMat()
            A_sliced.shape = (self.ROWS, indices.size)
            indices_MICMat = indices
            indices_MICMat.dtype = 1
            A_sliced.A = cmicmat.slice_cols(indices_MICMat.size, indices_MICMat.A_int, self.ROWS, self.COLS, self.A, indices_MICMat.offloaded, self.offloaded)

        elif type(indices) == list or type(indices) == np.ndarray:
            A_sliced = MICMat()
            A_sliced.shape = (self.ROWS, len(indices))
            indices_np = np.array(indices, dtype = np.int32)
            assert indices_np.max() < self.COLS and indices_np.min() >= 0, 'Indices specified are out of bounds for array of shape ' + `self.shape` + '.'
            A_sliced.A = cmicmat.slice_cols(len(indices), <int *> indices_np.data, self.ROWS, self.COLS, self.A, 0, self.offloaded)

        A_sliced.offloaded = self.offloaded
        return A_sliced

    def slice_rows(self, indices):     
        cdef np.ndarray[np.int32_t, ndim = 1] indices_np
        cdef MICMat indices_MICMat, A_sliced

        if type(indices) == MICMat:
            A_sliced = MICMat()
            A_sliced.shape = (indices.size, self.COLS)
            indices_MICMat = indices
            indices_MICMat.dtype = 1
            A_sliced.A = cmicmat.slice_rows(indices_MICMat.size, indices_MICMat.A_int, self.ROWS, self.COLS, self.A, indices_MICMat.offloaded, self.offloaded)

        elif type(indices) == list or type(indices) == np.ndarray:
            A_sliced = MICMat()
            A_sliced.shape = (len(indices), self.COLS)
            indices_np = np.array(indices, dtype = np.int32)

            assert indices_np.max() < self.ROWS and indices_np.min() >= 0, 'Indices specified are out of bounds for array of shape ' + `self.shape` + '.'
            A_sliced.A = cmicmat.slice_rows(len(indices), <int *> indices_np.data, self.ROWS, self.COLS, self.A, 0, self.offloaded)

        A_sliced.offloaded = self.offloaded
        return A_sliced


######### random (and non-random) number generation
    def fill_randn(self, RandGen stream, float mu, float sigma):
        cmicmat.fill_randn(stream.skip_num, self.size, self.A, mu, sigma)
        stream.skip_num = stream.skip_num + self.size
        return self

    def fill_uniform(self, RandGen stream):
        cmicmat.fill_uniform(stream.skip_num, self.size, self.A)
        stream.skip_num = stream.skip_num + self.size
        return self

    def fill_bernoulli(self, RandGen stream, float p):
        cmicmat.fill_bernoulli(stream.skip_num, self.size, self.A_int, p)
        stream.skip_num = stream.skip_num + self.size
        return self

    def fill_uniform_int(self, RandGen stream, int i_start, int i_end):
        cmicmat.fill_uniform_int(stream.skip_num, self.size, self.A_int, i_start, i_end)
        stream.skip_num = stream.skip_num + self.size
        return self

    def fill_geometric(self, RandGen stream, float p):
        cmicmat.fill_geometric(stream.skip_num, self.size, self.A, p)
        stream.skip_num = stream.skip_num + self.size
        return self

    def fill_nested(self, MICMat geometric_samples):
        assert geometric_samples.shape[1] == 1 and geometric_samples.shape[0] == self.shape[0], 'Shape of array ' + `self.shape` + ' and shape of geometric sample vector ' + `geometric_samples.shape` + 'don\'t match.'
        
        cmicmat.fill_nested(self.shape[0], self.shape[1], self.A, geometric_samples.A)
        
        return self

    def apply_dropout(self, MICMat dropout_mask):
        #assert dropout_mask.shape == self.shape[1:], 'Shape of array ' + `self.shape` + ' and shape of mask ' + `dropout_mask.shape` + ' don\'t match.'
        assert dropout_mask.shape == self.shape, 'Shape of array ' + `self.shape` + ' and shape of mask ' + `dropout_mask.shape` + ' don\'t match.'
        
        cmicmat.apply_dropout(self.shape[0], np.product(self.shape[1:]), self.A, dropout_mask.A_int)
        
        return self

    def fill_zeros(self):
        if self.dtype == 0:
            cmicmat.fill_zeros(self.size, self.A, self.offloaded)

        elif self.dtype == 1:
            cmicmat.fill_zeros_int(self.size, self.A_int, self.offloaded)

        return self  

    def fill_ones(self):
        if self.dtype == 0:
            cmicmat.fill_ones(self.size, self.A, self.offloaded)

        elif self.dtype == 1:
            cmicmat.fill_ones_int(self.size, self.A_int, self.offloaded)

        return self

    def fill_const(self, c):
        if self.dtype == 0:
            cmicmat.fill_const(self.size, self.A, <float> c, self.offloaded)

        elif self.dtype == 1:
            cmicmat.fill_const_int(self.size, self.A_int, <int> c, self.offloaded)

        return self


######### string and array outputting
    def print_host(self):
        cmicmat.print_slice(self.shape[-2], self.shape[-1], self.A, 0)

    def print_mic(self):
        cmicmat.print_slice(self.shape[-2], self.shape[-1], self.A, 1)

    def __str__(self):
        if self.offloaded:
            print 'MICMat of shape ' + `self.shape` + ' on MIC:'
            self.print_mic()
        else:
            print 'MICMat of shape ' + `self.shape` + ' on host:'
            self.print_host()

        return ''

    def __repr__(self):
        description_str = ''
        if self.ROWS == 1 and self.COLS == 1:
            description_str = `self.float()`

        return description_str

    def array(self):
        cdef np.ndarray[np.float32_t, ndim = 2] S = np.zeros([self.ROWS, self.COLS], dtype = np.float32)

        if self.offloaded:
            self.pull_mic()
        
        S.data = <char *> cmicmat.unalign_host(self.size, self.A)
        
        return S

    def ndarray(self):
        cdef np.ndarray[np.float32_t, ndim = 4] S = np.zeros(self.shape, dtype = np.float32)

        if self.offloaded:
            self.pull_mic()
        
        S.data = <char *> cmicmat.unalign_host(self.size, self.A)
        
        return S

    def float(self):
        assert self.size == 1, 'MICMat size must be 1 to convert to float.'

        if self.offloaded:
            self.pull_mic()
        
        cdef np.float32_t S = cmicmat.output_float(self.A)
        
        return S


######### elementwise operations
    def exp(self):
        cmicmat.expo(self.size, self.A)
        return self

    def floor(self):
        cmicmat.flooro(self.size, self.A)
        return self

    def sign(self):
        cmicmat.sign(self.size, self.A)
        return self
 
    def log(self):
        cmicmat.lg(self.size, self.A)
        return self

    def abs(self):
        cmicmat.abso(self.size, self.A)
        return self

    def sqrt(self):
        cmicmat.sqrto(self.size, self.A) 
        return self

    def scale(self, float c):
        cmicmat.scale(self.size, self.A, c)
        return self

    def pow(self, b):
        cmicmat.powo(self.size, self.A, <float> b) 
        return self

    def invert(self):
        cmicmat.invert(self.size, self.A)
        return self

    def clip(self, float lower, float upper):
        cmicmat.clip(self.size, self.A, lower, upper)
        return self

    def clip_low(self, float lower):
        cmicmat.clip_low(self.size, self.A, lower)
        return self


######### matrix operations
    def horizontal_reflection(self, MICMat should_flip, MICMat scratch):
        N, C, H, W = self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        assert should_flip.size == N, 'Need should_flip size to be exactly ' + `N` + ', the size of the given tensor.'
        
        cmicmat.horizontal_reflection(N, C, H, W, self.A, should_flip.A_int, scratch.A)

    def crop(self, MICMat crop_info, int output_H, int output_W, MICMat outputs):
        N, C, H, W = self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        assert crop_info.shape == (N, 2), 'Need crop_info shape to be exactly ' + `(N, 2)` + '.'
        assert outputs.shape == (N, C, output_H, output_W), 'Need output shape to be exactly ' + `(N, C, output_H, output_W)` + '.'

        cmicmat.crop(N, C, H, W, output_H, output_W, self.A, crop_info.A_int, outputs.A, self.offloaded)

    def response_normalization(self, MICMat inputs, float alpha, float beta, int local_radius):
        N, K, H, W = inputs.shape
        # assert inputs.shape == self.shape, 'Shapes of inputs and outputs must be the same.'
        # cdef int *groups_A_int = self.A_int
        # if local_radius > 0:
        #     assert groups.shape == (N, K), 'Shape of groups matrix must be ' + `(N, K)` + '.'

        # else:
        #     groups_A_int = groups.A_int


        cmicmat.response_normalization(N, K, H, W, inputs.A, self.A, alpha, beta, local_radius)

    def response_normalization_gradient(self, MICMat inputs, MICMat outputs, MICMat gradient_outputs, float alpha, float beta, int local_radius):
        N, K, H, W = outputs.shape

        cmicmat.response_normalization_gradient(N, K, H, W, inputs.A, outputs.A, self.A, gradient_outputs.A, alpha, beta, local_radius)

    def interleave_block(self, MICMat scratch, int block):
        C, H, W, N = self.shape
        cmicmat.interleave_block(N, C, H, W, block, self.A, scratch.A)

    def uninterleave_block(self, MICMat scratch, int block):
        C, H, W, N = self.shape
        cmicmat.uninterleave_block(N, C, H, W, block, self.A, scratch.A)

    def convolution(self, MICMat inputs, MICMat filters, MICMat argmaxs, int stride, int padding, int pooling_radius, int pooling_stride, layer, argmaxs_fixed, MICMat scratch):
        # asserts that check number of dimensions, sizes, etc

        C, H, W, N = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]
        # K, Y, X = filters.shape[0], filters.shape[2], filters.shape[3]
        _, Y, X, K = filters.shape

        output_H = (H + 2*padding - Y + 1)/stride
        output_W = (W + 2*padding - X + 1)/stride
        pooled_H = (output_H - pooling_radius + 1)/pooling_stride
        pooled_W = (output_W - pooling_radius + 1)/pooling_stride
        
        # assert self.shape == (K, pooled_H, pooled_W, N), 'Output shape is ' + `self.shape` + ' rather than ' + `(K, pooled_H, pooled_W, N)` + '.'
        # assert argmaxs.shape == (K, pooled_H, pooled_W, N), 'Argmax shape is ' + `self.shape` + ' rather than ' + `(K, pooled_H, pooled_W, N)` + '.'

        times = time.time()
        if layer == 1:
            start_time = time.time()
            #if not argmaxs_fixed:
            cmicmat.convolution_layer1(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)

            #else:
                #cmicmat.convolve_argmaxs_fixed_layer1(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded)
        
        elif layer == 2:
            #if not argmaxs_fixed:
            cmicmat.convolve_and_pool_layer2(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)

            #else:
                #cmicmat.convolve_argmaxs_fixed_layer2(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded)

    def convolution_gradient(self, MICMat inputs, MICMat filters, MICMat argmaxs,
        MICMat gradient_pooled_outputs, MICMat gradient_inputs, 
        int stride, int padding, int pooling_radius, int pooling_stride, layer, MICMat scratch):
        # self is the filters gradient
        C, H, W, N = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]
        K, Y, X = filters.shape[0], filters.shape[2], filters.shape[3]

        output_H = (H + 2*padding - Y + 1)/stride
        output_W = (W + 2*padding - X + 1)/stride
        pooled_H = (output_H - pooling_radius + 1)/pooling_stride
        pooled_W = (output_W - pooling_radius + 1)/pooling_stride
        
        assert self.shape == filters.shape, 'Filter shape is ' + self.shape + ' rather than ' + `filters.shape` + '.'
        assert gradient_inputs.shape == inputs.shape, 'gradient_inputs shape is ' + gradient_inputs.shape + ' rather than ' + `inputs.shape` + '.'
        assert gradient_pooled_outputs.shape == (K, pooled_H, pooled_W, N), 'gradient_pooled_outputs shape is ' + `gradient_pooled_outputs.shape` + ' rather than ' + `(K, pooled_H, pooled_W, N)` + '.'
        
        if layer == 1:
            cmicmat.convolution_gradient_layer1(N, C, H, W, inputs.A, K, Y, X, padding, filters.A, argmaxs.A_int, gradient_pooled_outputs.A, 
                gradient_inputs.A, self.A, scratch.A)

        elif layer == 2:
            cmicmat.convolve_gradient_layer2(N, C, H, W, inputs.A, K, Y, X, padding, filters.A, argmaxs.A_int, gradient_pooled_outputs.A, 
                gradient_inputs.A, self.A, scratch.A)

    def convolve_and_pool_replace(self, MICMat inputs, MICMat filters, MICMat argmaxs, int stride, int padding, int pooling_radius, int pooling_stride, layer, argmaxs_fixed, MICMat scratch, shadow):
        # asserts that check number of dimensions, sizes, etc

        N, C, H, W = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]
        K, Y, X = filters.shape[0], filters.shape[2], filters.shape[3]
        
        K_preshadow = K
        K = 2*K if shadow else K

        output_H = (H + 2*padding - Y + 1)/stride
        output_W = (W + 2*padding - X + 1)/stride
        pooled_H = (output_H - pooling_radius + 1)/pooling_stride
        pooled_W = (output_W - pooling_radius + 1)/pooling_stride
        
        assert self.shape == (N, K, pooled_H, pooled_W), 'Output shape is ' + `self.shape` + ' rather than ' + `(N, K, pooled_H, pooled_W)` + '.'
        assert argmaxs.shape == (N, K, pooled_H, pooled_W), 'Argmax shape is ' + `self.shape` + ' rather than ' + `(N, K, pooled_H, pooled_W)` + '.'

        times = time.time()
        if layer == 1:
            start_time = time.time()
            #if not argmaxs_fixed:
            if not shadow:
                cmicmat.convolve_and_pool_layer1(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)

            else:
                cmicmat.convolve_and_pool_shadow_layer1(N, C, H, W, inputs.A, K_preshadow, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)                

            #else:
                #cmicmat.convolve_argmaxs_fixed_layer1(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded)
        
        elif layer == 2:
            #if not argmaxs_fixed:
            cmicmat.convolve_and_pool_layer2(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)

            #else:
                #cmicmat.convolve_argmaxs_fixed_layer2(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded)

    def local_replace(self, MICMat inputs, MICMat filters, int stride, int padding, layer, MICMat scratch):
        # asserts that check number of dimensions, sizes, etc
        N, C, H, W = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]
        K, Y, X = filters.shape[0], filters.shape[-2], filters.shape[-1]

        output_H = (H + 2*padding - Y + 1)/stride
        output_W = (W + 2*padding - X + 1)/stride        
        
        assert self.shape == (N, K, output_H, output_W), 'Output shape is ' + `self.shape` + ' rather than ' + `(N, K, output_H, output_W)` + '.'

        times = time.time()
        if layer == 1:
            start_time = time.time()
            cmicmat.local_layer1(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, stride, padding, self.offloaded, scratch.A)
        
        elif layer == 2:
            cmicmat.local_layer2(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, stride, padding, self.offloaded, scratch.A)


    def speed_tester(self, MICMat outputs):
        cmicmat.speed_tester(self.shape[0], self.A, outputs.A)

    # def convolve_and_pool_replace(self, MICMat inputs, argmaxs, MICMat filters, int pool_radius, int stride, layer):
    #     # asserts that check number of dimensions, sizes, etc
    #     N, C, H, W = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]
    #     K, Y, X = filters.shape[0], filters.shape[2], filters.shape[3]

    #     pooled_H = np.ceil((<float> (H - Y + 1))/pool_radius)
    #     pooled_W = np.ceil((<float> (W - X + 1))/pool_radius)
        
    #     assert self.shape == (N, K, pooled_H, pooled_W), 'Output shape is ' + self.shape + ' rather than ' + `(N, K, pooled_H, pooled_W)` + '.'
        
    #     cdef MICMat argmaxs_MICMat
    #     cdef int argmaxs_fixed = 1
    #     times = time.time()
    #     if argmaxs is None:
    #         argmaxs_MICMat = MICMat(self.shape)
    #         if self.offloaded:
    #             argmaxs_MICMat.offload_mic()
    #         argmaxs_MICMat.astype('int')
    #         argmaxs_fixed = 0

    #     else:
    #         argmaxs_MICMat = argmaxs

    #     # print time.time() - times

    #     times = time.time()
    #     #argmaxs_MICMat.A_int = cmicmat.convolve_and_pool(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, pool_radius, stride, argmaxs_MICMat.A_int, argmaxs_fixed, self.offloaded)
    #     if layer == 1:
    #         argmaxs_MICMat.A_int = cmicmat.convolve_and_pool_layer1(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, pool_radius, stride, argmaxs_MICMat.A_int, argmaxs_fixed, self.offloaded)
        
    #     elif layer == 2:
    #         argmaxs_MICMat.A_int = cmicmat.convolve_and_pool_layer2(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, pool_radius, stride, argmaxs_MICMat.A_int, argmaxs_fixed, self.offloaded)

    #     # print time.time() - times

    #     return argmaxs_MICMat

    def convolve_gradient(self, MICMat inputs, MICMat filters, MICMat argmaxs,
        MICMat gradient_pooled_outputs, MICMat gradient_inputs, 
        int stride, int padding, int pooling_radius, int pooling_stride, layer, MICMat scratch, shadow):
        # self is the filters gradient
        N, C, H, W = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]
        K, Y, X = filters.shape[0], filters.shape[2], filters.shape[3]

        K_preshadow = K
        K = 2*K if shadow else K
        # pooled_H = np.ceil((<float> (H - Y + 1))/pool_radius)
        # pooled_W = np.ceil((<float> (W - X + 1))/pool_radius)
        output_H = (H + 2*padding - Y + 1)/stride
        output_W = (W + 2*padding - X + 1)/stride
        pooled_H = (output_H - pooling_radius + 1)/pooling_stride
        pooled_W = (output_W - pooling_radius + 1)/pooling_stride
        
        assert self.shape == filters.shape, 'Filter shape is ' + self.shape + ' rather than ' + `filters.shape` + '.'
        assert gradient_inputs.shape == inputs.shape, 'gradient_inputs shape is ' + gradient_inputs.shape + ' rather than ' + `inputs.shape` + '.'
        assert gradient_pooled_outputs.shape == (N, K, pooled_H, pooled_W), 'gradient_pooled_outputs shape is ' + `gradient_pooled_outputs.shape` + ' rather than ' + `(N, K, pooled_H, pooled_W)` + '.'

        #cmicmat.convolve_gradient(N, C, H, W, inputs.A, K, Y, X, filters.A, argmaxs.A_int, gradient_pooled_outputs.A, 
            #pool_radius, gradient_inputs.A, self.A)
        
        if layer == 1:
            cmicmat.convolve_gradient_layer1(N, C, H, W, inputs.A, K_preshadow, Y, X, padding, filters.A, argmaxs.A_int, gradient_pooled_outputs.A, 
                gradient_inputs.A, self.A, scratch.A)

            # print self
            if shadow:
                cmicmat.convolve_gradient_shadow_layer1(N, C, H, W, inputs.A, K_preshadow, Y, X, padding, filters.A, argmaxs.A_int, gradient_pooled_outputs.A, 
                    gradient_inputs.A, self.A, scratch.A)
                # self *= 2.
            # print self

        elif layer == 2:
            cmicmat.convolve_gradient_layer2(N, C, H, W, inputs.A, K, Y, X, padding, filters.A, argmaxs.A_int, gradient_pooled_outputs.A, 
                gradient_inputs.A, self.A, scratch.A)

    def local_gradient(self, MICMat inputs, MICMat filters, MICMat gradient_outputs, MICMat gradient_inputs, int stride, int padding, layer, MICMat scratch):
        # self is the filters gradient

        N, C, H, W = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]
        K, Y, X = filters.shape[0], filters.shape[-2], filters.shape[-1]

        output_H = (H + 2*padding - Y + 1)/stride
        output_W = (W + 2*padding - X + 1)/stride
        
        assert self.shape == filters.shape, 'Filter shape is ' + self.shape + ' rather than ' + `filters.shape` + '.'
        assert gradient_inputs.shape == inputs.shape, 'gradient_inputs shape is ' + gradient_inputs.shape + ' rather than ' + `inputs.shape` + '.'
        assert gradient_outputs.shape == (N, K, output_H, output_W), 'gradient_outputs shape is ' + `gradient_outputs.shape` + ' rather than ' + `(N, K, output_H, output_W)` + '.'

        if layer == 1:
            cmicmat.local_gradient_layer1(N, C, H, W, inputs.A, K, Y, X, padding, filters.A, gradient_outputs.A, gradient_inputs.A, self.A, scratch.A)

        elif layer == 2:
            cmicmat.local_gradient_layer2(N, C, H, W, inputs.A, K, Y, X, padding, filters.A, gradient_outputs.A, gradient_inputs.A, self.A, scratch.A)

    def T(self):
        cdef MICMat S = self.deepcopy()

        cmicmat.T(S.ROWS, S.COLS, S.A)
        S.shape = (self.shape[1], self.shape[0])

        return S

    def T_replace(self):
        cmicmat.T(self.ROWS, self.COLS, self.A)
        self.shape = (self.shape[1], self.shape[0])

        return self

    def rotate_dimensions(self, MICMat scratch, direction):
        if direction == 'forward':
            rows = self.shape[0]
            cols = np.product(self.shape[1:])
            self.shape = self.shape[1:] + (self.shape[0],)

        elif direction == 'backward':
            rows = np.product(self.shape[0:-1])
            cols = self.shape[-1]
            self.shape = (self.shape[-1],) + self.shape[0:-1]

        cmicmat.transpose_replace(rows, cols, self.A, scratch.A)

        return self

    def dot(self, MICMat V, *args):
        cdef MICMat S
        cdef int T_A = False
        cdef int T_B = False
        cdef int COLS_LEFT, ROWS_RIGHT, ROWS_LEFT, COLS_RIGHT

        if not args:
            pass

        else:
            T_A = args[0]
            T_B = args[1]

        if not T_A:
            COLS_LEFT = self.COLS
            ROWS_LEFT = self.ROWS
        else: 
            COLS_LEFT = self.ROWS
            ROWS_LEFT = self.COLS

        if not T_B:
            ROWS_RIGHT = V.ROWS
            COLS_RIGHT = V.COLS
        else: 
            ROWS_RIGHT = V.COLS
            COLS_RIGHT = V.ROWS
        
        assert COLS_LEFT == ROWS_RIGHT, 'Matrix dimensions ' + `self.shape` + ' and ' + `V.shape` + ' don\'t match in matrix product.'
        
        S = MICMat((ROWS_LEFT, COLS_RIGHT))
        if self.offloaded:
            S.offload_mic()
            
        cmicmat.dot_replace(self.ROWS, self.COLS, T_A, self.A, V.ROWS, V.COLS, T_B, V.A, 0., S.A)

        return S

    def dot_replace(self, MICMat M, MICMat V, *args):
    # self = M*V
        cdef int T_A = False
        cdef int T_B = False
        cdef int COLS_LEFT, ROWS_RIGHT, ROWS_LEFT, COLS_RIGHT

        if not args:
            pass

        else:
            T_A = args[0]
            T_B = args[1]

        if not T_A:
            COLS_LEFT = M.COLS
            ROWS_LEFT = M.ROWS
        else: 
            COLS_LEFT = M.ROWS
            ROWS_LEFT = M.COLS

        if not T_B:
            ROWS_RIGHT = V.ROWS
            COLS_RIGHT = V.COLS
        else: 
            ROWS_RIGHT = V.COLS
            COLS_RIGHT = V.ROWS


        assert COLS_LEFT == ROWS_RIGHT, 'Matrix dimensions ' + `M.shape` + ' and ' + `V.shape` + ' don\'t match in matrix product.'
        assert self.ROWS == ROWS_LEFT and self.COLS == COLS_RIGHT, 'Matrix product of matrices of dimensions ' + `M.shape` + ' and ' + `V.shape` + ' doesn\'t match output MICMat ' + `self.shape` + '.'
        
        cmicmat.dot_replace(M.ROWS, M.COLS, T_A, M.A, V.ROWS, V.COLS, T_B, V.A, 0., self.A)
        return self

    def dot_replace_update(self, MICMat M, MICMat V, *args):
    # self = self + M*V
        cdef int T_A = False
        cdef int T_B = False
        cdef COLS_LEFT, ROWS_RIGHT, ROWS_LEFT, COLS_RIGHT

        if not args:
            pass

        else:
            T_A = args[0]
            T_B = args[1]

        if not T_A:
            COLS_LEFT = M.COLS
            ROWS_LEFT = M.ROWS
        else: 
            COLS_LEFT = M.ROWS
            ROWS_LEFT = M.COLS

        if not T_B:
            ROWS_RIGHT = V.ROWS
            COLS_RIGHT = V.COLS
        else: 
            ROWS_RIGHT = V.COLS
            COLS_RIGHT = V.ROWS


        assert COLS_LEFT == ROWS_RIGHT, 'Matrix dimensions ' + `M.shape` + ' and ' + `V.shape` + ' don\'t match in matrix product.'
        assert self.ROWS == ROWS_LEFT and self.COLS == COLS_RIGHT, 'Matrix product of matrices of dimensions ' + `M.shape` + ' and ' + `V.shape` + ' doesn\'t match output MICMat ' + `self.shape` + '.'
        
        cmicmat.dot_replace(M.ROWS, M.COLS, T_A, M.A, V.ROWS, V.COLS, T_B, V.A, 1., self.A)

    def dot_vec(self, MICMat B):
    # sum(self .* B)
        cdef MICMat S = MICMat()
        S.shape = (1, 1)
        assert self.size == B.size, 'Matrix dimensions don\'t match for dot vector product.'
        S.A = cmicmat.dot_vec(self.size, self.A, B.A)
        S.offloaded = self.offloaded

        return S 

    def update(self, V, *args):
        cdef float ALPHA
        cdef MICMat V_MICMat
        if not args:
            ALPHA = 1.0
        else:
            ALPHA = args[0]

        if type(V) == MICMat:
            V_MICMat = V
            assert self.ndim == V_MICMat.ndim, 'Dimensions ' + `self.ndim` + ' and ' + `V_MICMat.ndim` + ' of MICMats do not match in update.'
            a1, a2, a3, a4, a5, a6 = tuple([1]*(6 - len(self.shape))) + self.shape
            b1, b2, b3, b4, b5, b6  = tuple([1]*(6 - len(V_MICMat.shape))) + V_MICMat.shape
            assert (b1 == a1 or b1 == 1) and (b2 == a2 or b2 == 1) and (b3 == a3 or b3 == 1) and (b4 == a4 or b4 == 1) and (b5 == a5 or b5 == 1) and (b6 == a6 or b6 == 1), 'Matrix dimensions ' + `self.shape` + ' and ' + `V_MICMat.shape` + ' don\'t match in update.'
            assert self.offloaded == V_MICMat.offloaded, 'Both MICMats must be either on the host or on the MIC.'

            cmicmat.update(a1, a2, a3, a4, a5, a6, self.A, b1, b2, b3, b4, b5, b6, V_MICMat.A, ALPHA, self.offloaded)

        elif type(V) == np.float32 or type(V) == np.float64 or type(V) == float:
            if type(V) == np.float64:
                V = V.astype(np.float32)
            cmicmat.update_const(self.size, self.A, ALPHA*V)

        return self

    def multiply(self, V):
        cdef MICMat V_MICMat

        if type(V) == MICMat:
            V_MICMat = V
            if V_MICMat.ndim == 2:
                assert (self.ROWS == V_MICMat.ROWS and self.COLS == V_MICMat.COLS) or (self.ROWS == V_MICMat.ROWS and V_MICMat.COLS == 1) or (V_MICMat.ROWS == 1 and self.COLS == V_MICMat.COLS) or (V_MICMat.ROWS == 1 and V_MICMat.COLS == 1), 'Matrix dimensions ' + `self.shape` + ' and ' + `V.shape` + ' don\'t match in update.'
                cmicmat.mult(self.ROWS, self.COLS, self.A, V_MICMat.ROWS, V_MICMat.COLS, V_MICMat.A)

            else: # multiply elementwise for tensors
                assert self.size == V_MICMat.size, 'Tensors multiplied must have same sizes, and instead have shapes ' + `self.shape` + ' and ' + `V_MICMat.shape` + '.'
                cmicmat.mult(self.size, 1, self.A, V_MICMat.size, 1, V_MICMat.A)                
        
        elif type(V) == np.float32 or type(V) == np.float64 or type(V) == float or type(V) == int:
            if type(V) == np.float64:
                V = V.astype(np.float32)
            cmicmat.scale(self.size, self.A, V)

        return self

    def divide(self, V):
        cdef MICMat V_MICMat

        if type(V) == MICMat:
            V_MICMat = V
            assert (self.ROWS == V_MICMat.ROWS and self.COLS == V_MICMat.COLS) or (self.ROWS == V_MICMat.ROWS and V_MICMat.COLS == 1) or (V_MICMat.ROWS == 1 and self.COLS == V_MICMat.COLS) or (V_MICMat.ROWS == 1 and V_MICMat.COLS == 1), 'Matrix dimensions ' + `self.shape` + ' and ' + `V.shape` + ' don\'t match in update.'
            cmicmat.divide(self.ROWS, self.COLS, self.A, V_MICMat.ROWS, V_MICMat.COLS, V_MICMat.A)
        elif type(V) == np.float32 or type(V) == np.float64 or type(V) == float or type(V) == int:
            if type(V) == np.float64:
                V = V.astype(np.float32)
            cmicmat.scale(self.size, self.A, 1.0/V)

        return self

    def sum(self, *args):         
        cdef MICMat S = MICMat()
        cdef int AXIS = 2

        if not args or args[0] == 2:
            S.shape = (1, 1)
            S.A = cmicmat.sum_axis(self.size, 1, self.A, AXIS)
        else:
            AXIS = args[0]
            if AXIS == 0:
                S.shape = (1, self.COLS)

            if AXIS == 1:
                S.shape = (self.ROWS, 1)
            
            S.A = cmicmat.sum_axis(self.ROWS, self.COLS, self.A, AXIS)    
        
        S.offloaded = self.offloaded
        return S

    def sum_replace(self, MICMat A, *args):         
        cdef int AXIS = 2

        if not args or args[0] == 2:
            assert self.shape == (1, 1), 'MICMat shape must be (1, 1).'
        else:
            AXIS = args[0]
            if AXIS == 0:
                assert self.shape == (1, A.COLS), 'MICMat shape must be ' + `(1, A.COLS)` + '.'

            if AXIS == 1:
                assert self.shape == (A.ROWS, 1), 'MICMat shape must be ' + `(A.ROWS, 1)` + '.'
            
            cmicmat.sum_axis_replace(A.ROWS, A.COLS, A.A, AXIS, self.A)    
        
        return self 

    # def sum(self, *args):         
        # cdef MICMat S = MICMat()
        # cdef int AXIS = -1

        # if not args or args[0] == -1:
        #     pass 

        # else:
        #     AXIS = args[0]
        
        # a1, a2, a3, a4 = tuple([1]*(4 - len(self.shape))) + self.shape    
        
        # if AXIS == -1:
        #     S.shape = tuple([1]*self.ndim)
        
        # else:
        #     shape = [a1, a2, a3, a4]
        #     shape = shape[-self.ndim:]
        #     shape[AXIS] = 1
        #     S.shape = tuple(shape)

        # S.A = cmicmat.sum_axis(a1, a2, a3, a4, self.A, AXIS, self.offloaded)    

        # S.offloaded = self.offloaded
        # return S 

    def max_and_arg(self, *args):         
        cdef MICMat I = MICMat()
        cdef int AXIS = 2

        if not args or args[0] == 2:
            I.shape = (1, 1)

        else:
            AXIS = args[0]
            if AXIS == 0:
                I.shape = (1, self.COLS)

            if AXIS == 1:
                I.shape = (self.ROWS, 1)
        
        I.A_int = cmicmat.max_axis(self.ROWS, self.COLS, self.A, AXIS)
        I.dtype = 1
        I.offloaded = self.offloaded
        cdef MICMat S = self.slice_inds(I)
        
        if AXIS != 2:
            cmicmat.index_global_to_local(self.ROWS, self.COLS, I.A_int, AXIS)
        
        return S, I

    def max(self, *args):
        cdef int AXIS = 2
        if not args:
            pass
        else:
            AXIS = args[0]

        S, _ = self.max_and_arg(AXIS)
        return S

    def min(self, *args):
        cdef int AXIS = 2
        if not args:
            pass
        else:
            AXIS = args[0]

        return self.scale(-1.).max(AXIS).scale(-1.)

    def argmax(self, *args):
        cdef int AXIS = 2
        if not args:
            pass
        else:
            AXIS = args[0]

        _, I = self.max_and_arg(AXIS)
        return I

    def mean(self, *args):
        cdef int AXIS = 2 
        cdef float SCALING

        if not args or args[0] == 2:
            SCALING = 1.0/(<float> self.size)
        else:
            AXIS = args[0]
            if AXIS == 0:
                SCALING = 1.0/(<float> self.ROWS)

            if AXIS == 1:
                SCALING = 1.0/(<float> self.COLS)

        
        cdef MICMat S = self.sum(AXIS)
        S.offloaded = self.offloaded
        S.scale(SCALING)
        return S      

    def norm(self, *args):

        cdef int AXIS = 2 
        if not args:
            pass
        else:
            AXIS = args[0]
        
        cdef MICMat B = self.deepcopy()
        B.pow(2.0)

        cdef MICMat S = B.sum(AXIS)
        S.offloaded = self.offloaded
        S.sqrt()
        B.free()
        return S 

    def var(self, *args):         
        cdef MICMat B = self.deepcopy()
        B.pow(2.0)

        cdef int AXIS = 2
        if not args:
            pass
        else:
            AXIS = args[0]

        cdef MICMat D = self.mean(AXIS)
        D.pow(2.0)
        
        B.update(D, -1.0)   
        
        cdef MICMat S = B.mean(AXIS)
        S.offloaded = self.offloaded

        return S

    def std(self, *args):
        cdef int AXIS = 2
        if not args:
            pass
        else:
            AXIS = args[0]

        cdef MICMat S = self.var(AXIS)
        S.offloaded = self.offloaded
        S.sqrt()
        return S


######### native Python operator implementations
    def __add__(self, V):
        cdef MICMat S
        if type(self) == float:
            S = V.deepcopy()
            S.update(self)
        else:
            S = self.deepcopy()
            S.update(V)

        return S

    def __sub__(self, V):
        cdef MICMat S
        if type(self) == float:
            S = -V.deepcopy()
            S.update(self)
        else:
            S = self.deepcopy()
            S.update(V, -1.0)

        return S
        
    def __mul__(self, V):
        cdef MICMat S
        if type(self) == float:
            S = V.deepcopy()
            S.multiply(self)
        else:
            S = self.deepcopy()
            S.multiply(V)

        return S

    def __div__(self, V):
        cdef MICMat S
        if type(self) == float:
            S = V.deepcopy()
            S.invert()
            return self * S
        else:
            S = self.deepcopy()
            S.divide(V)
        
        return S

    def __abs__(self):
        cdef MICMat S = self.deepcopy()
        return S.abs()

    def __neg__(self):
        cdef MICMat S = self.deepcopy()
        return S * (-1.0)
        

    def __iadd__(self, V):
        self.update(V)
        return self

    def __isub__(self, V):
        self.update(V, -1.0)
        return self

    def __imul__(self, V):
        self.multiply(V)
        return self

    def __idiv__(self, V):
        self.divide(V)
        return self

    def __len__(self):
        return self.size

    def equal(self, MICMat V):
        assert self.shape == V.shape, 'The shapes ' + `self.shape` + ' and ' + `V.shape` + ' of the compared MICMats don\'t match.'
        cdef MICMat S = MICMat()
        S.shape = V.shape
        S.offloaded = self.offloaded
        S.A = cmicmat.equal(self.size, self.A, V.A)

        return S

    def leq(self, float V):
        cdef MICMat S = MICMat()
        S.shape = self.shape
        S.offloaded = self.offloaded
        S.A = cmicmat.leq(self.size, self.A, V)

        return S

    def geq(self, float V):
        cdef MICMat S = MICMat()
        S.shape = self.shape
        S.offloaded = self.offloaded
        S.A = cmicmat.geq(self.size, self.A, V)

        return S

    def greater(self, float V):
        cdef MICMat S = MICMat()
        S.shape = self.shape
        S.offloaded = self.offloaded
        S.A = cmicmat.greater(self.size, self.A, V)

        return S

    def greater_replace(self, float V):
        cmicmat.greater_replace(self.size, self.A, V)

        return self

    def __or__(MICMat self, MICMat V):
        assert self.shape == V.shape, 'The shapes ' + `self.shape` + ' and ' + `V.shape` + ' of the compared MICMats don\'t match.'
        cdef MICMat S = MICMat()
        S.shape = V.shape
        S.offloaded = self.offloaded
        S.A = cmicmat.elementwise_or(self.size, self.A, V.A)
        return S

    def __richcmp__(self, V, int operation):
        if operation == 1:
            assert type(V) == float or type(V) == np.float32, 'Currently can only do comparison for RHS being a float.'
            return self.leq(V)

        elif operation == 2:
            return self.equal(V)

        elif operation == 3:
            return self.equal(V).update(-1.).scale(-1.)

        elif operation == 4:
            assert type(V) == float or type(V) == np.float32, 'Currently can only do comparison for RHS being a float.'
            return self.greater(V)

        elif operation == 5:
            assert type(V) == float or type(V) == np.float32, 'Currently can only do comparison for RHS being a float.'
            return self.geq(V)


######### misc
    def labels_to_vectors(self, int K):
        cdef MICMat S = MICMat()
        S.shape = (self.shape[0], K)
        S.offloaded = self.offloaded
        S.A = cmicmat.labels_to_vectors(self.shape[0], K, self.A)

        return S   

    def frac_zero(self):
        cdef MICMat zeros = MICMat(self.shape) 
        if self.offloaded:
            zeros.offload_mic()

        zeros.fill_zeros()
        return (self == zeros).sum().float()/self.size


cdef class Scratch(MICMat):
    cdef list shapes

    def __cinit__(self, *args):
        self.shapes = []

    def __getattr__(self, name):
        if name == 'shapes':
            return self.shapes

        else:
            return MICMat.__getattr__(self, name)

    def reset(self, args):
        if type(args) == MICMat:
            self.shapes = []
            return self.append(args)

        elif type(args) == tuple:
            self.shapes = [args]
            return self.get(0)

    def append(self, MICMat V):
        assert self.offloaded == V.offloaded, 'Input MICMat must be offloaded to the same resource as the scratch!'
        assert self.size >= self.contents_size() + V.size, 'The scratch space of size ' + `self.size` + ' is out of memory with contents of size ' + `self.contents_size()` + ' and incoming object of size ' + `V.size` + '!'
        cmicmat.replace_partial_host(self.size, self.contents_size(), self.A, V.size, 0, V.A)
        if self.offloaded:
            cmicmat.replace_partial_mic(self.size, self.contents_size(), self.A, V.size, 0, V.A)

        self.shapes.append(V.shape)

        return self.get(len(self.shapes) - 1)

    def get(self, index):
        cdef MICMat S = MICMat()
        S.shape = self.shapes[index]
        S.offloaded = self.offloaded
        S.persist = 1
        S.A = cmicmat.get_partial(S.size, self.contents_size(index), self.A)
        return S

    def contents_size(self, *args):
        if not args:
            return sum([np.product(shape) for shape in self.shapes])
        
        else:
            index = args[0]
            return sum([np.product(shape) for shape in self.shapes[:index]])

def ping_each_core():
        print 'Pinging each core on the MIC.'
        cmicmat.ping_each_core()

def check_mic_status():
    cmicmat.check_mic_status()

def tester():
    cmicmat.tester()