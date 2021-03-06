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

from timer import *

cdef long memory_used_host = 0
cdef long memory_used_mic = 0

stream = None
scratch = None

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


def set_stream():
    global stream
    stream = RandGen()

def set_scratch(size):
    global scratch
    scratch = Scratch((size, 1))
    scratch.offload_mic()
    scratch.fill_zeros()

cdef class MICMat:
    #cdef int ROWS, COLS
    cdef tuple shape
    cdef int offloaded
    cdef int persist # if is set to 1, memory is NOT deallocated by garbage collector. Use with caution!
    cdef int dtype # 0 is float, 1 is int
    cdef float *A
    cdef int *A_int
    cdef char *data_format

    def __cinit__(self, *args):
        if not args:
            pass

        else:
            if type(args[0]) == tuple:
                self.shape = tuple([int(v) for v in args[0]])
                self.allocate_host(self.shape)

            elif type(args[0]) == np.ndarray:
                B = args[0]
                
                assert B.dtype == np.float32 or B.dtype == np.float64, 'Unrecognized data type given to MICMat: ' + `B.dtype` + '.'
                # assert B.ndim == 2, 'Array given to MICMat has wrong number of dimensions: ' + B.ndim + '.'
                
                self.copy_host(B)

            elif type(args[0]) == MICMat:
                B = args[0]
                self.copy_host(B)

            else:
                raise NameError('Unrecognized type given to MICMat: ' + `type(args[0])` + '.')


######### allocation and deallocation functions
    @profile_routine
    def allocate_host(self, *args):
        if not args:
            assert self.size > 0, 'MICMat shape not set. Cannot allocate.'
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

    @profile_routine
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

    @profile_routine
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
    @profile_routine
    def offload_mic(self):
        if not self.offloaded:
            cmicmat.offload_mic(self.size, self.A)
            global memory_used_mic
            memory_used_mic += 4*self.size
            self.offloaded = True
        else:
            cmicmat.push_mic(self.size, self.A)    
        
        return self

    @profile_routine
    def offload_host(self):
        if self.offloaded:
            self.pull_mic()

        self.free_mic()
        return self

    #def push_mic(self):
    #    cmicmat.pull_mic(self.size, self.A)
    #    return self

    @profile_routine
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
            return int(np.prod(self.shape)) if self.shape is not None else None

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

            self.shape = tuple([int(v) for v in value])

            

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

    @profile_routine
    def astype(self, dtype, *args):
        cdef MICMat output
        # we assume that there are only 2 data types... float and int
        if not args:
            if dtype == 'float':
                self.A = cmicmat.cast_float(self.size, self.A_int, self.offloaded)
                self.dtype = 0

            elif dtype == 'int':
                self.A_int = cmicmat.cast_int(self.size, self.A, self.offloaded)
                self.dtype = 1
        
        else:
            output = args[0]
            if dtype == 'int':
                cmicmat.cast_int_replace(self.size, self.A, output.A_int, self.offloaded)

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
    @profile_routine
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

    @profile_routine
    def copy_mic(self, MICMat B):
        assert B.offloaded, 'MICMat object copied on MIC is not, in fact, on the MIC.'

        if self.offloaded: 
            self.free_mic()

        cmicmat.copy(B.size, self.A, B.A, 1)
        self.offloaded = True

        return self

    @profile_routine
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

    @profile_routine
    def replace(self, B):
        self.replace_host(B)

        if self.offloaded:
            self.replace_mic(B)

        return self


######### slicing
    @profile_routine
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

    # def slice_inds_replace(self, B, indices):
    #     cdef np.ndarray[np.int32_t, ndim = 1] indices_np
    #     cdef MICMat indices_MICMat

    #     assert self.shape == indices.shape, 'The shape of the replaced MICMat is ' + `self.shape` + ', but must be ' + `indices.shape` + ', the shape of the indices array.'

    #     if type(indices) == MICMat:
    #         indices_MICMat = indices
    #         indices_MICMat.dtype = 1
    #         cmicmat.slice_inds_replace(indices_MICMat.size, indices_MICMat.A_int, B.A, indices_MICMat.offloaded, B.offloaded, self.A)

    #     elif type(indices) == list or type(indices) == np.ndarray:
    #         indices_np = np.array(indices, dtype = np.int32)
    #         assert indices_np.max() < B.size and indices_np.min() >= 0, 'Indices specified are out of bounds for array of size ' + `B.size` + '.'
    #         A_sliced.A = cmicmat.slice_inds_replace(len(indices), <int *> indices_np.data, B.A, 0, B.offloaded, self.A)

    #     return self

    def __getslice__(self, i, j):
        return self.slice_inds(range(i, j)) 

    @profile_routine
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

    @profile_routine
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
    @profile_routine
    def fill_randn(self, RandGen stream, float mu, float sigma):
        cmicmat.fill_randn(stream.skip_num, self.size, self.A, mu, sigma)
        stream.skip_num = stream.skip_num + self.size
        return self

    @profile_routine
    def fill_uniform(self, float p, RandGen stream):
        cmicmat.fill_uniform(stream.skip_num, self.size, self.A)
        self.scale(p)
        stream.skip_num = stream.skip_num + self.size
        return self

    @profile_routine
    def get_dropout_from_cached_uniform(self, MICMat cached_uniform, float p, MICMat scratch):
        assert cached_uniform.size - self.size - 1 >= 0, 'There exists an array larger than the cached uniform.'
        # ind_start = np.random.randint(cached_uniform.size - self.size - 1)
        ind_start = int(64*(np.random.randint((cached_uniform.size - self.size - 1)/64)))

        cmicmat.replace_partial_mic(scratch.size, 0, scratch.A, scratch.size, ind_start, cached_uniform.A)        

        # convert uniform to Bernoulli, copy to self
        scratch += p
        scratch.astype('int', self)
        
        return self

    @profile_routine
    def fill_uniform_int_diverse(self, shift, MICMat scalings, RandGen stream, MICMat scratch):
        cmicmat.fill_uniform(stream.skip_num, scratch.size, scratch.A)
        stream.skip_num = stream.skip_num + self.size

        scratch.multiply(scalings)
        scratch.update(scalings, -0.5)
        scratch.update(float(shift))
        scratch.floor()

        scratch.astype('int', self)

        return self

    @profile_routine
    def fill_bernoulli(self, RandGen stream, float p, **kwargs):
        cdef int shift_const = kwargs.pop('shift', self.size)
        
        if shift_const < self.size:
            cmicmat.shift_int(self.size - shift_const, shift_const, self.A_int)

        cmicmat.fill_bernoulli(stream.skip_num, shift_const, self.A_int, p)
        stream.skip_num = stream.skip_num + shift_const
        return self

    @profile_routine
    def fill_uniform_int(self, RandGen stream, int i_start, int i_end):
        cmicmat.fill_uniform_int(stream.skip_num, self.size, self.A_int, i_start, i_end)
        stream.skip_num = stream.skip_num + self.size
        return self

    @profile_routine
    def fill_geometric(self, RandGen stream, float p):
        cmicmat.fill_geometric(stream.skip_num, self.size, self.A, p)
        stream.skip_num = stream.skip_num + self.size
        return self

    @profile_routine
    def fill_nested(self, MICMat geometric_samples):
        assert geometric_samples.shape[1] == 1 and geometric_samples.shape[0] == self.shape[0], 'Shape of array ' + `self.shape` + ' and shape of geometric sample vector ' + `geometric_samples.shape` + 'don\'t match.'
        
        cmicmat.fill_nested(self.shape[0], self.shape[1], self.A, geometric_samples.A)
        
        return self

    @profile_routine
    def apply_dropout(self, MICMat dropout_mask):
        assert dropout_mask.shape[1:] == self.shape[1:], 'Shape of array ' + `self.shape` + ' and shape of mask ' + `dropout_mask.shape` + ' don\'t match.'
        # assert dropout_mask.shape == self.shape, 'Shape of array ' + `self.shape` + ' and shape of mask ' + `dropout_mask.shape` + ' don\'t match.'
        
        cmicmat.apply_dropout(self.shape[0], np.product(self.shape[1:]), self.A, dropout_mask.A_int)
        
        return self

    @profile_routine
    def fill_zeros(self):
        if self.dtype == 0:
            cmicmat.fill_zeros(self.size, self.A, self.offloaded)

        elif self.dtype == 1:
            cmicmat.fill_zeros_int(self.size, self.A_int, self.offloaded)

        return self  

    @profile_routine
    def fill_ones(self):
        if self.dtype == 0:
            cmicmat.fill_ones(self.size, self.A, self.offloaded)

        elif self.dtype == 1:
            cmicmat.fill_ones_int(self.size, self.A_int, self.offloaded)

        return self

    @profile_routine
    def fill_const(self, c):
        if self.dtype == 0:
            cmicmat.fill_const(self.size, self.A, <float> c, self.offloaded)

        elif self.dtype == 1:
            cmicmat.fill_const_int(self.size, self.A_int, <int> c, self.offloaded)

        return self


######### string and array outputting
    @profile_routine
    def print_host(self):
        cmicmat.print_slice(self.shape[-2], self.shape[-1], self.A, 0)

    @profile_routine
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

    @profile_routine
    def array(self):
        cdef np.ndarray[np.float32_t, ndim = 2] S = np.zeros([self.ROWS, self.COLS], dtype = np.float32)

        if self.offloaded:
            self.pull_mic()
        
        S.data = <char *> cmicmat.unalign_host(self.size, self.A)
        
        return S

    @profile_routine
    def ndarray(self):
        cdef np.ndarray[np.float32_t, ndim = 1] S1
        cdef np.ndarray[np.float32_t, ndim = 2] S2
        cdef np.ndarray[np.float32_t, ndim = 3] S3
        cdef np.ndarray[np.float32_t, ndim = 4] S4
        cdef np.ndarray[np.float32_t, ndim = 5] S5
        cdef np.ndarray[np.float32_t, ndim = 6] S6

        if self.offloaded:
            self.pull_mic()

        if self.ndim == 1:
            S1 = np.zeros(self.shape, dtype = np.float32)
            S1.data = <char *> cmicmat.unalign_host(self.size, self.A)
            return S1

        elif self.ndim == 2:
            S2 = np.zeros(self.shape, dtype = np.float32)
            S2.data = <char *> cmicmat.unalign_host(self.size, self.A)
            return S2

        elif self.ndim == 3:
            S3 = np.zeros(self.shape, dtype = np.float32)
            S3.data = <char *> cmicmat.unalign_host(self.size, self.A)
            return S3

        elif self.ndim == 4:
            S4 = np.zeros(self.shape, dtype = np.float32)
            S4.data = <char *> cmicmat.unalign_host(self.size, self.A)
            return S4

        elif self.ndim == 5:
            S5 = np.zeros(self.shape, dtype = np.float32)
            S5.data = <char *> cmicmat.unalign_host(self.size, self.A)
            return S5

        elif self.ndim == 6:
            S6 = np.zeros(self.shape, dtype = np.float32)
            S6.data = <char *> cmicmat.unalign_host(self.size, self.A)
            return S6

    @profile_routine
    def float(self):
        assert self.size == 1, 'MICMat size must be 1 to convert to float.'

        if self.offloaded:
            self.pull_mic()
        
        cdef np.float32_t S = cmicmat.output_float(self.A)
        
        return S


######### elementwise operations
    @profile_routine
    def exp(self):
        cmicmat.expo(self.size, self.A)
        return self

    @profile_routine
    def floor(self):
        cmicmat.flooro(self.size, self.A)
        return self

    @profile_routine
    def sign(self):
        cmicmat.sign(self.size, self.A)
        return self
    
    @profile_routine
    def log(self):
        cmicmat.lg(self.size, self.A)
        return self
    
    @profile_routine
    def abs(self):
        cmicmat.abso(self.size, self.A)
        return self
    
    @profile_routine
    def sqrt(self):
        cmicmat.sqrto(self.size, self.A) 
        return self
    
    @profile_routine
    def neg(self):
        cmicmat.scale(self.size, self.A, -1.0)
        return self

    @profile_routine
    def scale(self, float c):
        cmicmat.scale(self.size, self.A, c)
        return self
    
    @profile_routine
    def pow(self, b):
        cmicmat.powo(self.size, self.A, <float> b) 
        return self
    
    @profile_routine
    def invert(self):
        cmicmat.invert(self.size, self.A)
        return self

    # @profile_routine
    # def shift(self, int shift, direction):
    #     # void replace_partial_host(int N_A, int SHIFT_A, float *restrict A, int N_B, int SHIFT_B, float *restrict B){

    # # A[SHIFT_A: N_B] = B[SHIFT_B : N_B];
    #     if direction == 'forward':
    #         remaining_size = self.size - shift
    #         if self.dtype == 1:
            
    #         else:
    #         cmicmat.replace_partial_host(remaining_size, shift, self.A, remaining_size, 0, self.A)

    #     else:
    #         raise NameError('Not implemented for other shift forms.')

    #     return self

    @profile_routine
    def frac_nan(self):
        cdef float f = cmicmat.frac_nan(self.size, self.A)
        return f

    @profile_routine
    def treat_nan(self):
        cdef float f = self.frac_nan()
        if f > 0.:
            print 'WARNING: ignoring ' + `f` + ' fraction of NaN elements.'
            cmicmat.treat_nan(self.size, self.A)

        return self
    
    @profile_routine
    def clip(self, float lower, float upper):
        cmicmat.clip(self.size, self.A, lower, upper)
        return self
    
    @profile_routine
    def clip_low(self, float lower):
        cmicmat.clip_low(self.size, self.A, lower)
        return self


    ######### matrix operations
    @profile_routine
    def horizontal_reflection(self, MICMat should_flip, MICMat scratch):
        N, C, H, W = self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        assert should_flip.size == N, 'Need should_flip size to be exactly ' + `N` + ', the size of the given tensor.'
        
        cmicmat.horizontal_reflection(N, C, H, W, self.A, should_flip.A_int, scratch.A)

    @profile_routine
    def vertical_reflection(self, MICMat should_flip, MICMat scratch):
        N, C, H, W = self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        assert should_flip.size == N, 'Need should_flip size to be exactly ' + `N` + ', the size of the given tensor.'
        
        cmicmat.vertical_reflection(N, C, H, W, self.A, should_flip.A_int, scratch.A)
    
    @profile_routine
    def crop(self, MICMat crop_info, int output_H, int output_W, MICMat outputs):
        N, C, H, W = self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        assert crop_info.shape == (N, 2), 'Need crop_info shape to be exactly ' + `(N, 2)` + '.'
        assert outputs.shape == (N, C, output_H, output_W), 'Need output shape to be exactly ' + `(N, C, output_H, output_W)` + '.'

        cmicmat.crop(N, C, H, W, output_H, output_W, self.A, crop_info.A_int, outputs.A, self.offloaded)

    @profile_routine
    def rgb_to_hsv(self, MICMat outputs, MICMat scratch):
        N, C, H, W = self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        assert C == 3, 'Need inputs to be in RGB tensor format.'
        assert outputs.shape == (N, C, H, W), 'Need output shape to be exactly ' + `(N, C, H, W)` + '.'

        cmicmat.rgb_to_hsv(N, H, W, self.A, outputs.A, scratch.A, self.offloaded)

    @profile_routine
    def hsv_to_rgb(self, MICMat outputs, MICMat scratch):
        N, C, H, W = self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        assert C == 3, 'Need inputs to be in HSV tensor format.'
        assert outputs.shape == (N, C, H, W), 'Need output shape to be exactly ' + `(N, C, H, W)` + '.'

        cmicmat.hsv_to_rgb(N, H, W, self.A, outputs.A, scratch.A, self.offloaded)
    
    @profile_routine
    def perturb_hue(self, MICMat perturbations):
        N, C, H, W = self.shape
        assert C == 3, 'Input must be in image tensor format.'
        assert perturbations.size == N, 'Need exactly one perturbation element per input image.'

        cmicmat.perturb_hue(N, C, H, W, self.A, perturbations.A, self.offloaded)

        return self

    @profile_routine
    def perturb_saturation(self, MICMat scale_perturbations, MICMat shift_perturbations):
        N, C, H, W = self.shape
        assert scale_perturbations.size == N, 'Need exactly one perturbation element per input image.'
        assert shift_perturbations.size == N, 'Need exactly one perturbation element per input image.'
        cmicmat.perturb_saturation(N, C, H, W, self.A, scale_perturbations.A, shift_perturbations.A, self.offloaded)

        return self

    @profile_routine
    def perturb_value(self, MICMat scale_perturbations, MICMat shift_perturbations):
        N, C, H, W = self.shape
        assert scale_perturbations.size == N, 'Need exactly one perturbation element per input image.'
        assert shift_perturbations.size == N, 'Need exactly one perturbation element per input image.'

        if C == 1:
            self.scale(1.0/255.0)

        cmicmat.perturb_value(N, C, H, W, self.A, scale_perturbations.A, shift_perturbations.A, self.offloaded)

        if C == 1:
            self.scale(255.0)

        return self

    @profile_routine
    def response_normalization(self, MICMat inputs, float alpha, float beta, int local_radius):
        N, K, H, W = inputs.shape
        # assert inputs.shape == self.shape, 'Shapes of inputs and outputs must be the same.'
        # cdef int *groups_A_int = self.A_int
        # if local_radius > 0:
        #     assert groups.shape == (N, K), 'Shape of groups matrix must be ' + `(N, K)` + '.'

        # else:
        #     groups_A_int = groups.A_int


        cmicmat.response_normalization(N, K, H, W, inputs.A, self.A, alpha, beta, local_radius)
    
    @profile_routine
    def response_normalization_gradient(self, MICMat inputs, MICMat outputs, MICMat gradient_outputs, float alpha, float beta, int local_radius):
        N, K, H, W = outputs.shape

        cmicmat.response_normalization_gradient(N, K, H, W, inputs.A, outputs.A, self.A, gradient_outputs.A, alpha, beta, local_radius)
    
    @profile_routine
    def get_argamxs(self, MICMat inputs, MICMat argmaxs):
        C, H, W, N = self.shape
        cmicmat.get_argmaxs(N, C, H, W, inputs.A, self.A, argmaxs.A_int)
    
    @profile_routine
    def permute_dimensions(self, dimensions, MICMat output):
        D1, D2, D3, D4 = self.shape
        perm1, perm2, perm3, perm4 = dimensions
        
        if self.dtype == 0:
            cmicmat.permute_dimensions(D1, D2, D3, D4, perm1, perm2, perm3, perm4, self.A, output.A)

        elif self.dtype == 1:
            cmicmat.permute_dimensions_int(D1, D2, D3, D4, perm1, perm2, perm3, perm4, self.A_int, output.A)            

        cdef MICMat reshaped_object
        reshaped_object = output if self.dtype == 0 else self
        reshaped_object.shape = tuple([self.shape[p] for p in dimensions])

        return reshaped_object
    
    @profile_routine
    def interleave_block(self, int block, MICMat output):
        N = self.shape[-1]
        C = np.product(self.shape[0:-1])
        
        if N != block:
            if self.dtype == 0:
                cmicmat.interleave_block(N, C, block, self.A, output.A)

            elif self.dtype == 1:
                cmicmat.interleave_block_int(N, C, block, self.A_int, output.A) 

        else:
            cmicmat.replace_mic(self.size, output.A, self.A)

        cdef MICMat reshaped_object
        reshaped_object = output if self.dtype == 0 else self

        reshaped_object.shape = (N/block,) + self.shape[0:-1] + (block,)

        return reshaped_object
    
    @profile_routine
    def uninterleave_block(self, MICMat output):
        block = self.shape[-1]
        N_block = self.shape[0]
        N = block*N_block
        C = np.product(self.shape[1:-1])

        if N_block != 1:
            if self.dtype == 0:
                cmicmat.uninterleave_block(N, C, block, self.A, output.A)

            elif self.dtype == 1:
                cmicmat.uninterleave_block_int(N, C, block, self.A_int, output.A)

        else:
            cmicmat.replace_mic(self.size, output.A, self.A)

        cdef MICMat reshaped_object
        reshaped_object = output if self.dtype == 0 else self

        reshaped_object.shape = self.shape[1:-1] + (N,)

        return reshaped_object
    
    @profile_routine
    def interleave_for_gradient(self, int block, MICMat output):
        N, C, H, W = self.shape
        
        # if block > 1:
        cmicmat.interleave_for_gradient(N, C, H, W, block, self.A, output.A)

        output.shape = (N, C/block, H, W, block)

        return output
    
    @profile_routine
    def uninterleave_for_gradient(self, MICMat output):
        N, C_block, H, W, block = self.shape

        C = C_block * block
        
        # if block > 1:
        cmicmat.uninterleave_for_gradient(N, C, H, W, block, self.A, output.A)

        output.shape = (N, C, H, W)
        
        return output

    # def convert_data_structure(self, MICMat output, char *target_type, **kwargs):
    #     reshape_only = kwargs.pop('reshape_only', False)

    #     inputs.rotate_dimensions('forward', inputs_rotated)
    #     inputs_rotated.interleave_block(layer['N block'], inputs_interleaved)

    @profile_routine
    def global_averaging(self, MICMat outputs):
        original_shape = self.shape
        assert self.ndim == 4, 'This routine only applies to standard 4D tensors.'
        assert outputs.shape == self.shape[:2]
        N, K, H, W = self.shape
        
        self.shape = (N*K, H*W)
        outputs.shape = (N*K, 1)
        
        outputs.sum_replace(self, 1)
        outputs.scale(1./(H*W))
        # print 'forward'
        # print self.abs_mean()
        # print outputs.abs_mean()
        self.shape = original_shape
        outputs.shape = original_shape[:2]
        
        return outputs

    @profile_routine
    def global_averaging_gradient(self, MICMat gradient_outputs):
        # self is gradient_inputs
        original_shape = self.shape
        assert self.ndim == 4, 'This routine only applies to standard 4D tensors.'
        assert gradient_outputs.shape == self.shape[:2]
        N, K, H, W = self.shape

        self.shape = (N*K, H*W)
        gradient_outputs.shape = (N*K, 1)
        # self.fill_zeros()
        self.update(gradient_outputs)
        self.scale(1./(H*W))

        self.shape = original_shape
        gradient_outputs.shape = original_shape[:2]
        
        # print 'backward'
        # print self.abs_mean()
        # print gradient_outputs.abs_mean()
        return self

    @profile_routine
    def convolution(self, MICMat inputs, MICMat filters, MICMat argmaxs, int stride, int padding, int pooling_radius, int pooling_stride, layer, argmaxs_fixed, MICMat scratch):
        # asserts that check number of dimensions, sizes, etc

        # C, H, W, N = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]
        # K, Y, X = filters.shape[0], filters.shape[2], filters.shape[3]
        # _, Y, X, K = filters.shape
        
        N_block, C, H, W, N_blocksize = inputs.shape
        K_block, _, Y, X, K_blocksize = filters.shape

        N = N_block*N_blocksize
        K = K_block*K_blocksize

        output_H = (H + 2*padding - Y + 1)/stride
        output_W = (W + 2*padding - X + 1)/stride
        pooled_H = int(np.ceil((output_H - pooling_radius + 1.)/pooling_stride))
        pooled_W = int(np.ceil((output_W - pooling_radius + 1.)/pooling_stride))
        
        # assert self.shape == (K, pooled_H, pooled_W, N), 'Output shape is ' + `self.shape` + ' rather than ' + `(K, pooled_H, pooled_W, N)` + '.'
        # assert argmaxs.shape == (K, pooled_H, pooled_W, N), 'Argmax shape is ' + `self.shape` + ' rather than ' + `(K, pooled_H, pooled_W, N)` + '.'

        if layer == 1:
            cmicmat.convolution_layer1(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)
        
        elif layer == 2:
            cmicmat.convolution_layer2(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)

        elif layer == 3:
            cmicmat.convolution_layer3(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)

        elif layer == 4:
            cmicmat.convolution_layer4(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)

        elif layer == 5:
            cmicmat.convolution_layer5(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)

        elif layer == 6:
            cmicmat.convolution_layer6(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)

        elif layer == 7:
            cmicmat.convolution_layer7(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)

        elif layer == 8:
            cmicmat.convolution_layer8(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)

        elif layer == 9:
            cmicmat.convolution_layer9(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)

        elif layer == 10:
            cmicmat.convolution_layer10(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)

    @profile_routine
    def convolution_gradient(self, MICMat inputs, MICMat filters, MICMat argmaxs,
        MICMat gradient_pooled_outputs, MICMat gradient_inputs, 
        int stride, int padding, int pooling_radius, int pooling_stride, 
        int layer, MICMat scratch):
        # self is the filters gradient

        # INPUTS data structure [N, C/C_BLOCK, H, W, C_BLOCK]
        # D_FILTERS/FILTERS data structure [C/C_BLOCK, Y/Y_BLOCK, K, Y_BLOCK, X, C_BLOCK]
        # ARGMAXS/OUTPUTS/D_POOLED_OUTPUTS data structure [N, pooled_H, pooled_W, K]

        N, C_block, H, W, C_blocksize = inputs.shape
        
        # if layer == 2:
        _, Y, K, X, _ = self.shape # [C/C_BLOCK, Y, K, X, C_BLOCK]

        # else:
            # K, Y, X, _ = self.shape


        C = C_block*C_blocksize

        output_H = (H + 2*padding - Y + 1)/stride
        output_W = (W + 2*padding - X + 1)/stride
        pooled_H = int(np.ceil((output_H - pooling_radius + 1.)/pooling_stride))
        pooled_W = int(np.ceil((output_W - pooling_radius + 1.)/pooling_stride))
        
        #assert self.shape == filters.shape, 'Filter shape is ' + self.shape + ' rather than ' + `filters.shape` + '.'
        #assert gradient_inputs.shape == inputs.shape, 'gradient_inputs shape is ' + gradient_inputs.shape + ' rather than ' + `inputs.shape` + '.'
        #assert gradient_pooled_outputs.shape == (K, pooled_H, pooled_W, N), 'gradient_pooled_outputs shape is ' + `gradient_pooled_outputs.shape` + ' rather than ' + `(K, pooled_H, pooled_W, N)` + '.'
        
        if layer == 1:
            cmicmat.convolution_gradient_layer1(N, C, H, W, inputs.A, K, Y, X, padding, filters.A, argmaxs.A_int, gradient_pooled_outputs.A, 
                gradient_inputs.A, self.A, scratch.A)

        elif layer == 2:
            cmicmat.convolution_gradient_layer2(N, C, H, W, inputs.A, K, Y, X, padding, filters.A, argmaxs.A_int, gradient_pooled_outputs.A, 
                gradient_inputs.A, self.A, scratch.A)

        elif layer == 3:
            cmicmat.convolution_gradient_layer3(N, C, H, W, inputs.A, K, Y, X, padding, filters.A, argmaxs.A_int, gradient_pooled_outputs.A, 
                gradient_inputs.A, self.A, scratch.A)

        elif layer == 4:
            cmicmat.convolution_gradient_layer4(N, C, H, W, inputs.A, K, Y, X, padding, filters.A, argmaxs.A_int, gradient_pooled_outputs.A, 
                gradient_inputs.A, self.A, scratch.A)

        elif layer == 5:
            cmicmat.convolution_gradient_layer5(N, C, H, W, inputs.A, K, Y, X, padding, filters.A, argmaxs.A_int, gradient_pooled_outputs.A, 
                gradient_inputs.A, self.A, scratch.A)

        elif layer == 6:
            cmicmat.convolution_gradient_layer6(N, C, H, W, inputs.A, K, Y, X, padding, filters.A, argmaxs.A_int, gradient_pooled_outputs.A, 
                gradient_inputs.A, self.A, scratch.A)

        elif layer == 7:
            cmicmat.convolution_gradient_layer7(N, C, H, W, inputs.A, K, Y, X, padding, filters.A, argmaxs.A_int, gradient_pooled_outputs.A, 
                gradient_inputs.A, self.A, scratch.A)

        elif layer == 8:
            cmicmat.convolution_gradient_layer8(N, C, H, W, inputs.A, K, Y, X, padding, filters.A, argmaxs.A_int, gradient_pooled_outputs.A, 
                gradient_inputs.A, self.A, scratch.A)

        elif layer == 9:
            cmicmat.convolution_gradient_layer9(N, C, H, W, inputs.A, K, Y, X, padding, filters.A, argmaxs.A_int, gradient_pooled_outputs.A, 
                gradient_inputs.A, self.A, scratch.A)

        elif layer == 10:
            cmicmat.convolution_gradient_layer10(N, C, H, W, inputs.A, K, Y, X, padding, filters.A, argmaxs.A_int, gradient_pooled_outputs.A, 
                gradient_inputs.A, self.A, scratch.A)
            
    @profile_routine
    def convolve_and_pool_replace(self, MICMat inputs, MICMat filters, MICMat argmaxs, int stride, int padding, int pooling_radius, int pooling_stride, layer, argmaxs_fixed, MICMat scratch):
        # asserts that check number of dimensions, sizes, etc

        N, C, H, W = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]
        K, Y, X = filters.shape[0], filters.shape[2], filters.shape[3]
        
        K_preshadow = K
        # K = 2*K if shadow else K

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
            # if not shadow:
            cmicmat.convolve_and_pool_layer1(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)

            # else:
                # cmicmat.convolve_and_pool_shadow_layer1(N, C, H, W, inputs.A, K_preshadow, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)                

            #else:
                #cmicmat.convolve_argmaxs_fixed_layer1(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded)
        
        elif layer == 2:
            #if not argmaxs_fixed:
            cmicmat.convolve_and_pool_layer2(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)

            #else:
                #cmicmat.convolve_argmaxs_fixed_layer2(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded)

    @profile_routine
    def local_filtering(self, MICMat inputs, MICMat filters, MICMat argmaxs, int stride, int padding, int pooling_radius, int pooling_stride, layer, argmaxs_fixed, MICMat scratch):
        # asserts that check number of dimensions, sizes, etc

        C, H, W, N = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]
        # K, Y, X = filters.shape[0], filters.shape[2], filters.shape[3]
        _, _, _, Y, X, K = filters.shape

        output_H = (H + 2*padding - Y + 1)/stride
        output_W = (W + 2*padding - X + 1)/stride
        pooled_H = int(np.ceil((output_H - pooling_radius + 1.)/pooling_stride))
        pooled_W = int(np.ceil((output_W - pooling_radius + 1.)/pooling_stride))
        
        # assert self.shape == (K, pooled_H, pooled_W, N), 'Output shape is ' + `self.shape` + ' rather than ' + `(K, pooled_H, pooled_W, N)` + '.'
        # assert argmaxs.shape == (K, pooled_H, pooled_W, N), 'Argmax shape is ' + `self.shape` + ' rather than ' + `(K, pooled_H, pooled_W, N)` + '.'

        times = time.time()
        if layer == 1:
            start_time = time.time()
            #if not argmaxs_fixed:
            cmicmat.local_filtering_layer1(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)

            #else:
                #cmicmat.convolve_argmaxs_fixed_layer1(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded)
        
        # elif layer == 2:
            #if not argmaxs_fixed:
            # cmicmat.local_filtering_layer2(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded, scratch.A)

            #else:
                #cmicmat.convolve_argmaxs_fixed_layer2(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, argmaxs.A_int, stride, padding, pooling_radius, pooling_stride, self.offloaded)

    @profile_routine
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

    @profile_routine
    def fft(self, *args):
        is_forward = True
        frequencies_shape = self.shape[:-2] + (self.shape[-2], 2*(self.shape[-1]/2 + 1))

        cdef MICMat frequencies
        if not args:
            frequencies = MICMat(frequencies_shape).offload_mic()

        else:
            frequencies = args[0]

        assert frequencies.shape == frequencies_shape, 'Expected frequency shape is ' + `frequencies_shape` + '.'
        cmicmat.fft(np.product(self.shape[:-2]), self.shape[-2], self.shape[-1], is_forward, self.A, frequencies.A)

        return frequencies

    @profile_routine
    def ifft(self, MICMat spatials):
        is_forward = False

        assert self.shape == spatials.shape[:-2] + (spatials.shape[-2], 2*(spatials.shape[-1]/2 + 1)), 'Expected frequency shape is ' + `spatials.shape[:-2] + (spatials.shape[-2], 2*(spatials.shape[-1]/2 + 1))` + '.'
        cmicmat.fft(np.product(spatials.shape[:-2]), spatials.shape[-2], spatials.shape[-1], is_forward, spatials.A, self.A)

        return spatials

    @profile_routine
    def fft_full(self, *args):
        is_forward = True
        frequencies_shape = self.shape

        cdef MICMat frequencies
        if not args:
            frequencies = MICMat(frequencies_shape).offload_mic()

        else:
            frequencies = args[0]

        assert frequencies.shape == frequencies_shape, 'Expected frequency shape is ' + `frequencies_shape` + ' (with complex double-counting).'
        cmicmat.fft_full(np.product(self.shape[:-2]), self.shape[-2], self.shape[-1]/2, is_forward, self.A, frequencies.A)

        return frequencies

    @profile_routine
    def ifft_full(self, MICMat spatials):
        is_forward = False

        assert self.shape == spatials.shape, 'Expected frequency shape is ' + `spatials.shape` + ' (with complex double-counting).'
        cmicmat.fft_full(np.product(spatials.shape[:-2]), spatials.shape[-2], spatials.shape[-1]/2, is_forward, spatials.A, self.A)

        return spatials

    @profile_routine
    def low_pass_filter(self, MICMat outputs, int H_out, int W_out):
        # assert 2*(outputs.shape[-2]/2) == outputs.shape[-2], 'Low-pass diameter must be even.'
        outputs.fill_zeros()
        cmicmat.low_pass_filter(np.product(self.shape[:-2]), self.shape[-2], self.shape[-1]/2, H_out, W_out, self.A, outputs.A)

        return self

    @profile_routine
    def low_pass_filter_gradient(self, MICMat gradient_outputs, int H_out, int W_out):
        cmicmat.low_pass_filter_gradient(np.product(self.shape[:-2]), self.shape[-2], self.shape[-1]/2, H_out, W_out, self.A, gradient_outputs.A)

        return self

    @profile_routine
    def wipe_out_irrelevant_entries(self):
        cmicmat.wipe_out_irrelevant_entries(np.product(self.shape[:-2]), self.shape[-2], self.shape[-1]/2, self.A)

        return self

    @profile_routine
    def wipe_out_irrelevant_entries_full(self):
        cmicmat.wipe_out_irrelevant_entries_full(np.product(self.shape[:-2]), self.shape[-2], self.shape[-1]/2, self.A)

        return self

    @profile_routine
    def fft_conjugate_symmetry_scaling(self, spatial_W, MICMat scratch, **kwargs):
        N, K, H = self.shape[:-1]
        cdef int forward = kwargs.pop('forward', 1)
        cmicmat.fft_conjugate_symmetry_scaling(N, K, H, forward, spatial_W, self.A, scratch.A)

        return self

    @profile_routine
    def complex_multiply(self, MICMat inputs, MICMat filters):
        assert self.shape[0] == inputs.shape[0]
        assert self.shape[1] == filters.shape[0]
        assert self.shape[2:] == inputs.shape[2:]
        assert self.shape[2:] == filters.shape[2:]

        N, K, C, H = self.shape[0], self.shape[1], filters.shape[1], filters.shape[-2]
        W = filters.shape[-1]/2
        
        cmicmat.cmult(N, K, C, H*W, inputs.A, filters.A, self.A)

        return self

    @profile_routine
    def complex_multiply_gradient(self, MICMat inputs, MICMat inputs_gradient, MICMat filters, MICMat outputs_gradient):

        N, K, C, H = outputs_gradient.shape[0], outputs_gradient.shape[1], filters.shape[1], filters.shape[-2]
        W = filters.shape[-1]/2

        cmicmat.cmult_gradient(N, K, C, H*W, inputs.A, inputs_gradient.A, filters.A, 
            self.A, outputs_gradient.A)

        return self

    @profile_routine
    def conjugate(self):
        cmicmat.conjugate(self.size/2, self.A)

        return self

    @profile_routine
    def real_and_imaginary(self):
        halved_shape = self.shape[:-1] + (self.shape[-1]/2,)
        
        cdef MICMat real = MICMat(halved_shape).offload_mic()
        cdef MICMat imaginary = MICMat(halved_shape).offload_mic()

        cmicmat.real_and_imaginary(real.size, self.A, real.A, imaginary.A)

        return real, imaginary

    @profile_routine
    def amplitude_and_phase(self):
        halved_shape = self.shape[:-1] + (self.shape[-1]/2,)
        
        cdef MICMat amplitude = MICMat(halved_shape).offload_mic()
        cdef MICMat phase = MICMat(halved_shape).offload_mic()

        cmicmat.amplitude_and_phase(amplitude.size, self.A, amplitude.A, phase.A)

        return amplitude, phase

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

    @profile_routine
    def convolve_gradient(self, MICMat inputs, MICMat filters, MICMat argmaxs,
        MICMat gradient_pooled_outputs, MICMat gradient_inputs, 
        int stride, int padding, int pooling_radius, int pooling_stride, layer, MICMat scratch):
        # self is the filters gradient
        N, C, H, W = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]
        K, Y, X = filters.shape[0], filters.shape[2], filters.shape[3]

        K_preshadow = K
        # K = 2*K if shadow else K

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
            # if shadow:
            #     cmicmat.convolve_gradient_shadow_layer1(N, C, H, W, inputs.A, K_preshadow, Y, X, padding, filters.A, argmaxs.A_int, gradient_pooled_outputs.A, 
            #         gradient_inputs.A, self.A, scratch.A)
                # self *= 2.
            # print self

        elif layer == 2:
            cmicmat.convolve_gradient_layer2(N, C, H, W, inputs.A, K, Y, X, padding, filters.A, argmaxs.A_int, gradient_pooled_outputs.A, 
                gradient_inputs.A, self.A, scratch.A)

    @profile_routine
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

    @profile_routine
    def T(self):
        cdef MICMat S = self.deepcopy()

        cmicmat.T(S.ROWS, S.COLS, S.A)
        S.shape = (self.shape[1], self.shape[0])

        return S

    @profile_routine
    def T_replace(self):
        cmicmat.T(self.ROWS, self.COLS, self.A)
        self.shape = (self.shape[1], self.shape[0])

        return self

    @profile_routine
    def rotate_dimensions(self, direction, MICMat output):
        cdef MICMat reshaped_object

        reshaped_object = output if self.dtype == 0 else self

        if direction == 'forward':
            rows = self.shape[0]
            cols = np.product(self.shape[1:])
            reshaped_object.shape = self.shape[1:] + (self.shape[0],)

        elif direction == 'backward':
            rows = np.product(self.shape[0:-1])
            cols = self.shape[-1]
            reshaped_object.shape = (self.shape[-1],) + self.shape[0:-1]

        if self.dtype == 0:
            cmicmat.transpose_replace(rows, cols, self.A, output.A)

        elif self.dtype == 1:
            cmicmat.transpose_replace_int(rows, cols, self.A_int, output.A)

        return self

    def gram_schmidt(self, MICMat scratch):
        assert self.shape[0] <= np.prod(self.shape[1:]), 'Cannot orthonormalize matrix. Number of vectors greater than matrix rank.'
        cmicmat.gram_schmidt(self.shape[0], np.product(self.shape[1:]), self.A, scratch.A)

        return self

    @profile_routine
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

    @profile_routine
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

    @profile_routine
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

    @profile_routine
    def dot_vec(self, MICMat B):
    # sum(self .* B)
        cdef MICMat S = MICMat()
        S.shape = (1, 1)
        assert self.size == B.size, 'Matrix dimensions don\'t match for dot vector product.'
        S.A = cmicmat.dot_vec(self.size, self.A, B.A)
        S.offloaded = self.offloaded

        return S 

    @profile_routine
    def update(self, V, *args):
        cdef float ALPHA
        cdef MICMat V_MICMat
        if not args:
            ALPHA = 1.0
        else:
            ALPHA = args[0]

        if type(V) == MICMat:
            V_MICMat = V
            if V_MICMat.size > 1:
                assert self.ndim == V_MICMat.ndim, 'Dimensions ' + `self.ndim` + ' and ' + `V_MICMat.ndim` + ' of MICMats do not match in update.'

            a1, a2, a3, a4, a5, a6 = tuple([1]*(6 - len(self.shape))) + self.shape
            b1, b2, b3, b4, b5, b6  = tuple([1]*(6 - len(V_MICMat.shape))) + V_MICMat.shape
            assert (b1 == a1 or b1 == 1) and (b2 == a2 or b2 == 1) and (b3 == a3 or b3 == 1) and (b4 == a4 or b4 == 1) and (b5 == a5 or b5 == 1) and (b6 == a6 or b6 == 1), 'Matrix dimensions ' + `self.shape` + ' and ' + `V_MICMat.shape` + ' don\'t match in update.'
            assert self.offloaded == V_MICMat.offloaded, 'Both MICMats must be either on the host or on the MIC.'

            cmicmat.update(a1, a2, a3, a4, a5, a6, self.A, b1, b2, b3, b4, b5, b6, V_MICMat.A, ALPHA, self.offloaded)

        elif type(V) in {np.float32, np.float64, float}:
            if type(V) == np.float64:
                V = V.astype(np.float32)
            cmicmat.update_const(self.size, self.A, ALPHA*V)

        return self

    @profile_routine
    def update_curve(self, diff, dm1, d0, t, compute):
        a = 0.0001
        if compute == 'curve':
            self.update(d0, t/a)
            
            self.update(d0, 2.*(t/a)**2.)
            self.update(dm1, (t/a)**2.)
            self.update(diff, 3.*(t/a)**2.)

            self.update(d0, (t/a)**3.)
            self.update(dm1, (t/a)**3.)
            self.update(diff, 2.*(t/a)**3.)

        elif compute == 'gradient':
            self.replace(d0)
            self.scale(1./a)
            
            self.update(d0, 4.*(t/a)/a)
            self.update(dm1, 2.*(t/a)/a)
            self.update(diff, 6.*(t/a)/a)

            self.update(d0, 3./a*(t/a)**2.)
            self.update(dm1, 3./a*(t/a)**2.)
            self.update(diff, 6./a*(t/a)**2.)

    @profile_routine
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

    @profile_routine
    def divide(self, V):
        cdef MICMat V_MICMat

        if type(V) == MICMat:
            V_MICMat = V
            if V_MICMat.ndim == 2:
                assert (self.ROWS == V_MICMat.ROWS and self.COLS == V_MICMat.COLS) or (self.ROWS == V_MICMat.ROWS and V_MICMat.COLS == 1) or (V_MICMat.ROWS == 1 and self.COLS == V_MICMat.COLS) or (V_MICMat.ROWS == 1 and V_MICMat.COLS == 1), 'Matrix dimensions ' + `self.shape` + ' and ' + `V.shape` + ' don\'t match in update.'
                cmicmat.divide(self.ROWS, self.COLS, self.A, V_MICMat.ROWS, V_MICMat.COLS, V_MICMat.A)

            else: # multiply elementwise for tensors
                assert self.size == V_MICMat.size, 'Tensors multiplied must have same sizes, and instead have shapes ' + `self.shape` + ' and ' + `V_MICMat.shape` + '.'    
                cmicmat.divide(self.size, 1, self.A, V_MICMat.size, 1, V_MICMat.A)          
        
        elif type(V) == np.float32 or type(V) == np.float64 or type(V) == float or type(V) == int:
            if type(V) == np.float64:
                V = V.astype(np.float32)
            cmicmat.scale(self.size, self.A, 1./V)

        # if type(V) == MICMat:
        #     V_MICMat = V
        #     assert (self.ROWS == V_MICMat.ROWS and self.COLS == V_MICMat.COLS) or (self.ROWS == V_MICMat.ROWS and V_MICMat.COLS == 1) or (V_MICMat.ROWS == 1 and self.COLS == V_MICMat.COLS) or (V_MICMat.ROWS == 1 and V_MICMat.COLS == 1), 'Matrix dimensions ' + `self.shape` + ' and ' + `V.shape` + ' don\'t match in update.'
        #     cmicmat.divide(self.ROWS, self.COLS, self.A, V_MICMat.ROWS, V_MICMat.COLS, V_MICMat.A)
        # elif type(V) == np.float32 or type(V) == np.float64 or type(V) == float or type(V) == int:
        #     if type(V) == np.float64:
        #         V = V.astype(np.float32)
        #     cmicmat.scale(self.size, self.A, 1.0/V)

        return self

    @profile_routine
    def sum(self, *args):         
        cdef MICMat S_MICMat = MICMat()
        cdef int AXIS = 2

        # if not args or args[0] == 2:
        #     S_MICMat.shape = (1, 1)
        #     S_MICMat.A = cmicmat.sum_axis(self.size, 1, self.A, AXIS)
        # else:
        if not not args:
            AXIS = args[0]

        if AXIS == 0:
            S_MICMat.shape = (1, self.COLS)

        if AXIS == 1:
            S_MICMat.shape = (self.ROWS, 1)
        
        if AXIS == 2:
            S = cmicmat.sumo(self.size, self.A)

        else:
            S_MICMat.A = cmicmat.sum_axis(self.ROWS, self.COLS, self.A, AXIS)
            S_MICMat.offloaded = self.offloaded
            S = S_MICMat

        return S

    @profile_routine
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
            
            # self.fill_zeros()
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

    @profile_routine
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
        
        if self.ndim == 2:
            I.A_int = cmicmat.max_axis(self.ROWS, self.COLS, self.A, AXIS)

        elif self.ndim == 4 and AXIS == 2:
            I.A_int = cmicmat.max_axis(self.size, 1, self.A, AXIS)

        else:
            raise NameError('Currently cannot compute max/min for tensor.')
            # assert AXIS == 2
            # I.A_int = cmicmat.max_axis(self.size, 1, self.A, AXIS)            

        I.dtype = 1
        I.offloaded = self.offloaded
        cdef MICMat S = self.slice_inds(I)
        
        if AXIS != 2:
            cmicmat.index_global_to_local(self.ROWS, self.COLS, I.A_int, AXIS)
        
        return S, I

    # def max_and_arg_replace(self, B, *args):         
    #     cdef MICMat I = MICMat()
    #     cdef int AXIS = 2

    #     if not args or args[0] == 2:
    #         I.shape = (1, 1)

    #     else:
    #         AXIS = args[0]
    #         if AXIS == 0:
    #             I.shape = (1, B.COLS)

    #         if AXIS == 1:
    #             I.shape = (B.ROWS, 1)
        
    #     if B.ndim == 2:
    #         I.A_int = cmicmat.max_axis(B.ROWS, B.COLS, B.A, AXIS)

    #     else:
    #         raise NameError('Currently cannot compute max/min for tensor.')
    #         # assert AXIS == 2
    #         # I.A_int = cmicmat.max_axis(self.size, 1, self.A, AXIS)            

    #     I.dtype = 1
    #     I.offloaded = self.offloaded
    #     self.slice_inds_replace(B, I)
        
    #     if AXIS != 2:
    #         cmicmat.index_global_to_local(self.ROWS, self.COLS, I.A_int, AXIS)

    @profile_routine
    def max(self, *args):
        cdef int AXIS = 2
        if not args:
            pass
        else:
            AXIS = args[0]

        S, _ = self.max_and_arg(AXIS)
        return S

    @profile_routine
    def max_elementwise_replace(self, MICMat A, MICMat B):
        self.fill_zeros()
        self += A
        self -= B
        self.abs() # self = |A - B|

        self += A
        self += B # self = |A - B| + A + B

        self.scale(0.5)
        return self

    @profile_routine
    def max_replace(self, MICMat S, *args):
        cdef int AXIS = 2

        if not args or args[0] == 2:
            assert self.shape == (1, 1), 'MICMat shape must be (1, 1).'

        else:
            AXIS = args[0]
            if AXIS == 0:
                assert self.shape == (1, S.COLS), 'MICMat shape must be ' + `(1, S.COLS)` + '.'

            if AXIS == 1:
                assert self.shape == (S.ROWS, 1), 'MICMat shape must be ' + `(S.ROWS, 1)` + '.'

            cmicmat.max_axis_replace(S.ROWS, S.COLS, S.A, AXIS, self.A)

        return self

    @profile_routine
    def min(self, *args):
        cdef int AXIS = 2
        if not args:
            pass
        else:
            AXIS = args[0]

        return self.scale(-1.).max(AXIS).scale(-1.)

    @profile_routine
    def argmax(self, *args):
        cdef int AXIS = 2
        if not args:
            pass
        else:
            AXIS = args[0]

        _, I = self.max_and_arg(AXIS)
        return I

    @profile_routine
    def mean(self, *args):
        cdef int AXIS = 2 
        cdef float SCALING
        cdef MICMat S_MICMat

        if not args or args[0] == 2:
            SCALING = 1.0/(<float> self.size)
            S = self.sum(AXIS)
            S *= SCALING
        else:
            AXIS = args[0]
            if AXIS == 0:
                SCALING = 1.0/(<float> self.ROWS)

            if AXIS == 1:
                SCALING = 1.0/(<float> self.COLS)

            S_MICMat = self.sum(AXIS)        
            S_MICMat.offloaded = self.offloaded
            S_MICMat.scale(SCALING)
            S = S_MICMat

        return S      

    @profile_routine
    def norm(self, *args):
        cdef MICMat S_MICMat
        cdef int AXIS = 2 
        if not args:
            pass
        else:
            AXIS = args[0]
        
        cdef MICMat B = self.deepcopy()
        B.pow(2.0)

        if AXIS == 2:
            S = B.sum(AXIS)
        
        else:
            S_MICMat = B.sum(AXIS)
            S_MICMat.offloaded = self.offloaded
            S_MICMat.sqrt()
            S = S_MICMat

        B.free()
        return S 

    @profile_routine
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

    @profile_routine
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

    def abs_mean(self):
        return self.deepcopy().abs().mean()


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

    @profile_routine
    def equal(self, MICMat V):
        assert self.shape == V.shape, 'The shapes ' + `self.shape` + ' and ' + `V.shape` + ' of the compared MICMats don\'t match.'
        cdef MICMat S = MICMat()
        S.shape = V.shape
        S.offloaded = self.offloaded
        S.A = cmicmat.equal(self.size, self.A, V.A)

        return S

    @profile_routine
    def leq(self, float V):
        cdef MICMat S = MICMat()
        S.shape = self.shape
        S.offloaded = self.offloaded
        S.A = cmicmat.leq(self.size, self.A, V)

        return S

    @profile_routine
    def geq(self, float V):
        cdef MICMat S = MICMat()
        S.shape = self.shape
        S.offloaded = self.offloaded
        S.A = cmicmat.geq(self.size, self.A, V)

        return S

    @profile_routine
    def greater(self, float V):
        cdef MICMat S = MICMat()
        S.shape = self.shape
        S.offloaded = self.offloaded
        S.A = cmicmat.greater(self.size, self.A, V)

        return S

    @profile_routine
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
    @profile_routine
    def labels_to_vectors(self, int K, MICMat targets):
        # cdef MICMat S = MICMat()
        # S.shape = (self.shape[0], K)
        # S.offloaded = self.offloaded
        cmicmat.labels_to_vectors(self.shape[0], K, self.A, targets.A)

    @profile_routine
    def frac_zero(self):
        cdef MICMat zeros = MICMat(self.shape) 
        if self.offloaded:
            zeros.offload_mic()

        zeros.fill_zeros()
        return (self == zeros).sum()/self.size


cdef class Scratch(MICMat):
    cdef list shapes
    cdef list shifts
    cdef int updated
    cdef MICMat end

    def __cinit__(self, *args):
        self.shapes = []
        self.shifts = [0]
        self.updated = False

    def __setattr__(self, name, value):
        if name == 'end':
            self.end = value

        else:
            MICMat.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == 'shapes':
            return self.shapes

        elif name == 'contents_size':
            return self.shifts[-1]

        elif name == 'end':
            if not self.updated:
                self.update_end()

            return self.end

        else:
            return MICMat.__getattr__(self, name)

    @profile_routine
    def reset(self, *args):
        cdef MICMat S

        if not args:
            self.shapes = []
            self.shifts = [0]

        else:
            if type(args[0]) == MICMat:
                self.shapes = []
                self.shifts = [0]
                S = self.append(args[0])

            elif type(args[0]) == tuple:
                self.shapes = [args[0]]
                self.shifts = [0, np.product(args[0])]
                S = self.get(0)

            return S

    @profile_routine
    def append(self, args, **kwargs):
        should_fill_zeros = kwargs.pop('fill_zeros', False)
        cdef MICMat V
        self.updated = False

        if type(args) == MICMat:
            V = args
            assert self.offloaded == V.offloaded, 'Input MICMat must be offloaded to the same resource as the scratch!'
            
            if self.offloaded:
                cmicmat.replace_partial_mic(self.size, self.contents_size, self.A, V.size, 0, V.A)

            else:
                cmicmat.replace_partial_host(self.size, self.contents_size, self.A, V.size, 0, V.A)

            shape = V.shape
            new_size = V.size

        elif type(args) == tuple:
            shape = args
            new_size = np.product(shape)
            
        assert self.size >= self.contents_size + new_size, 'The scratch space of size ' + `self.size` + ' is out of memory with contents of size ' + `self.contents_size` + ' and incoming object of size ' + `new_size` + '!'
        self.shapes.append(shape)
        self.shifts.append(self.shifts[-1] + new_size)

        cdef MICMat S = self.get(len(self.shapes) - 1)
        if should_fill_zeros:
            S.fill_zeros()
        return S

    def update_end(self):
        self.end = self.get()
        self.updated = True

    @profile_routine
    def get(self, *args):
        cdef MICMat S = MICMat()
        
        if not args: # get pointer to end of scratch
            S.shape = (self.size - self.contents_size, 1)
            shift = self.contents_size

        else: 
            index = args[0]
            S.shape = self.shapes[index]
            shift = self.shifts[index]

        S.offloaded = self.offloaded 
        S.persist = 1
        S.A = cmicmat.get_partial(S.size, shift, self.A)
        return S

    # @profile_routine
    # def contents_size(self, index):
    #     # return sum([np.product(shape) for shape in self.shapes[:index]])
    #     if index == 0:
    #         return 0
    #     else:
    #         return self.shifts[index-1]

# def initialize_locks():
#     cdef OmpLock lock
#     lock = cmicmat.initialize_locks()
#     return lock

def ping_each_core():
        print 'Pinging each core on the MIC.'
        cmicmat.ping_each_core()

def check_mic_status():
    cmicmat.check_mic_status()

def tester():
    cmicmat.tester()