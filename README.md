# MICMat

MICMat (MIC Matrix) is a package that enables interfacing with Intel's Xeon Phi Coprocessor in a convenient way, directly from pure Python. Some of the features it offers include:

* **Efficient computation**: MICMat is optimized for high performance. Behind the scenes it interfaces with Intel's MKL (BLAS, VSL, VML, and so on), and takes into account parallelism considerations.
* **Object-oriented development in pure Python**: The MICMat objects are very easy to use. They are manipulated with syntax very similar to [NumPy](http://www.numpy.org/)'s, with additional magical knobs for interaction with the MIC.
* **Offload management**: All routines are designed to work either on the host or on the MIC. It is very straightforward to control data transfer between the host and the MIC, and data persistence is fully supported.
* **Automatic memory management**: MICMat offers automatic garbage collection: it is compatible with the Python garbage collection routines. However, it also offers explicit memory management and reuse, which allows avoiding expensive memory allocation and transfer.

* **Extensive matrix and tensor manipulation library**: Various matrix and multidimensional array operations are supported and are optimized for speed.

This package is under development. Please let us know of any bugs you come across.


# Installation
## Dependencies
MICMat requires:

* Python 2.7. The [Anaconda](http://continuum.io/downloads) distribution is recommended
* Intel Composer XE 2013 SP 1


## Compilation
From within the micmat sub-directory, build the package:  
```    
. micmat_build.sh
```

## Importing into Python
In order import MICMat into Python and to maximize performance on Xeon Phi, various environment variables and paths must be configured in advance. These are all automatically set by the file `mic_settings.sh` with the command
```
source $MICMAT_PATH/mic_settings.sh
```
where `$MICMAT_PATH` contains the path to the package. It may be convenient to add this command to your bash profile. 

In order to import the package into Python, simply add the command
```
import micmat_wrap as mm
```
to the top of your pure Python file.


# Usage
The file `demo.py` contains a few examples of MICMat object manipulation. After successful installation of the package, you should be able to run this file without any problems. 

## Initialization and casting
To create an empty array of size MxN:
```
A = mm.MICMat((M, N))
```

To cast an array from NumPy to MICMat:
```
A_np = np.random.randn(2000, 2000)
A = mm.MICMat(A_np)
```

To cast a matrix from MICMat back to NumPy:
```
A_np = A.array()
```

## Attributes
MICMat objects have attributes very similar to NumPy. For example:
```
A.print()
A.shape
A.size
A.ndim
```
These in addition have the offloaded attribute, which specifies whether the object is on the host or on the MIC:
```
A.offloaded
```

## Offload management
It is very straightforward to offload arrays to the MIC:
```
A.offload_mic()
```
A similar command is used to pull arrays back to the host:
```
A.offload_host()
```

## Memory management
The package has allocation and deallocation routines both for the host and for the MIC, which have been integrated with the Python garbage collector. Hence, memory management is done automatically. When a MICMat array of a particular size is created, a corresponding float array is allocated.

To manually free an array:
```
A.free()
```
Since memory allocation can be costly, whenever possible, the package will avoid allocating new memory. This often means that arrays must be explicitly copied before they are manipulated in order to preserve the structure of the original objects:
```
A.deepcopy()
```
It is often desirable to reuse memory, and so one array that has already been allocated can be replaced with another array of the same shape:
```
A.replace(B)
```
Here, `A` was filled with the contents of `B`.

## Array manipulation
Reshaping an array:
```
A.reshape(shape)
```
where `shape` is a tuple. This requires the original and the reshaped array to have the same size.

A similar command can be used to flatten an array:
```
A.flatten()
```

Given a set `I` (MICMat, NumPy array or a list) containing *linear* indices of array `A`, the routine
```
A.slice_inds(I)
```
returns an array containing the elements of `A` corresponding to these indices, in the shape of `I`. 

Specifically for matices, there exist routines that return copies of array slices across different dimensions:
```
A.slice_rows(I)
A.slice_cols(I)
```
Here, `I` is an index set corresponding to the appropriate axis. 



## Mathematical functions
### Elementwise operations
All elementwise operations by default operate on the original array itself without copying it first. Also, all routines return a pointer to the array they operated on. This is useful when one wants to perform a sequence of operations in a single line, such as `A.pow(2.0).neg().exp()`.

Fill the array with all zeros or all ones:
```
A.fill_zeros()
A.fill_ones()
```

Absolute value:
```
A.abs()
```
Alternatively, `abs(A)` returns an absolute-valued copy of `A`.

Exponentiation and logarithmication:
```
A.exp()
A.log()
```

Power of `c` (where c is a constant):
```
A.pow(c)
```

Reciprocation:
```
A.invert()
```

Negation:
```
A.neg()
```
Note that this can also be done with the command `-A`, but this returns a negated *copy* of `A` without modifying the original. 

Floor (round down to nearest integer):
```
A.floor()
```

Signum:
```
A.sign()
```

Square root:
```
A.sqrt()
```

Scale by a constant `c`:
```
A.scale(c)
```
Alternatively, `c * A` returns a copy of `A` scaled by `c`. 

Clip to range [a, b]:
```
A.clip(a, b)
```

Clip from below to `a`:
```
A.clip_low(a)
```

### Random number generation
Random number generation is slightly (but only slightly) more complicated for MICMat objects. Before random numbers are generated, a *stream* must be initialized:
```
stream = mm.RandGen()
```
This must then be supplied to the random function generators to keep track of the location of the generators along ths stream. 

To fill `A` with independent uniform [0, 1] continuous random variables:
```
A.fill_uniform(stream)
```

To fill `A` with independent Bernoulli discrete random variables with success probability `p`:
```
A.fill_bernoulli(stream, p)
```

To fill `A` with independent and identically-distributed normals with mean `mu` and variance `sigma`:
```
A.fill_randn(stream, mu, sigma)
```

### Array operations
Transposition:
```
A.T()
```
This returns a transposed copy of `A`.

Summation:
```
A.sum()
A.sum(ax)
```
The former returns the sum of all the elements of `A`, while the latter returns an array summed along the `ax` axis. 

Similar expressions can be used to compute various other properties of interest.

Mean:
```
A.mean()
A.mean(ax)
```

Standard deviation:
```
A.std()
A.std(ax)
```

Variance:
```
A.var()
A.var(ax)
```

Maxima:
```
A.max()
A.max(ax)
```

Minima:
```
A.min()
A.min(ax)
```

Arguments of maxima:
```
A.argmax()
A.argmax(ax)
```

2-norm:
```
A.norm()
A.norm(ax)
```

### Broadcasting and operations with more than one array
Most if not all operations that involve more than a single array perform automatic broadcasting. For example, let `A` be an array of size MxN, and let `B` be an array of size `Mx1` or `1xN` or simply a single float. 

Array addition:
```
A += B
```
This can also be done with 
```
A.update(B)
```
which returns the pointer to `A`. The latter is patricularly useful when one wants to perform a sequence of operations in a single line, such as `A.update(B).exp()`. Moreover, the command `A+B` returns a new array containing the sum of `A` and `B`.

Array subtraction:
```
A -= B
```
The command `A-B` returns a new array which contains the difference of `A` and `B`. 

Array multiplication and division:
```
A *= B
A /= B
```
This can also be done with `A.multiply(B)` and `A.divide(B)`. The commands `A * B` and `A / B` return newly-allocated arrays containing the corresponding computations. 


### Comparisons
The commands
```
A > c
A >= c
A < c
A <= c
A == c
A != c
```
compare MICMat `A` to float `c`. They return a binary matrix where each element corresponds to the elementwise evaluation of the appropriate condition.




### Matrix products
To initialize `C` as the matrix product of matrices `A` and `B`:
```
C = A.dot(B)
```

If `C` is already initialized to be of the appropriate dimensions, then memory for it does not need to be allocated, but can be rather reused:
```
C.dot_replace(A, B)
```

If you want to preserve the contents of `C` and want to add to them the product of `A` and `B`:
```
C.dot_replace_update(A, B)




### Convolutions
Let `inputs` be a 4-dimensional tensor of shape (N, C, H, W) corresponding to N examples, each of C channels, height H and width W. Let `filters` be a 4-dimensional tensor of shape (K, C, Y, X) corresponding to K filters, each of C channels, height Y and width X. The command
```
outputs.convolve_and_pool_replace(inputs, argmaxs, filters, pool_radius, stride)
```
first computes the (valid set) convolution of `inputs` and `filters` with stride `stride` over `inputs`, after which it applies max-pooling with pooling radius `pool_radius`. The result is placed in the tensor `outputs` which must be of dimensions (N, K, H_pooled, W_pooled) where H_pooled = ceil((H - Y + 1)/pool_radius) and W_pooled = ceil((W - X + 1)/pool_radius).


# License (BSD 3-Clause)
Copyright (c) 2014, Oren Rippel and Ryan P. Adams

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.