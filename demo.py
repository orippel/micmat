#Copyright (c) 2014, Oren Rippel and Ryan P. Adams
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import micmat_wrap as mm
import numpy as np

# This file is a quick demo file that is meant to test the working condition of the package.
# A more comprehensive description of the available routines and their modes of operation 
# can be found online on the repository page.
def main():
    
    mm.check_mic_status()

    # casting an array from NumPy to MICMat
    A_np = np.random.randn(2000, 2000)
    A = mm.MICMat(A_np)

    # offloading the array to the MIC
    A.offload_mic()
    
    # compute the absolute value
    A.abs() 

    # compute the exp()
    A.exp() 

    # copy A as B
    B = A.deepcopy()
    
    # compute the dot product of A and B and stick it in allocated memory C
    C = A.dot(B)
    
    # if memory for the product already exists, reuse the memory for it
    C.dot_replace(A, B)

    # array broadcasting. A.mean(0) returns a row vector where each element is the mean along the corresponding column
    B = (A - A.mean(0)) / A.std(0)

    print 'File ran successfully.'

if __name__ == '__main__':
    main()
