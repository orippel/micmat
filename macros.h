// Copyright (c) 2014, Oren Rippel and Ryan P. Adams
// All rights reserved.

// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#ifndef MIC_DEV
#define MIC_DEV 0
#define ALIGN (64)
#endif

#define ALLOC alloc_if(1) free_if(0)
#define FREE alloc_if(0) free_if(1)
#define ALLOC_FREE alloc_if(1) free_if(1)
#define REUSE alloc_if(0) free_if(0)

#define md(a, b) ((a) - ((b)*((a)/(b))))
#define mn(a, b) (((a) > (b)) ? (b) : (a))
#define mx(a, b) (((a) > (b)) ? (a) : (b))
#define ti2(a, b, B) ((a)*(B) + (b))
#define ti3(a, b, c, B, C) ((a)*(B)*(C) + (b)*(C) + (c))
#define ti(a, b, c, d, B, C, D) ((a)*(B)*(C)*(D) + (b)*(C)*(D) + (c)*(D) + (d))
#define ti5(a, b, c, d, e, B, C, D, E) ((a)*(B)*(C)*(D)*(E) + (b)*(C)*(D)*(E) + (c)*(D)*(E) + (d)*(E) + (e))
#define ti6(a, b, c, d, e, f, B, C, D, E, F) ((a)*(B)*(C)*(D)*(E)*(F) + (b)*(C)*(D)*(E)*(F) + (c)*(D)*(E)*(F) + (d)*(E)*(F) + (e)*(F) + (f))
#define ti7(a, b, c, d, e, f, g, B, C, D, E, F, G) ((a)*(B)*(C)*(D)*(E)*(F)*(G) + (b)*(C)*(D)*(E)*(F)*(G) + (c)*(D)*(E)*(F)*(G) + (d)*(E)*(F)*(G) + (e)*(F)*(G) + (f)*(G) + (g))

#define it(s, i, dim, B, C, D) do{ \
    if (dim == 0) s = i / (B*C*D); \
    else if (dim == 1) s = (i % (B*C*D)) / (C*D); \
    else if (dim == 2) s = (i % (C*D)) / D; \
    else if (dim == 3) s = i % D; \
} while(0)