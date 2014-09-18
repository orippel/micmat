# Copyright (c) 2014, Oren Rippel and Ryan P. Adams
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension(
    "micmat_wrap",
    sources=[os.environ["SPECIFIC_MICMAT_PATH"] + "micmat_wrap.c"],
        # extra_objects=["micmat.o"], #, "micmatMIC.o"
        extra_link_args = ["-Wl,-rpath=" + os.environ["SPECIFIC_MICMAT_PATH"]],
        # library_dirs = ["/global/homes/r/rippel/sota/micmat"],
        library_dirs = [os.getcwd()],
        libraries=["micmat"],
        # include_dirs = [numpy.get_include(), "/global/homes/r/rippel/sota/micmat"]
        include_dirs = [numpy.get_include(), os.getcwd()]
    )]


setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    # build_temp = os.environ['SPECIFIC_MICMAT_PATH']
    )