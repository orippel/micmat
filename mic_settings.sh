# Copyright (c) 2014, Oren Rippel and Ryan P. Adams
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

export COMPOSER_ROOT=/global/babbage/nsg/opt/intel/composerxe/composer_xe_2013_sp1.0.080
export MICMAT_PATH=${PWD}
export PATH=~/anaconda/bin:$PATH
export PYTHONPATH=$PYTHONPATH:$MICMAT_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MICMAT_PATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COMPOSER_ROOT/mkl/lib/intel64
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/mic/coi/host-linux-release/lib
# export MIC_LD_LIBRARY_PATH=$MIC_LD_LIBRARY_PATH:$MKLROOT/lib/mic


source $COMPOSER_ROOT/bin/compilervars.sh intel64 
source $MKLROOT/bin/mklvars.sh mic lp64
source /opt/intel/impi/4.1.1.036/bin64/mpivars.sh

# export OFFLOAD_REPORT=2
export MIC_ENV_PREFIX=PHI
export PHI_OMP_NUM_THREADS=236
export OFFLOAD_INIT=on_start
export PHI_USE_2MB_BUFFERS=16K
# export PHI_KMP_AFFINITY="granularity=fine,balanced"
export PHI_KMP_AFFINITY="granularity=fine,compact"