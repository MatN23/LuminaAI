from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUDA architecture for your GPU
# T4 = sm_75, A100 = sm_80, V100 = sm_70
cuda_arch = os.environ.get('TORCH_CUDA_ARCH_LIST', '7.5;8.0')

setup(
    name='moe_cuda_ops',
    ext_modules=[
        CUDAExtension(
            name='moe_cuda_ops',
            sources=[
                'moe_cuda_ops.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode', f'arch=compute_75,code=sm_75',  # T4
                    '-gencode', f'arch=compute_80,code=sm_80',  # A100
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)