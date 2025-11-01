"""
LuminaAI Dataset Accelerator - Setup Script
Builds C++/CUDA extensions for high-performance data loading
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

# Check for CUDA
def check_cuda():
    """Check if CUDA is available."""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True)
        if result.returncode == 0:
            print("✓ CUDA detected:")
            print(result.stdout.split('\n')[3])  # Version line
            return True
    except FileNotFoundError:
        pass
    
    print("✗ CUDA not detected - building CPU-only version")
    return False

CUDA_AVAILABLE = check_cuda()

# Compiler flags
extra_compile_args = {
    'cxx': ['-std=c++14', '-O3', '-march=native', '-fopenmp'],
    'nvcc': ['-std=c++14', '-O3', '--use_fast_math', '-lineinfo']
}

extra_link_args = ['-fopenmp']

# Source files
sources = [
    'dataset_accelerator/bindings.cpp',
    'dataset_accelerator/core.cpp'
]

# Add CUDA sources if available
if CUDA_AVAILABLE:
    sources.append('dataset_accelerator/cuda_ops.cu')
    extra_compile_args['cxx'].append('-DCUDA_AVAILABLE')
    extra_link_args.extend(['-lcudart', '-lcuda'])
    print(f"Building with CUDA support")
else:
    print(f"Building CPU-only version")

# Define extension
ext_modules = [
    Extension(
        'dataset_accelerator._core',
        sources=sources,
        include_dirs=[
            '/usr/local/cuda/include' if CUDA_AVAILABLE else '',
        ],
        library_dirs=[
            '/usr/local/cuda/lib64' if CUDA_AVAILABLE else '',
        ],
        extra_compile_args=extra_compile_args['cxx'],
        extra_link_args=extra_link_args,
        language='c++'
    )
]

# Custom build to handle CUDA
class CustomBuildExt(build_ext):
    def build_extensions(self):
        # Compiler-specific options
        ct = self.compiler.compiler_type
        
        if ct == 'unix':
            for ext in self.extensions:
                # Add OpenMP flag
                if '-fopenmp' not in ext.extra_compile_args:
                    ext.extra_compile_args.append('-fopenmp')
                if '-fopenmp' not in ext.extra_link_args:
                    ext.extra_link_args.append('-fopenmp')
        
        # Build CUDA sources separately if needed
        if CUDA_AVAILABLE:
            self._build_cuda()
        
        build_ext.build_extensions(self)
    
    def _build_cuda(self):
        """Compile CUDA sources with nvcc."""
        print("Compiling CUDA sources...")
        
        cuda_sources = [s for s in self.extensions[0].sources if s.endswith('.cu')]
        
        for cuda_source in cuda_sources:
            obj_file = cuda_source.replace('.cu', '.o')
            
            cmd = [
                'nvcc',
                '-c',
                cuda_source,
                '-o', obj_file,
                '-std=c++14',
                '-O3',
                '--compiler-options', '-fPIC',
                '--use_fast_math',
                '-DCUDA_AVAILABLE'
            ]
            
            print(f"  Compiling {cuda_source}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"CUDA compilation failed:")
                print(result.stderr)
                raise RuntimeError("CUDA compilation failed")
            
            # Add object file to linker
            self.extensions[0].extra_objects.append(obj_file)
        
        print("✓ CUDA compilation successful")

setup(
    name='dataset_accelerator',
    version='1.0.0',
    description='High-performance dataset operations for LuminaAI',
    author='MatN23',
    packages=['dataset_accelerator'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': CustomBuildExt},
    install_requires=[
        'numpy>=1.20.0',
        'torch>=2.0.0',
    ],
    python_requires='>=3.8',
    zip_safe=False,
)