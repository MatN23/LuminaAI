"""
LuminaAI Dataset Accelerator - Build System
Cross-platform C++/CUDA compilation with automatic fallback
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            print("CMake not found. Installing CPU-only version...")
            self.build_cpu_only()
            return

        for ext in self.extensions:
            self.build_extension(ext)

    def build_cpu_only(self):
        """Build CPU-only version without CMake"""
        try:
            from pybind11.setup_helpers import Pybind11Extension, build_ext
            
            ext_modules = [
                Pybind11Extension(
                    "dataset_accelerator._core",
                    ["dataset_accelerator/core.cpp", "dataset_accelerator/bindings.cpp"],
                    extra_compile_args=['-O3', '-march=native'] if platform.system() != 'Windows' else ['/O2'],
                    language='c++'
                ),
            ]
            
            build_ext.run(self)
            print("✓ Built CPU-only acceleration")
        except Exception as e:
            print(f"Warning: Could not build C++ extensions: {e}")
            print("Falling back to pure Python implementation")

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Detect CUDA
        cuda_available = self.check_cuda()
        
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCUDA_AVAILABLE={"ON" if cuda_available else "OFF"}',
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}']
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
            build_args += ['--', '-j4']

        # macOS specific
        if platform.system() == "Darwin":
            cmake_args += ['-DCMAKE_OSX_DEPLOYMENT_TARGET=10.14']

        env = os.environ.copy()
        env['CXXFLAGS'] = f"{env.get('CXXFLAGS', '')} -DVERSION_INFO=\\'{self.distribution.get_version()}\\'"
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        try:
            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
            print(f"✓ Built with {'CUDA' if cuda_available else 'CPU-only'} acceleration")
        except subprocess.CalledProcessError as e:
            print(f"Warning: CMake build failed: {e}")
            print("Attempting CPU-only build...")
            self.build_cpu_only()

    def check_cuda(self):
        """Check if CUDA is available"""
        try:
            subprocess.check_output(['nvcc', '--version'])
            return True
        except (OSError, subprocess.CalledProcessError):
            return False

setup(
    name='dataset_accelerator',
    version='1.0.0',
    author='LuminaAI',
    description='High-performance dataset operations for transformer training',
    long_description='',
    ext_modules=[CMakeExtension('dataset_accelerator._core')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.19.0',
        'torch>=1.9.0',
    ],
    extras_require={
        'build': ['cmake>=3.18', 'pybind11>=2.6.0'],
    },
)