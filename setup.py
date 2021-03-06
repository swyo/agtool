#!bin/bash
from setuptools import find_packages
from setuptools import setup, Extension

from Cython.Build import cythonize

import numpy as np

with open('README.md', 'r') as fin:
    long_description = fin.read()

install_requires = [
    'cython',
    'numpy',
    'torch>=1.10.0'
]

ext = [
    Extension(
        "agtool.cm.sampling",
        [
            "./agtool/cm/sampling/cython_negative.pyx",
            "./agtool/cm/lib/sampling/negative.cpp"
        ],
        include_dirs=['./agtool/cm/include'] + [np.get_include()],
        libraries=['gomp'],
        language='c++',
        extra_compile_args=['-fopenmp', '-std=c++17']
    )
]

setup(
    name='agtool',
    version='0.0.1',
    author='swyo',
    author_email='l22491360@gmail.com',
    description='Algorithm tools for python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/swyo/agtool',
    install_requires=install_requires,
    keywords=['python', 'packaging'],
    python_requires='>=3.9',
    packages=find_packages('.', exclude=('docs', 'tests')),
    zip_safe=False,
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    ext_modules=cythonize(ext)
)
