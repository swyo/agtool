#!bin/bash
import setuptools

with open('README.md', 'r') as fin:
    long_description = fin.read()

setuptools.setup(
    name='agtool',
    version='0.0.1',
    author='swyo',
    author_email='l22491360@gmail.com',
    description='Algorithm tools for python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/swyo/agtool',
    install_requires=[],
    keywords=['python', 'packaging'],
    python_requires='>=3.9',
    packages=setuptools.find_packages('.', exclude=('docs', 'tests')),
    zip_safe=False,
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
