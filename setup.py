from setuptools import setup, find_packages

setup(
    name="immatch",
    version='0.1.0',
    install_requires=[
        'transforms3d',
        'tqdm',
        'pyyaml',
        'einops',
        'kornia',
        'yacs',
        'pillow',
        'h5py',
    ],
    packages=find_packages(),
    author='Qunjie Zhou',
    python_requires='>=3.7',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',        
    ],
    license='MIT',    
    keywords='image feature matching',
)