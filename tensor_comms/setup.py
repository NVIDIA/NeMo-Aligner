# setup.py

from setuptools import setup, find_packages

setup(
    name='tensor_comms',         # Your package name
    version='0.1.0',           # Version number
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=[         # List your dependencies here
        # 'some_dependency>=1.0',
    ],
    author='NVIDIA',
    author_email='sahilj@nvidia.com',
    description='tensor comms for local sharing',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.8',
)
