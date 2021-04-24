import fsdl_lib as fl

from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open('requirements.txt') as f:
    install_requirements = f.read().splitlines()

test_requirements = []

setup(
    name=fl.__name__,
    version=fl.__version__,
    url='https://github.com/tspthomas/fsdl2021_project',
    author=fl.__author__,
    description='Library with general Python scripts for the FSDL2021 project.',
    install_requires=install_requirements,
    tests_require=test_requirements,
    packages=find_packages(exclude=['docs', 'tests']),
    zip_safe=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
)
