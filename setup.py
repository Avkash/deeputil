from setuptools import setup
from setuptools import find_packages


setup(name='Deeputil',
      version='0.0.1',
      description='Deep Learning Utilities for Python',
      author='Avkash Chauhan',
      author_email='avkash@gmail.com',
      url='https://github.com/Avkash/deeputil',
      download_url='https://github.com/Avkash/deeputil/download/tarball/0.0.1/',
      license='Apache License 2.0',
      install_requires=['keras'],
      extras_require={
          'h5py': ['h5py'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov'],
      },
      keywords='deeplearning keras utilities development tools',
      packages=find_packages())
