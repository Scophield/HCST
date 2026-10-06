"""Compatibility for older laboratory setuptools with --no-build-isolation."""
from setuptools import setup
setup(name='fully-analog-reram',version='0.1.0',license='MIT',author='Scophield',
      description='Source-informed HCST computational framework for fully analog ReRAM inference',
      package_dir={'':'src'},packages=['reram'],python_requires='>=3.9',
      install_requires=['numpy>=1.23,<3'],
      extras_require={'test':['pytest>=7'],'legacy':['torch>=1.12']},
      entry_points={'console_scripts':['reram=reram.cli:main']},
      url='https://github.com/Scophield/HCST')
