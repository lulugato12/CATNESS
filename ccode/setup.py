from setuptools import setup, Extension

setup(
    #...
    ext_modules=[Extension('mim', ['mim.cpp'],),],
)
