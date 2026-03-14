"""Build script for Cython extensions.

Usage::

    pip install cython
    python setup_cython.py build_ext --inplace
"""

from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
except ImportError:
    raise SystemExit(
        "Cython is required to build native extensions.\n"
        "Install it with: pip install cython"
    )

extensions = [
    Extension(
        "pyaccelerate._native._fast_deque",
        ["_fast_deque.pyx"],
    ),
]

setup(
    name="pyaccelerate-native",
    ext_modules=cythonize(extensions, language_level="3"),
)
