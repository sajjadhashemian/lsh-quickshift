from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
# import skimage

setup(
    ext_modules=cythonize(
        "_quickshift_cy.pyx", compiler_directives={"language_level": "3"}, annotate=True
    ),
    include_dirs=["/opt/anaconda3/envs/faiss_env/include", np.get_include()],  # Include directories
    library_dirs=["/opt/anaconda3/envs/faiss_env/lib"],  # Library directories
)

# python setup_quickshift.py build_ext --inplace