from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "qs",  # Name of the module
        ["qs.cpp"],  # Source file(s)
        include_dirs=["/opt/anaconda3/envs/faiss_env/include"],  # Include directories
        library_dirs=["/opt/anaconda3/envs/faiss_env/lib"],  # Library directories
        libraries=["faiss"],  # Libraries to link
        extra_compile_args=["-std=c++17"],  # C++ standard
        extra_link_args=["-Wl,-rpath,/opt/anaconda3/envs/faiss_env/lib"],  # Add rpath
    )
]

# Setup configuration
setup(
    name="qs",
    version="0.1",
    description="A Pybind11 wrapper for fast QuickShift clustering algorithm",
    ext_modules=ext_modules,
    zip_safe=False,  # Ensure the module is not zipped
)

# python setup.py build_ext --inplace
# g++ -O3 -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` \
#     qs.cpp -o qs`python3-config --extension-suffix` \
#     -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -lfaiss -Wl,-rpath,$CONDA_PREFIX/lib