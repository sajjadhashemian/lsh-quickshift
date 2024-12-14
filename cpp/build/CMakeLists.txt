cmake_minimum_required(VERSION 3.4)
project(qs)

# Find Python and pybind11
find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)

# Include directories
include_directories(
    ${pybind11_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIRS}
    /opt/anaconda3/envs/faiss_env/include
)

# Define the library
add_library(faiss SHARED IMPORTED)
set_target_properties(faiss PROPERTIES
    IMPORTED_LOCATION /opt/anaconda3/envs/faiss_env/lib
)

# Define the extension module
pybind11_add_module(qs qs.cpp)

# Link the library
target_link_libraries(qs PRIVATE faiss)