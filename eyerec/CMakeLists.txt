cmake_minimum_required (VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# For add_cython_target
find_package(Cython REQUIRED)
# For python_extension_module
find_package(PythonExtensions REQUIRED)

find_package(NumPy REQUIRED)
find_package(OpenCV REQUIRED)

add_cython_target(_eyerec CXX PY3)
add_library(_eyerec MODULE ${_eyerec})
python_extension_module(_eyerec)

target_link_libraries(_eyerec eyerec::eyerec ${OpenCV_LIBS})
include_directories(${NumPy_INCLUDE_DIRS})

install(TARGETS _eyerec LIBRARY DESTINATION eyerec)
