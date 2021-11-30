cmake_minimum_required(VERSION 3.16)

project(cpp_agtool)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp -g")

include_directories("./agtool/cm/include")

# load all source code to a variable.
file(GLOB SAMPLING_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/agtool/cm/lib/sampling/*.cpp)
# file(GLOB OTHER_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/agtool/cm/lib/*.cpp)
# generate lib[LIBNAME].[so|lib|dylib] to be linked to source code.
add_library(cpp_agtool SHARED ${SAMPLING_SRCS})

# make install 
# generate /usr/local/lib/lib[LIBNAME].[so|lib|dylib]
install(
	TARGETS cpp_agtool
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
# generate /usr/local/agtool/cm/include/[LIBNAME].hpp
install(
	DIRECTORY "${CMAKE_SOURCE_DIR}/agtool/cm/include" # source directory
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" # target directory
	FILES_MATCHING # install only matched files
	PATTERN "*.hpp" # select header files
)

# generate excutable file.
# ipython
# >>> from agtool.downlaod import ml_100k
# >>> ml_100k()
add_executable(TestNegative test/cpp/test_negative.cpp)
target_link_libraries(TestNegative cpp_agtool)
