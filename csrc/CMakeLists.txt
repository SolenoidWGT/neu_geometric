cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
# 项目名称：csrc, 开启CUDA选项
project(csrc CXX CUDA)
# 指定cuda版本
SET(CUDA_VERSION 10.1)
# cmake寻找cuda
FIND_PACKAGE(CUDA ${CUDA_VERSION} REQUIRED)
# 检测是否找到
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)
# C++和CUDA的编译参数，可选。如果CMAKE_CXX_FLAGS和CUDA_NVCC_FLAGS有相同参数会报错
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;)
# 头文件路径
# INCLUDE_DIRECTORIES(
#    ${PROJECT_SOURCE_DIR}/cpu
#    ${PROJECT_SOURCE_DIR}/cuda
# )


# cmake寻找python包
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})


# 指定torch包的路径
# set(Torch /home/wgt/libtorch)
# 将torch包加入编译搜索路径
# set(Torch_DIR /home/wgt/libtorch)
# set(DCMAKE_PREFIX_PATH /home/wgt/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# 将当前目录下所有的源文件加入到DIR_SRCS变量
aux_source_directory(. DIR_SRCS)

# 指定生成目标，生成可执行文件
cuda_add_executable(${PROJECT_NAME} ${DIR_SRCS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)


# 添加连接库torch
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
# 连接库python
target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES})
# 连接库cuda
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${CUDA_LIBRARIES})



# set(Python home/wgt/anaconda3/envs/dgl/include)
# FIND_PACKAGE(Python REQUIRED NO_MODULE PATHS /home/wgt/anaconda3/envs/dgl/include  NO_DEFAULT_PATH)


# 连接cuda外部库cudalib
# TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${CUDA_cudadevrt_LIBRARY})

# set_property(TARGET example-app PROPERTY CXX_STANDARD 14)








# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)
