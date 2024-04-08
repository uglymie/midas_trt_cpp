cmake_minimum_required(VERSION 3.10)
project(midas_trt_cpp VERSION 0.1.0 LANGUAGES C CXX CUDA)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/nvidia_common.cmake)

message(*********************************************************)
message(STATUS TensorRT_ROOT = ${TensorRT_ROOT})
message(*********************************************************)

message(STATUS ${NVIDIA_ALL_LIBS})
file(GLOB PRO_SRC 
  ${TensorRT_ROOT}/samples/common/logger.cpp 
  ${TensorRT_ROOT}/samples/common/sampleUtils.cpp
  ${TensorRT_ROOT}/samples/common/sampleOptions.cpp 
  )

set(OpenCV_DIR "D:\\opencv-4.5.4-cuda\\install")

find_package(OpenCV REQUIRED)

include_directories(${INCLUDE_DRIS}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(include 
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/libwebsockets_x64-windows/include
    ${PROJECT_SOURCE_DIR}/openssl_x64-windows/include
)

FILE(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS FOLLOW_SYMLINKS
    include/*.h
    include/*.hpp
)
FILE(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS FOLLOW_SYMLINKS src/*.cpp)

add_executable(midas_trt_cpp ${HEADERS} ${SOURCES} ${PRO_SRC})

# 设置编译选项
target_compile_options(midas_trt_cpp PRIVATE "/NODEFAULTLIB:LIBCMT")

target_link_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/libwebsockets_x64-windows/lib
    ${PROJECT_SOURCE_DIR}/openssl_x64-windows/lib
)

target_link_libraries(midas_trt_cpp ${NVIDIA_ALL_LIBS} ${OpenCV_LIBRARIES})
target_link_libraries(midas_trt_cpp websockets )

