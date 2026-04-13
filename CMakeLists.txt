cmake_minimum_required(VERSION 3.10)
project(InfodifTrafficAnalysis)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})  # for json.hpp

# onnxruntime
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib)

# tracker lib
add_library(CentroidTracker CentroidTracker.cpp)
target_link_libraries(CentroidTracker ${OpenCV_LIBS})

# detector lib
add_library(YoloDetector YoloDetector.cpp)
target_link_libraries(YoloDetector ${OpenCV_LIBS} onnxruntime)

# main
add_executable(main main.cpp)
target_link_libraries(main CentroidTracker YoloDetector ${OpenCV_LIBS} Threads::Threads onnxruntime)
