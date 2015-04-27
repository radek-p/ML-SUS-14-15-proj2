cmake_minimum_required(VERSION 3.1)
project(binary_image_clustering)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wall -Wextra")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES
        main.cpp
)
add_executable(binary-image-clustering ${SOURCE_FILES})

target_link_libraries(binary-image-clustering
        ${OpenCV_LIBS}
)
