cmake_minimum_required(VERSION 3.1)
project(binary_image_clustering)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -Wall -Wextra")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -Wall -Wextra")

find_package(OpenCV REQUIRED)
find_package(Boost 1.55 COMPONENTS system filesystem REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIR})

set(SOURCE_FILES
	main.cpp
)
add_executable(binary-image-clustering ${SOURCE_FILES})

target_link_libraries(binary-image-clustering
	${OpenCV_LIBS}
	${Boost_LIBRARIES}
)
