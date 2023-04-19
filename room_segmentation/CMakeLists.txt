cmake_minimum_required(VERSION 3.11)
project(
	room_segmentation
	LANGUAGES CXX
)

add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread filesystem)

add_subdirectory(lib)

add_executable(run_room_segmentation main.cpp)

target_link_libraries(run_room_segmentation PUBLIC
	ipa_room_segmentation
	${OpenCV_LIBS}
	${Boost_LIBRARIES}
)

target_include_directories(run_room_segmentation PUBLIC
	${CMAKE_SOURCE_DIR}/include
	${OpenCV_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
)
