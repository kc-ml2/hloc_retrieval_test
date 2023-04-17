file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/ipa_room_segmentation/*.cpp
)

add_library(ipa_room_segmentation STATIC ${SRC_FILES})

target_link_libraries(ipa_room_segmentation PUBLIC
	${OpenCV_LIBS}
	${Boost_LIBS}
)

target_include_directories(ipa_room_segmentation PUBLIC
    ${CMAKE_SOURCE_DIR}/include
	${OpenCV_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
)

target_compile_options(ipa_room_segmentation PRIVATE)
