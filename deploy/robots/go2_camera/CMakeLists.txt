cmake_minimum_required(VERSION 3.12)
project(go2_camera_controller)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)
find_package(realsense2 QUIET)
find_package(OpenCV QUIET COMPONENTS highgui imgproc)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-aarch64-1.22.0")
endif()

include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx
  /usr/local/include/iceoryx/v2.0.2
  include
  ${ONNXRUNTIME_DIR}/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_libraries(
  unitree_sdk2 ddsc ddscxx rt pthread
  libboost_program_options.a libyaml-cpp.a fmt
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.22.0
)

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})

if(realsense2_FOUND)
  target_compile_definitions(${PROJECT_NAME}_lib PUBLIC GO2_CAMERA_HAS_REALSENSE=1)
  if(TARGET realsense2::realsense2)
    target_link_libraries(${PROJECT_NAME}_lib realsense2::realsense2)
  else()
    target_link_libraries(${PROJECT_NAME}_lib realsense2)
  endif()
else()
  message(WARNING "librealsense2 not found. go2_camera will build with zero-vector camera fallback.")
endif()

if(OpenCV_FOUND)
  target_compile_definitions(${PROJECT_NAME}_lib PUBLIC GO2_CAMERA_HAS_OPENCV=1)
  target_include_directories(${PROJECT_NAME}_lib PUBLIC ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME}_lib ${OpenCV_LIBS})
else()
  message(WARNING "OpenCV not found. go2_camera will build without camera window visualization.")
endif()

link_libraries(${PROJECT_NAME}_lib)

add_executable(go2_camera_ctrl main.cpp)
add_executable(go2_camera_viewer camera_viewer.cpp)
