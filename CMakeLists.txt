cmake_minimum_required(VERSION 3.0.2)

project(ObjectTracking)
set(PROJECT_VERSION_STRING "1.0.0")
set(PROJECT_TARGET ${PROJECT_NAME})
set(PROJECT_INCLUDE_PREFIX ${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif ()
string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWERCASE)
if (${CMAKE_BUILD_TYPE_LOWERCASE} MATCHES release)
    SET(CMAKE_CXX_FLAGS "-O2")
endif ()

# opencv
set(OpenCV_DIR /usr/local/share/OpenCV/)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# include
include_directories(
    ${PROJECT_SOURCE_DIR}/include/
)

# add library from source files
file(GLOB_RECURSE SRC_FILES src/*.cpp)
add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
set(ALL_INSTALL_TARGETS ${PROJECT_NAME} ${ALL_INSTALL_TARGETS})

# add executable
add_executable(demo_${PROJECT_NAME} main.cpp)
target_link_libraries(demo_${PROJECT_NAME} ${PROJECT_NAME} ${VisualPerception_LIBRARY})

include(cmake/installConfig.cmake)
