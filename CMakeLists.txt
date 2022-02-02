cmake_minimum_required(VERSION 3.16)

project(onnxruntime_example CXX)

set(CONAN_SYSTEM_INCLUDES ON)
set(CONAN_CMAKE_SILENT_OUTPUT ON)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/${PROJECT_NAME})
if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
    file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/master/conan.cmake" "${CMAKE_BINARY_DIR}/conan.cmake" TLS_VERIFY ON)
endif()
include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_run(CONANFILE conanfile.py BUILD missing)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
find_package(OpenCV)

add_executable(${PROJECT_NAME} src/main.cpp)
if(UNIX)
    target_include_directories(${PROJECT_NAME} PRIVATE onnxruntime/linux-gpu/include)
    target_link_directories(${PROJECT_NAME} PRIVATE onnxruntime/linux-gpu/lib)
elseif(WIN32)
    target_include_directories(${PROJECT_NAME} PRIVATE onnxruntime/windows/include)
    target_link_directories(${PROJECT_NAME} PRIVATE onnxruntime/windows/lib)
endif()

target_link_libraries(${PROJECT_NAME}
    PRIVATE opencv::core
    PRIVATE opencv::dnn
    PRIVATE opencv::imgproc
    PRIVATE opencv::imgcodecs
    PRIVATE onnxruntime
    # PRIVATE onnxruntime_providers_shared 
    # PRIVATE onnxruntime_providers_tensorrt
    # PRIVATE onnxruntime_providers_cuda
)