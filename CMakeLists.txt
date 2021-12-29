cmake_minimum_required(VERSION 3.16)

project(onnxrunime_example CXX)

add_executable(${PROJECT_NAME} src/main.cpp)
if(UNIX)
    target_include_directories(${PROJECT_NAME} PRIVATE onnxrunime/linux/include)
    target_link_directories(${PROJECT_NAME} PRIVATE onnxrunime/linux/lib)
elseif(WIN32)
    target_include_directories(${PROJECT_NAME} PRIVATE onnxrunime/win/include)
    target_link_directories(${PROJECT_NAME} PRIVATE onnxrunime/win/lib)
endif()

target_link_libraries(${PROJECT_NAME}
    PRIVATE onnxruntime
    PRIVATE onnxruntime_providers_shared 
    PRIVATE onnxruntime_providers_tensorrt
    PRIVATE onnxruntime_providers_cuda
)