from conans import ConanFile, CMake

class OnnxRuntimeExample(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake_find_package"
    requires = "opencv/4.5.3"
        