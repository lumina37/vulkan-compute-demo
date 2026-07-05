add_custom_target(_vkc-shader
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/compile_shaders.py"
            --glslang "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}"
        COMMENT "Compiling shaders"
)

file(GLOB_RECURSE _VKC_SHADER_SRCS "*.cpp")

add_library(vkc-shader STATIC ${_VKC_SHADER_SRCS})
add_dependencies(vkc-shader _vkc-shader)
target_compile_features(vkc-shader PUBLIC cxx_std_23)
target_include_directories(vkc-shader PUBLIC .)
add_library(vkc::shader ALIAS vkc-shader)
