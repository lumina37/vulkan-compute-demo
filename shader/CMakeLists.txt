set(_VKC_GLSL_SRCS
        gaussFilter/v0
        gaussFilter/v1
        grayscale/ro
        grayscale/rw
        sgemm/v0
)
set(_VKC_SPIRV_SHADERS)

function(vkc_add_glsl2spirv name)
    set(__VKC_IN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/glsl/${name}.comp)
    set(__VKC_OUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/spirv/${name}.h)
    list(APPEND _VKC_SPIRV_SHADERS ${__VKC_OUT_PATH})
    set(_VKC_SPIRV_SHADERS ${_VKC_SPIRV_SHADERS} PARENT_SCOPE)
    add_custom_command(
            OUTPUT ${__VKC_OUT_PATH}
            COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V ${__VKC_IN_PATH} --vn code --target-env vulkan1.1 -o ${__VKC_OUT_PATH}
            MAIN_DEPENDENCY ${__VKC_IN_PATH}
            COMMENT "Compile shader - ${name}"
    )
endfunction()

foreach (__VKC_GLSL_SRC ${_VKC_GLSL_SRCS})
    vkc_add_glsl2spirv(${__VKC_GLSL_SRC})
endforeach ()

add_custom_target(_vkc-shader DEPENDS ${_VKC_SPIRV_SHADERS})

file(GLOB_RECURSE _VKC_SHADER_SRCS "*.cpp")

add_library(vkc-shader STATIC ${_VKC_SHADER_SRCS})
add_dependencies(vkc-shader _vkc-shader)
target_compile_features(vkc-shader PUBLIC cxx_std_23)
target_include_directories(vkc-shader PUBLIC .)
add_library(vkc::shader ALIAS vkc-shader)
