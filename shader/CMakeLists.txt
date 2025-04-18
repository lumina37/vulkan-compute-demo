set(_VKC_GLSL_SRCS
        gaussFilterV0
        gaussFilterV1
        gaussFilterV2
)
set(_VKC_SPIRV_SHADERS)

function(vkc_add_glsl2spirv name)
    set(__VKC_IN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/glsl/${name}.comp)
    set(__VKC_OUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/spirv/${name}.h)
    list(APPEND _VKC_SPIRV_SHADERS ${__VKC_OUT_PATH})
    set(_VKC_SPIRV_SHADERS ${_VKC_SPIRV_SHADERS} PARENT_SCOPE)
    add_custom_command(
            OUTPUT ${__VKC_OUT_PATH}
            COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V ${__VKC_IN_PATH} --vn spirvCode -o ${__VKC_OUT_PATH}
            MAIN_DEPENDENCY ${__VKC_IN_PATH}
            COMMENT "Compile shader - ${name}"
    )
endfunction()

foreach (__VKC_GLSL_SRC ${_VKC_GLSL_SRCS})
    vkc_add_glsl2spirv(${__VKC_GLSL_SRC})
endforeach ()

add_custom_target(_vkc-shader DEPENDS ${_VKC_SPIRV_SHADERS})
add_library(vkc-shader INTERFACE)
add_dependencies(vkc-shader _vkc-shader)
target_include_directories(vkc-shader INTERFACE .)
