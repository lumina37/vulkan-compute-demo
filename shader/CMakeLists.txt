set(_VKC_HLSL_SRCS
        gaussFilter
        grayscale
)
set(_VKC_SPIRV_SHADERS)

function(vkc_add_hlsl2spirv name)
    set(__VKC_IN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/hlsl/${name}.hlsl)
    set(__VKC_OUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/spirv/${name}.hlsl.h)
    list(APPEND _VKC_SPIRV_SHADERS ${__VKC_OUT_PATH})
    set(_VKC_SPIRV_SHADERS ${_VKC_SPIRV_SHADERS} PARENT_SCOPE)
    add_custom_command(
            OUTPUT ${__VKC_OUT_PATH}
            COMMAND ${Vulkan_dxc_EXECUTABLE} -T cs_6_0 -spirv -Fh ${__VKC_OUT_PATH} ${__VKC_IN_PATH}
            MAIN_DEPENDENCY ${__VKC_IN_PATH}
            COMMENT "Compile shader - ${name}"
    )
endfunction()

foreach (__VKC_HLSL_SRC ${_VKC_HLSL_SRCS})
    vkc_add_hlsl2spirv(${__VKC_HLSL_SRC})
endforeach ()

add_custom_target(_vkc-shader DEPENDS ${_VKC_SPIRV_SHADERS})
add_library(vkc-shader INTERFACE)
add_dependencies(vkc-shader _vkc-shader)
target_include_directories(vkc-shader INTERFACE .)
