set(_VKC_HLSL_SHADERS
        gaussianBlur.hlsl
        grayscale.hlsl
)
set(_VKC_SPIRV_SHADERS)

function(vkc_add_shader src)
    set(__VKC_IN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${src})
    set(__VKC_SPIRV_HEADER_DIR ${PROJECT_SOURCE_DIR}/src/include/vkc/_spirv)
    set(__VKC_OUT_PATH ${__VKC_SPIRV_HEADER_DIR}/${src}.h)
    list(APPEND _VKC_SPIRV_SHADERS ${__VKC_OUT_PATH})
    set(_VKC_SPIRV_SHADERS ${_VKC_SPIRV_SHADERS} PARENT_SCOPE)
    add_custom_command(
            OUTPUT ${__VKC_OUT_PATH}
            COMMAND ${Vulkan_dxc_EXECUTABLE} -T cs_6_0 -spirv -Fh ${__VKC_OUT_PATH} ${__VKC_IN_PATH}
            MAIN_DEPENDENCY ${__VKC_IN_PATH}
            COMMENT "Compile shader - ${__VKC_IN_PATH}"
    )
endfunction()

foreach (__VKC_HLSL_SHADER ${_VKC_HLSL_SHADERS})
    vkc_add_shader(${__VKC_HLSL_SHADER})
endforeach ()

add_custom_target(vkc-shader-gaussianBlur DEPENDS ${_VKC_SPIRV_SHADERS})
