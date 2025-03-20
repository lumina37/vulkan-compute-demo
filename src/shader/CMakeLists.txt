set(_VKC_SPIRV_HEADER_DIR ${PROJECT_SOURCE_DIR}/src/include/vkc/_spirv)

function(vkc_add_shader src)
    set(__VKC_IN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${src})
    set(__VKC_OUT_PATH ${_VKC_SPIRV_HEADER_DIR}/${src}.h)
    add_custom_command(
            OUTPUT ${__VKC_OUT_PATH}
            COMMAND ${Vulkan_dxc_EXECUTABLE} -T cs_6_0 -spirv -Fh ${__VKC_OUT_PATH} ${__VKC_IN_PATH}
            MAIN_DEPENDENCY ${__VKC_IN_PATH}
            COMMENT "Compile shader - ${__VKC_IN_PATH}"
    )
endfunction()

set(_VKC_HLSL_SHADERS
        gaussianBlur.hlsl
)
foreach (__VKC_HLSL_SHADER ${_VKC_HLSL_SHADERS})
    vkc_add_shader(${__VKC_HLSL_SHADER})
endforeach ()

file(GLOB_RECURSE _VKC_SPIRV_SHADERS ${_VKC_SPIRV_HEADER_DIR}/*.h)
add_custom_target(vkc-shader-gaussianBlur DEPENDS ${_VKC_SPIRV_SHADERS})
