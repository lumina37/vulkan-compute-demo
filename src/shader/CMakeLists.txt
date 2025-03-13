function(vkc_add_shader src)
    set(__VKC_SPIRV_HEADER_DIR ${PROJECT_SOURCE_DIR}/src/include/vkc/_spirv)
    set(__VKC_IN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${src})
    set(__VKC_OUT_PATH ${__VKC_SPIRV_HEADER_DIR}/${src}.h)
    add_custom_command(
            OUTPUT ${src}
            COMMAND ${Vulkan_dxc_EXECUTABLE} -T cs_6_0 -spirv -Fh ${__VKC_OUT_PATH} ${__VKC_IN_PATH}
            MAIN_DEPENDENCY ${__VKC_IN_PATH}
            COMMENT "Compile shader - ${__VKC_IN_PATH}"
    )
endfunction()

set(_VKC_SHADERS
        gaussianBlur.hlsl
)
foreach (_VKC_SHADER ${_VKC_SHADERS})
    vkc_add_shader(${_VKC_SHADER})
endforeach ()

add_custom_target(vkc-shader-gaussianBlur DEPENDS gaussianBlur.hlsl)
