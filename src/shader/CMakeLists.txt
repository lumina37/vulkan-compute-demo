function(vkc_add_shader src)
    set(__VKC_IN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${src})
    set(__VKC_OUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${src}.spv)
    add_custom_command(
            OUTPUT ${src}
            COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V ${__VKC_IN_PATH} -o ${__VKC_OUT_PATH}
            MAIN_DEPENDENCY ${__VKC_IN_PATH}
            COMMENT "Compile shader - ${__VKC_IN_PATH}"
    )
endfunction()

set(_VKC_SHADERS
        gaussianBlur.comp
)
foreach (_VKC_SHADER ${_VKC_SHADERS})
    vkc_add_shader(${_VKC_SHADER})
endforeach ()

add_custom_target(vkc-shader-gaussianBlur DEPENDS gaussianBlur.comp)
