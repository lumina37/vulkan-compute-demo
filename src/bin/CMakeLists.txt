function(vkc_add_executable name src)
    add_executable(${name} ${src})

    target_include_directories(${name} PRIVATE ${stb_SOURCE_DIR})

    target_link_libraries(${name} PRIVATE vkc-lib)

    target_compile_definitions(${name} PRIVATE
            STB_IMAGE_IMPLEMENTATION
            STB_IMAGE_WRITE_IMPLEMENTATION
    )

    copy_dlls_if_needed(${name})
endfunction()

vkc_add_executable(vkc-bin-gaussianBlur main-gaussianBlur.cpp)
add_dependencies(vkc-bin-gaussianBlur vkc-shader-gaussianBlur)
