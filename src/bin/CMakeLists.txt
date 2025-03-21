function(vkc_add_executable name src)
    add_executable(${name} ${src})

    target_link_libraries(${name} PRIVATE vkc-lib)

    copy_dlls_if_needed(${name})
endfunction()

vkc_add_executable(vkc-bin-gaussianBlur main-gaussianBlur.cpp)
add_dependencies(vkc-bin-gaussianBlur vkc-shader-gaussianBlur)
