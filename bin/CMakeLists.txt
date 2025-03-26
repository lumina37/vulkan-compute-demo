function(vkc_add_executable name src)
    add_executable(${name} ${src})

    target_link_libraries(${name} PRIVATE vkc-lib vkc-shader)

    copy_dlls_if_needed(${name})
endfunction()

vkc_add_executable(vkc-bin-gaussFilter-texture main-gaussFilter-texture.cpp)

vkc_add_executable(vkc-bin-gaussAndGrayscale main-gaussAndGrayscale.cpp)
