function(vkc_add_executable name src)
    add_executable(${name} ${src})

    target_link_libraries(${name} PRIVATE vkc-lib vkc-shader)

    copy_dlls_if_needed(${name})
endfunction()

vkc_add_executable(vkc-bin-gaussFilter-v1 main-gaussFilter-v1.cpp)

vkc_add_executable(vkc-bin-gaussAndGrayscale main-gaussAndGrayscale.cpp)
