function(vkc_add_executable name src)
    add_executable(${name} ${src})

    target_link_libraries(${name} PRIVATE
            vkc-lib
            vkc-shader
    )

    vkc_copy_dlls_if_needed(${name})
endfunction()

vkc_add_executable(vkc-bin-gaussFilter-v0-baseline main-gaussFilter-v0-baseline.cpp)
vkc_add_executable(vkc-bin-gaussFilter-v1-separate main-gaussFilter-v1-separate.cpp)
vkc_add_executable(vkc-bin-gaussFilter-v2-sampling main-gaussFilter-v2-sampling.cpp)
