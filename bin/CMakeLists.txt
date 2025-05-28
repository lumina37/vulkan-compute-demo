function(vkc_add_executable name src)
    add_executable(${name} ${src})

    target_link_libraries(${name} PRIVATE vkc::lib::static vkc::shader)

    vkc_copy_dlls_if_needed(${name})
endfunction()

vkc_add_executable(vkc-bin-gaussFilter-v0-baseline gaussFilter/v0-baseline.cpp)
vkc_add_executable(vkc-bin-gaussFilter-v1-separate gaussFilter/v1-separate.cpp)
vkc_add_executable(vkc-bin-gaussFilter-v2-sampling gaussFilter/v2-sampling.cpp)

vkc_add_executable(vkc-bin-gaussFilter-storage-image-as-input gaussFilter/storage-image-as-input.cpp)
vkc_add_executable(vkc-bin-gaussFilter-process-with-roi gaussFilter/process-with-roi.cpp)

vkc_add_executable(vkc-bin-grayscale-after-gauss grayscale/grayscale-after-gauss.cpp)
