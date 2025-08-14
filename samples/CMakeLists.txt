function(vkc_add_sample name src)
    add_executable(${name} ${src})

    target_link_libraries(${name} PRIVATE vkc::lib::static vkc::shader)

    vkc_copy_dlls_if_needed(${name})
endfunction()

vkc_add_sample(vkc-bin-gaussFilter-v0 gaussFilter/v0-baseline.cpp)
vkc_add_sample(vkc-bin-gaussFilter-v1 gaussFilter/v1-separate.cpp)
vkc_add_sample(vkc-bin-gaussFilter-grayscale-after-gauss gaussFilter/grayscale-after-gauss.cpp)

vkc_add_sample(vkc-bin-grayscale-common grayscale/common.cpp)
vkc_add_sample(vkc-bin-grayscale-storage-image-as-input grayscale/storage-image-as-input.cpp)
vkc_add_sample(vkc-bin-grayscale-process-with-roi grayscale/process-with-roi.cpp)

vkc_add_sample(vkc-bin-sgemm-v0 sgemm/v0-naive.cpp)
vkc_add_sample(vkc-bin-sgemm-v1 sgemm/v1-block-tile.cpp)
vkc_add_sample(vkc-bin-sgemm-v2 sgemm/v2-thread-tile.cpp)

vkc_add_sample(vkc-bin-gui gui.cpp)
vkc_add_sample(vkc-bin-perf perf.cpp)
