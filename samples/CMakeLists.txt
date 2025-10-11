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

vkc_add_sample(vkc-bin-sgemm-simt-v0 sgemm/simt/v0-naive.cpp)
vkc_add_sample(vkc-bin-sgemm-simt-v1 sgemm/simt/v1-block-tile.cpp)
vkc_add_sample(vkc-bin-sgemm-simt-v2 sgemm/simt/v2-thread-tile-shm.cpp)
vkc_add_sample(vkc-bin-sgemm-simt-v3 sgemm/simt/v3-thread-tile-reg.cpp)
vkc_add_sample(vkc-bin-sgemm-simt-v4 sgemm/simt/v4-double-buffer.cpp)
vkc_add_sample(vkc-bin-sgemm-simt-v5 sgemm/simt/v5-float4.cpp)
vkc_add_sample(vkc-bin-sgemm-simt-v6 sgemm/simt/v6-bank-conflict.cpp)
vkc_add_sample(vkc-bin-sgemm-simt-v7 sgemm/simt/v7-hide-s2g.cpp)

vkc_add_sample(vkc-bin-sgemm-tcore-v0 sgemm/tcore/v0-naive.cpp)
vkc_add_sample(vkc-bin-sgemm-tcore-v1 sgemm/tcore/v1-block-tile.cpp)
vkc_add_sample(vkc-bin-sgemm-tcore-v2 sgemm/tcore/v2-wrap-tile.cpp)
vkc_add_sample(vkc-bin-sgemm-tcore-v3 sgemm/tcore/v3-double-buffer.cpp)
vkc_add_sample(vkc-bin-sgemm-tcore-v4 sgemm/tcore/v4-bank-conflict.cpp)
vkc_add_sample(vkc-bin-sgemm-tcore-v5 sgemm/tcore/v5-coopmat2.cpp)

vkc_add_sample(vkc-bin-gui gui.cpp)
vkc_add_sample(vkc-bin-perf perf.cpp)
