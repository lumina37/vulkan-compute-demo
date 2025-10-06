enable_testing()

set(_VKC_TEST_TARGETS)

function(vkc_add_test name lib srcs)
    add_executable(${name} ${srcs})
    target_link_libraries(${name} PRIVATE
            Catch2::Catch2WithMain
            ${lib}
            vkc::shader
    )

    catch_discover_tests(${name})
    list(APPEND _VKC_TEST_TARGETS ${name})
    set(_VKC_TEST_TARGETS ${_VKC_TEST_TARGETS} PARENT_SCOPE)

    vkc_copy_dlls_if_needed(${name})
endfunction()

vkc_add_test(test_gauss_filter vkc::lib::static "test_gauss_filter.cpp")
vkc_add_test(test_sgemm_simt vkc::lib::static "test_sgemm_simt.cpp")
vkc_add_test(test_sgemm_tcore vkc::lib::static "test_sgemm_tcore.cpp")
vkc_add_test(test_flash_attention2 vkc::lib::static "test_flash_attention2.cpp")
vkc_add_test(test_prefix_sum vkc::lib::static "test_prefix_sum.cpp")
vkc_add_test(test_topk vkc::lib::static "test_topk.cpp")

vkc_add_test(test_gauss_filter_header_only vkc::lib::header-only "test_gauss_filter.cpp")

add_custom_target(vkc-tests DEPENDS ${_VKC_TEST_TARGETS})
