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

vkc_add_test(test_gaussFilter vkc::lib::static "test_gaussFilter.cpp")
vkc_add_test(test_sgemm vkc::lib::static "test_sgemm.cpp")
vkc_add_test(test_sgemm_tensor_core vkc::lib::static "test_sgemm_tensor_core.cpp")
vkc_add_test(test_flash_attention2 vkc::lib::static "test_flash_attention2.cpp")

vkc_add_test(test_gaussFilter_header_only vkc::lib::header-only "test_gaussFilter.cpp")

add_custom_target(vkc-tests DEPENDS ${_VKC_TEST_TARGETS})
