enable_testing()

set(_VKC_TEST_TARGETS)

function(vkc_add_test name srcs)
    add_executable(${name} ${srcs})
    target_link_libraries(${name} PRIVATE
            Catch2::Catch2WithMain
            vkc-lib
            vkc-shader
    )

    catch_discover_tests(${name})
    list(APPEND _VKC_TEST_TARGETS ${name})
    set(_VKC_TEST_TARGETS ${_VKC_TEST_TARGETS} PARENT_SCOPE)

    vkc_copy_dlls_if_needed(${name})
endfunction()

vkc_add_test(test_gaussian_filter "test_gaussian_filter.cpp")

add_custom_target(vkc-tests DEPENDS ${_VKC_TEST_TARGETS})
