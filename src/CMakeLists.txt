function(vkc_common_lib_setting name)
    get_target_property(__TYPE ${name} TYPE)
    if (${__TYPE} STREQUAL "INTERFACE_LIBRARY")
        set(__PUB_DEP_SCOPE "INTERFACE")
    else ()
        set(__PUB_DEP_SCOPE "PUBLIC")
    endif ()

    target_compile_features(${name} ${__PUB_DEP_SCOPE} cxx_std_23)

    target_include_directories(${name} ${__PUB_DEP_SCOPE}
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${stb_SOURCE_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_link_libraries(${name} ${__PUB_DEP_SCOPE} Vulkan::Vulkan)

    if (VKC_VERBOSE_WARNING)
        if (MSVC)
            target_compile_options(${name} ${__PUB_DEP_SCOPE} /W4)
        else ()
            target_compile_options(${name} ${__PUB_DEP_SCOPE} -Wall -Wextra)
        endif ()
    endif ()
endfunction()

file(GLOB_RECURSE _VKC_SRCS "include/*.cpp")

add_library(vkc-lib-static STATIC ${_VKC_SRCS})
vkc_common_lib_setting(vkc-lib-static)
add_library(vkc::lib::static ALIAS vkc-lib-static)

add_library(vkc-lib-header-only INTERFACE)
vkc_common_lib_setting(vkc-lib-header-only)
target_compile_definitions(vkc-lib-header-only INTERFACE "_VKC_LIB_HEADER_ONLY")
add_library(vkc::lib::header-only ALIAS vkc-lib-header-only)
