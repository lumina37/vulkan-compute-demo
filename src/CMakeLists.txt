file(GLOB_RECURSE _VKC_SRCS "include/*.cpp")
add_library(vkc-lib STATIC ${_VKC_SRCS})

target_compile_features(vkc-lib PUBLIC cxx_std_23)
target_link_libraries(vkc-lib PUBLIC Vulkan::Vulkan)
target_include_directories(vkc-lib PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${stb_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(vkc-lib PRIVATE ${stb_SOURCE_DIR})

if (VKC_VERBOSE_WARNING)
    if (MSVC)
        target_compile_options(vkc-lib PUBLIC /W4)
    else ()
        target_compile_options(vkc-lib PUBLIC -Wall -Wextra)
    endif ()
endif ()
