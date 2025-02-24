add_library(vkc-lib INTERFACE)

target_compile_features(vkc-lib INTERFACE
        cxx_std_23
)

target_link_libraries(vkc-lib INTERFACE
        Vulkan::Vulkan
)

target_include_directories(vkc-lib INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
        $<INSTALL_INTERFACE:include>
)
