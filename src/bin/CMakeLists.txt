add_executable(vkc-bin
        main.cpp
)

target_include_directories(vkc-bin PRIVATE
        ${stb_SOURCE_DIR}
)

target_link_libraries(vkc-bin PRIVATE
        vkc-lib
)

target_compile_definitions(vkc-bin PRIVATE
        STB_IMAGE_IMPLEMENTATION
        STB_IMAGE_WRITE_IMPLEMENTATION
)

copy_dlls_if_needed(vkc-bin)
