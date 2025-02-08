add_executable(vkc-bin
        main.cpp
)

target_link_libraries(vkc-bin PRIVATE
        vkc-lib
)

copy_dlls_if_needed(vkc-bin)
