cmake_minimum_required(VERSION 3.15)

project(VulkanComputeDemo
        VERSION 0.0.1
        DESCRIPTION "Vulkan compute demo"
        LANGUAGES CXX
)

# Dependencies
include(FetchContent)

find_package(Vulkan REQUIRED)

FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG master
)
FetchContent_MakeAvailable(stb)

# Helper
function(copy_dlls_if_needed name)
    if (WIN32)
        add_custom_command(TARGET ${name} POST_BUILD COMMAND ${CMAKE_COMMAND} -E
                copy_if_different $<TARGET_RUNTIME_DLLS:${name}> $<TARGET_FILE:${name}> $<TARGET_FILE_DIR:${name}> COMMAND_EXPAND_LISTS)
    endif ()
endfunction()

# Sources
add_subdirectory(src)
