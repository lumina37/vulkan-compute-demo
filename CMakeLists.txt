cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(VulkanComputeDemo
        VERSION 0.0.1
        DESCRIPTION "Vulkan compute demo"
        LANGUAGES CXX
)

# Options
option(VKC_ENABLE_TESTS "Enable tests" OFF)
option(VKC_VERBOSE_WARNING "Show verbose compiler warnings" OFF)

# Dependencies
include(FetchContent)

find_package(Vulkan)

FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG fede005abaf93d9d7f3a679d1999b2db341b360f
)
FetchContent_MakeAvailable(stb)

FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
)
FetchContent_MakeAvailable(glfw)

if (VKC_ENABLE_TESTS)
    include(CTest)

    FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.8.0
    )
    FetchContent_MakeAvailable(Catch2)
    include(Catch)
endif ()

# Helper
function(vkc_copy_dlls_if_needed name)
    if (WIN32)
        add_custom_command(TARGET ${name} POST_BUILD COMMAND ${CMAKE_COMMAND} -E
                copy_if_different $<TARGET_RUNTIME_DLLS:${name}> $<TARGET_FILE:${name}> $<TARGET_FILE_DIR:${name}> COMMAND_EXPAND_LISTS)
    endif ()
endfunction()

# Sources
add_subdirectory(src)
add_subdirectory(shader)
add_subdirectory(samples)

if (VKC_ENABLE_TESTS)
    add_subdirectory(tests)
endif ()
