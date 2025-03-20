#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <print>
#include <vector>

#include "vkc/helper/defines.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/helper/readfile.hpp"
#endif

namespace vkc {

namespace fs = std::filesystem;

std::vector<std::byte> readFile(const fs::path& path) {
    std::ifstream file{path, std::ios::ate | std::ios::binary};

    if constexpr (ENABLE_DEBUG) {
        if (!file.is_open()) {
            std::println(std::cerr, "Cannot open file: {}", path.string());
        }
    }

    std::streamsize fileSize = file.tellg();
    std::vector<std::byte> buffer(fileSize);

    file.seekg(0);
    file.read((char*)buffer.data(), fileSize);

    return buffer;
}

}  // namespace vkc
