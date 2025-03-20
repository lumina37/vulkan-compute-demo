#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

namespace vkc {

namespace fs = std::filesystem;

std::vector<std::byte> readFile(const fs::path& path);

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/helper/readfile.cpp"
#endif
