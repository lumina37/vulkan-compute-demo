#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <vector>

namespace vkc {

namespace fs = std::filesystem;

std::expected<std::vector<std::byte>, Error> readFile(const fs::path& path) noexcept;

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/helper/readfile.cpp"
#endif
