#include <cstddef>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <utility>
#include <vector>

#include "vkc/helper/error.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/helper/readfile.hpp"
#endif

namespace vkc {

namespace fs = std::filesystem;

std::expected<std::vector<std::byte>, Error> readFile(const fs::path& path) noexcept {
    std::ifstream ifs{path, std::ios::ate | std::ios::binary};
    if (!ifs.good()) {
        auto errMsg = std::format("cannot open file: {}", path.string());
        return std::unexpected{Error{-1, std::move(errMsg)}};
    }

    const std::streamsize fileSize = ifs.tellg();
    std::vector<std::byte> buffer(fileSize);

    ifs.seekg(0);
    ifs.read((char*)buffer.data(), fileSize);

    return buffer;
}

}  // namespace vkc
