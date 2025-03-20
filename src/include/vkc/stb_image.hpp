#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <utility>

#include "vkc/extent.hpp"

namespace vkc {

namespace fs = std::filesystem;

class StbImageManager {
public:
    StbImageManager(const fs::path& path);
    StbImageManager(const ExtentManager& extent);
    ~StbImageManager() noexcept;

    std::span<std::byte> getImageSpan() const noexcept { return {image_, extent_.size()}; }

    template <typename Self>
    [[nodiscard]] auto&& getExtent(this Self&& self) noexcept {
        return std::forward_like<Self>(self).extent_;
    }

    void saveTo(const fs::path& path) const;

private:
    std::byte* image_;
    ExtentManager extent_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/stb_image.cpp"
#endif
