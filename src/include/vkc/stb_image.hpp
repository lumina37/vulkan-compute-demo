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
    StbImageManager(const Extent& extent);
    ~StbImageManager() noexcept;

    [[nodiscard]] std::span<std::byte> getImageSpan() const noexcept { return {image_, extent_.size()}; }

    template <typename Self>
    [[nodiscard]] auto&& getExtent(this Self&& self) noexcept {
        return std::forward_like<Self>(self).extent_;
    }

    void saveTo(const fs::path& path) const;

    static constexpr vk::Format mapStbCompsToVkFormat(int comps) noexcept;

private:
    std::byte* image_;
    Extent extent_;
};

constexpr vk::Format StbImageManager::mapStbCompsToVkFormat(const int comps) noexcept {
    switch (comps) {
        case 1:
            return vk::Format::eR8Unorm;
        case 2:
            return vk::Format::eR8G8Unorm;
        case 3:
            return vk::Format::eR8G8B8Unorm;
        case 4:
            return vk::Format::eR8G8B8A8Unorm;
        default:
            std::unreachable();
    }
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/stb_image.cpp"
#endif
