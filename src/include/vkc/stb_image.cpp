#include <cstddef>
#include <expected>
#include <filesystem>
#include <utility>

#pragma push_macro("STB_IMAGE_IMPLEMENTATION")
#pragma push_macro("STB_IMAGE_WRITE_IMPLEMENTATION")
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>
#pragma pop_macro("STB_IMAGE_WRITE_IMPLEMENTATION")
#pragma pop_macro("STB_IMAGE_IMPLEMENTATION")

#include "vkc/extent.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/stb_image.hpp"
#endif

namespace vkc {

namespace fs = std::filesystem;

StbImageBox::StbImageBox(std::byte* image, Extent extent) noexcept : image_(image), extent_(extent) {}

StbImageBox::StbImageBox(StbImageBox&& rhs) noexcept
    : image_(std::exchange(rhs.image_, nullptr)), extent_(rhs.extent_) {}

StbImageBox::~StbImageBox() noexcept {
    if (image_ == nullptr) return;
    STBI_FREE(image_);
    image_ = nullptr;
}

std::expected<StbImageBox, Error> StbImageBox::createFromPath(const fs::path& path) noexcept {
    int width, height, oriComps;
    constexpr int comps = 4;

    std::byte* image = (std::byte*)stbi_load(path.string().c_str(), &width, &height, &oriComps, comps);
    if (image == nullptr) return std::unexpected{Error{ECate::eStb, 0, "failed to load image"}};

    Extent extent{width, height, mapStbCompsToVkFormat(comps)};

    return StbImageBox{image, extent};
}

std::expected<StbImageBox, Error> StbImageBox::createWithExtent(const Extent extent) noexcept {
    std::byte* image = (std::byte*)STBI_MALLOC(extent.size());
    if (image == nullptr) return std::unexpected{Error{ECate::eStb, 0}};

    return StbImageBox{image, extent};
}

std::expected<void, Error> StbImageBox::saveTo(const fs::path& path) const noexcept {
    const int stbErr =
        stbi_write_png(path.string().c_str(), extent_.width(), extent_.height(), extent_.bpp(), image_, 0);
    if (stbErr == 0) return std::unexpected{Error{ECate::eStb, 0, "failed to save image"}};

    return {};
}

constexpr vk::Format StbImageBox::mapStbCompsToVkFormat(const int comps) noexcept {
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
