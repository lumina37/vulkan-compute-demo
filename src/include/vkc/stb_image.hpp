#pragma once

#include <filesystem>
#include <span>

#include <stb_image.h>
#include <stb_image_write.h>

#include "vkc/extent.hpp"

namespace vkc {

namespace fs = std::filesystem;

class StbImageManager {
public:
    inline StbImageManager(const fs::path& path);
    inline StbImageManager(const ExtentManager& extent);
    inline ~StbImageManager() noexcept;

    inline std::span<std::byte> getImageSpan() const noexcept { return {image_, extent_.size()}; }

    template <typename Self>
    [[nodiscard]] auto&& getExtent(this Self&& self) noexcept {
        return std::forward_like<Self>(self).extent_;
    }

    inline void saveTo(const fs::path& path) const;

private:
    std::byte* image_;
    ExtentManager extent_;
};

StbImageManager::StbImageManager(const fs::path& path) {
    int width, height, oriComps;
    constexpr int comps = 4;
    image_ = (std::byte*)stbi_load((char*)path.string().c_str(), &width, &height, &oriComps, comps);
    extent_ = {width, height, comps};
}

StbImageManager::StbImageManager(const ExtentManager& extent) : extent_(extent) {
    image_ = (std::byte*)STBI_MALLOC(extent.size());
}

StbImageManager::~StbImageManager() noexcept { STBI_FREE(image_); }

void StbImageManager::saveTo(const fs::path& path) const {
    stbi_write_png((char*)path.string().c_str(), extent_.width(), extent_.height(), extent_.comps(), image_, 0);
}

}  // namespace vkc
