#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <span>
#include <utility>

#include "vkc/extent.hpp"
#include "vkc/helper/error.hpp"

namespace vkc {

namespace fs = std::filesystem;

class StbImageBox {
    StbImageBox(std::byte* image, Extent extent) noexcept;

public:
    StbImageBox(const StbImageBox&) = delete;
    StbImageBox(StbImageBox&& rhs) noexcept;
    ~StbImageBox() noexcept;

    [[nodiscard]] static std::expected<StbImageBox, Error> createFromPath(const fs::path& path) noexcept;
    [[nodiscard]] static std::expected<StbImageBox, Error> createWithExtent(Extent extent) noexcept;

    [[nodiscard]] std::span<std::byte> getImageSpan() const noexcept { return {image_, extent_.size()}; }
    [[nodiscard]] std::byte* getPData() const noexcept { return image_; }

    template <typename Self>
    [[nodiscard]] auto&& getExtent(this Self&& self) noexcept {
        return std::forward_like<Self>(self).extent_;
    }

    [[nodiscard]] std::expected<void, Error> saveTo(const fs::path& path) const noexcept;

    static constexpr vk::Format mapStbCompsToVkFormat(int comps) noexcept;

private:
    std::byte* image_;
    Extent extent_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/stb_image.cpp"
#endif
