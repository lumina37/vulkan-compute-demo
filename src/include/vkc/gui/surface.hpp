#pragma once

#include <expected>
#include <memory>

#include <vulkan/vulkan.hpp>

#include "vkc/device/instance.hpp"
#include "vkc/gui/window.hpp"
#include "vkc/helper/error.hpp"

namespace vkc {

class SurfaceBox {
    SurfaceBox(std::shared_ptr<InstanceBox>&& pInstanceBox, vk::SurfaceKHR surface) noexcept;

public:
    SurfaceBox(SurfaceBox&& rhs) noexcept;
    ~SurfaceBox() noexcept;

    [[nodiscard]] static std::expected<SurfaceBox, Error> create(std::shared_ptr<InstanceBox> pInstanceBox,
                                                                 const WindowBox& windowBox) noexcept;

    [[nodiscard]] vk::SurfaceKHR getSurface() const noexcept { return surface_; }

private:
    std::shared_ptr<InstanceBox> pInstanceBox_;

    vk::SurfaceKHR surface_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/gui/surface.cpp"
#endif
