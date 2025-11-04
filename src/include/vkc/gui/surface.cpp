#include <memory>

#include "vkc/device/instance.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/std.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/gui/surface.hpp"
#endif

namespace vkc {

SurfaceBox::SurfaceBox(std::shared_ptr<InstanceBox>&& pInstanceBox, vk::SurfaceKHR surface) noexcept
    : pInstanceBox_(std::move(pInstanceBox)), surface_(surface) {}

SurfaceBox::SurfaceBox(SurfaceBox&& rhs) noexcept
    : pInstanceBox_(std::move(rhs.pInstanceBox_)), surface_(std::exchange(rhs.surface_, nullptr)) {}

SurfaceBox::~SurfaceBox() noexcept {
    if (surface_ == nullptr) return;
    vk::Instance instance = pInstanceBox_->getInstance();
    instance.destroySurfaceKHR(surface_);
    surface_ = nullptr;
}

std::expected<SurfaceBox, Error> SurfaceBox::create(std::shared_ptr<InstanceBox> pInstanceBox,
                                                    const WindowBox& windowBox) noexcept {
    vk::SurfaceKHR surface;
    glfwCreateWindowSurface(pInstanceBox->getInstance(), windowBox.getWindow(), nullptr, (VkSurfaceKHR*)&surface);

    return SurfaceBox{std::move(pInstanceBox), surface};
}

}  // namespace vkc
