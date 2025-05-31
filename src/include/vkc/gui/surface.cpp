#include <expected>
#include <memory>
#include <utility>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/gui/surface.hpp"
#endif

namespace vkc {

SurfaceManager::SurfaceManager(std::shared_ptr<InstanceManager>&& pInstanceMgr, vk::SurfaceKHR surface) noexcept
    : pInstanceMgr_(std::move(pInstanceMgr)), surface_(surface) {}

SurfaceManager::SurfaceManager(SurfaceManager&& rhs) noexcept
    : pInstanceMgr_(std::move(rhs.pInstanceMgr_)), surface_(std::exchange(rhs.surface_, nullptr)) {}

SurfaceManager::~SurfaceManager() noexcept {
    if (surface_ == nullptr) return;
    auto instance = pInstanceMgr_->getInstance();
    instance.destroySurfaceKHR(surface_);
}

std::expected<SurfaceManager, Error> SurfaceManager::create(std::shared_ptr<InstanceManager> pInstanceMgr,
                                                            const WindowManager& windowMgr) noexcept {
    vk::SurfaceKHR surface;
    glfwCreateWindowSurface(pInstanceMgr->getInstance(), windowMgr.getWindow(), nullptr, (VkSurfaceKHR*)&surface);

    return SurfaceManager{std::move(pInstanceMgr), surface};
}

}  // namespace vkc
