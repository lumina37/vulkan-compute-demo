#pragma once

#include <expected>
#include <memory>

#include <vulkan/vulkan.hpp>

#include "vkc/device/instance.hpp"
#include "vkc/gui/window.hpp"
#include "vkc/helper/error.hpp"

namespace vkc {

class SurfaceManager {
    SurfaceManager(std::shared_ptr<InstanceManager>&& pInstanceMgr, vk::SurfaceKHR surface) noexcept;

public:
    SurfaceManager(SurfaceManager&& rhs) noexcept;
    ~SurfaceManager() noexcept;

    [[nodiscard]] static std::expected<SurfaceManager, Error> create(std::shared_ptr<InstanceManager> pInstanceMgr,
                                                                     const WindowManager& windowMgr) noexcept;

    [[nodiscard]] vk::SurfaceKHR getSurface() const noexcept { return surface_; }

private:
    std::shared_ptr<InstanceManager> pInstanceMgr_;

    vk::SurfaceKHR surface_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/gui/surface.cpp"
#endif
