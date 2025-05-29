#pragma once

#include <expected>
#include <memory>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class SemaphoreManager {
    SemaphoreManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::Semaphore semaphore) noexcept;

public:
    SemaphoreManager(SemaphoreManager&& rhs) noexcept;
    ~SemaphoreManager() noexcept;

    [[nodiscard]] static std::expected<SemaphoreManager, Error> create(
        std::shared_ptr<DeviceManager> pDeviceMgr) noexcept;

    [[nodiscard]] vk::Semaphore getSemaphore() const noexcept { return semaphore_; }

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    vk::Semaphore semaphore_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/sync/semaphore.cpp"
#endif
