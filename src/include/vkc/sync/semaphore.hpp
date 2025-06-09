#pragma once

#include <expected>
#include <memory>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class SemaphoreBox {
    SemaphoreBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::Semaphore semaphore) noexcept;

public:
    SemaphoreBox(const SemaphoreBox&) = delete;
    SemaphoreBox(SemaphoreBox&& rhs) noexcept;
    ~SemaphoreBox() noexcept;

    [[nodiscard]] static std::expected<SemaphoreBox, Error> create(std::shared_ptr<DeviceBox> pDeviceBox) noexcept;

    [[nodiscard]] vk::Semaphore getSemaphore() const noexcept { return semaphore_; }

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    vk::Semaphore semaphore_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/sync/semaphore.cpp"
#endif
