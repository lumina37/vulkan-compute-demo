#include <expected>
#include <memory>
#include <utility>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/sync/semaphore.hpp"
#endif

namespace vkc {

SemaphoreBox::SemaphoreBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::Semaphore semaphore) noexcept
    : pDeviceBox_(std::move(pDeviceBox)), semaphore_(semaphore) {}

SemaphoreBox::SemaphoreBox(SemaphoreBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)), semaphore_(std::exchange(rhs.semaphore_, nullptr)) {}

SemaphoreBox::~SemaphoreBox() noexcept {
    if (semaphore_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    device.destroySemaphore(semaphore_);
    semaphore_ = nullptr;
}

std::expected<SemaphoreBox, Error> SemaphoreBox::create(std::shared_ptr<DeviceBox> pDeviceBox) noexcept {
    vk::Device device = pDeviceBox->getDevice();
    vk::SemaphoreCreateInfo semaphoreInfo;
    const auto [semaphoreRes, semaphore] = device.createSemaphore(semaphoreInfo);
    if (semaphoreRes != vk::Result::eSuccess) {
        return std::unexpected{Error{semaphoreRes}};
    }

    return SemaphoreBox{std::move(pDeviceBox), semaphore};
}

}  // namespace vkc
