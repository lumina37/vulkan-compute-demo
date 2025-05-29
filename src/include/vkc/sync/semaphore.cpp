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

SemaphoreManager::SemaphoreManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::Semaphore semaphore) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)), semaphore_(semaphore) {}

SemaphoreManager::SemaphoreManager(SemaphoreManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)), semaphore_(std::exchange(rhs.semaphore_, nullptr)) {}

SemaphoreManager::~SemaphoreManager() noexcept {
    if (semaphore_ == nullptr) return;
    vk::Device device = pDeviceMgr_->getDevice();
    device.destroySemaphore(semaphore_);
    semaphore_ = nullptr;
}

std::expected<SemaphoreManager, Error> SemaphoreManager::create(std::shared_ptr<DeviceManager> pDeviceMgr) noexcept {
    vk::Device device = pDeviceMgr->getDevice();
    vk::SemaphoreCreateInfo semaphoreInfo;
    const auto [semaphoreRes, semaphore] = device.createSemaphore(semaphoreInfo);
    if (semaphoreRes != vk::Result::eSuccess) {
        return std::unexpected{Error{semaphoreRes}};
    }

    return SemaphoreManager{std::move(pDeviceMgr), semaphore};
}

}  // namespace vkc
