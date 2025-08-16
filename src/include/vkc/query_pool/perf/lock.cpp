#include <cstdint>
#include <expected>
#include <limits>
#include <memory>
#include <utility>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/query_pool/perf/lock.hpp"
#endif

namespace vkc {

ProfilingLockBox::ProfilingLockBox(std::shared_ptr<DeviceBox>&& pDeviceBox) noexcept
    : pDeviceBox_(std::move(pDeviceBox)) {}

ProfilingLockBox::~ProfilingLockBox() noexcept {
    if (pDeviceBox_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    device.releaseProfilingLockKHR();
    pDeviceBox_ = nullptr;
}

std::expected<ProfilingLockBox, Error> ProfilingLockBox::create(std::shared_ptr<DeviceBox> pDeviceBox) noexcept {
    vk::AcquireProfilingLockInfoKHR profilingLockInfo;
    profilingLockInfo.setTimeout(std::numeric_limits<uint64_t>::max());

    vk::Device device = pDeviceBox->getDevice();
    const auto acquireRes = device.acquireProfilingLockKHR(profilingLockInfo);
    if (acquireRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, acquireRes}};
    }

    return ProfilingLockBox{std::move(pDeviceBox)};
}

}  // namespace vkc
