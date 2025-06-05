#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <utility>

#include "vkc/device/logical.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/pool.hpp"
#endif

namespace vkc {

DescPoolBox::DescPoolBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::DescriptorPool descPool) noexcept
    : pDeviceBox_(std::move(pDeviceBox)), descPool_(descPool) {}

DescPoolBox::DescPoolBox(DescPoolBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)), descPool_(std::exchange(rhs.descPool_, nullptr)) {}

DescPoolBox::~DescPoolBox() noexcept {
    if (descPool_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    device.destroyDescriptorPool(descPool_);
    descPool_ = nullptr;
}

std::expected<DescPoolBox, Error> DescPoolBox::create(
    std::shared_ptr<DeviceBox> pDeviceBox, std::span<const vk::DescriptorPoolSize> poolSizes) noexcept {
    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.setMaxSets((uint32_t)poolSizes.size());
    poolInfo.setPoolSizes(poolSizes);

    vk::Device device = pDeviceBox->getDevice();
    const auto [descPoolRes, descPool] = device.createDescriptorPool(poolInfo);
    if (descPoolRes != vk::Result::eSuccess) {
        return std::unexpected{Error{descPoolRes}};
    }

    return DescPoolBox{std::move(pDeviceBox), descPool};
}

}  // namespace vkc
