#pragma once

#include <cstdint>
#include <expected>
#include <vector>

#include "vkc/device/physical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

struct QueueIndex {
    vk::QueueFlags type;
    uint32_t familyIndex;
};

class DeviceBox {
    DeviceBox(vk::Device device, std::vector<QueueIndex>&& queueIndices) noexcept;

public:
    DeviceBox(const DeviceBox&) = delete;
    DeviceBox(DeviceBox&& rhs) noexcept;
    ~DeviceBox() noexcept;

    [[nodiscard]] static std::expected<DeviceBox, Error> create(PhyDeviceBox& phyDeviceBox,
                                                                QueueIndex requiredQueueIndex) noexcept;

    [[nodiscard]] static std::expected<DeviceBox, Error> createWithExts(
        PhyDeviceBox& phyDeviceBox, QueueIndex requiredQueueIndex, std::span<const std::string_view> enableExtNames,
        vk::PhysicalDeviceFeatures2* pFeature) noexcept;

    [[nodiscard]] static std::expected<DeviceBox, Error> createWithMultiQueueAndExts(
        PhyDeviceBox& phyDeviceBox, std::span<const QueueIndex> requiredQueueIndices,
        std::span<const std::string_view> enableExtNames, vk::PhysicalDeviceFeatures2* pFeature) noexcept;

    [[nodiscard]] vk::Device getDevice() const noexcept { return device_; }

    [[nodiscard]] std::expected<vk::Queue, Error> getQueue(vk::QueueFlags type) const noexcept;

private:
    vk::Device device_;
    std::vector<QueueIndex> queueIndices_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/logical.cpp"
#endif
