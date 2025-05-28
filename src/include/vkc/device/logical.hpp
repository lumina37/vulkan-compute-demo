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

class DeviceManager {
    DeviceManager(vk::Device device, std::vector<QueueIndex>&& queueIndices) noexcept;

public:
    DeviceManager(DeviceManager&& rhs) noexcept;
    ~DeviceManager() noexcept;

    [[nodiscard]] static std::expected<DeviceManager, Error> create(PhyDeviceManager& phyDeviceMgr,
                                                                    QueueIndex requiredQueueIndex) noexcept;

    [[nodiscard]] static std::expected<DeviceManager, Error> createWithExts(
        PhyDeviceManager& phyDeviceMgr, QueueIndex requiredQueueIndex,
        std::span<const std::string_view> enableExtNames) noexcept;

    [[nodiscard]] static std::expected<DeviceManager, Error> createWithMultiQueueAndExts(
        PhyDeviceManager& phyDeviceMgr, std::span<const QueueIndex> requiredQueueIndices,
        std::span<const std::string_view> enableExtNames) noexcept;

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
