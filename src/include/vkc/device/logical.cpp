#include <cstdint>
#include <expected>
#include <format>
#include <ranges>
#include <utility>
#include <vector>

#include "vkc/device/physical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/logical.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

DeviceBox::DeviceBox(vk::Device device, std::vector<QueueIndex>&& queueIndices) noexcept
    : device_(device), queueIndices_(queueIndices) {}

DeviceBox::DeviceBox(DeviceBox&& rhs) noexcept
    : device_(std::exchange(rhs.device_, nullptr)), queueIndices_(std::move(rhs.queueIndices_)) {}

DeviceBox::~DeviceBox() noexcept {
    if (device_ == nullptr) return;
    device_.destroy();
    device_ = nullptr;
}

std::expected<DeviceBox, Error> DeviceBox::create(PhyDeviceBox& phyDeviceBox,
                                                          QueueIndex requiredQueueIndex) noexcept {
    return createWithExts(phyDeviceBox, requiredQueueIndex, {});
}

std::expected<DeviceBox, Error> DeviceBox::createWithExts(
    PhyDeviceBox& phyDeviceBox, QueueIndex requiredQueueIndex,
    std::span<const std::string_view> enableExtNames) noexcept {
    const std::array requiredQueueIndices{requiredQueueIndex};
    return createWithMultiQueueAndExts(phyDeviceBox, requiredQueueIndices, enableExtNames);
}

std::expected<DeviceBox, Error> DeviceBox::createWithMultiQueueAndExts(
    PhyDeviceBox& phyDeviceBox, std::span<const QueueIndex> requiredQueueIndices,
    std::span<const std::string_view> enableExtNames) noexcept {
    vk::DeviceCreateInfo deviceInfo;

    constexpr auto genQueueInfo = [](QueueIndex idx) {
        vk::DeviceQueueCreateInfo queueInfo;
        constexpr float priority = 1.0f;
        queueInfo.setQueuePriorities(priority);
        queueInfo.setQueueFamilyIndex(idx.familyIndex);
        queueInfo.setQueueCount(1);
        return queueInfo;
    };

    auto queueInfos = requiredQueueIndices | rgs::views::transform(genQueueInfo) | rgs::to<std::vector>();
    deviceInfo.setQueueCreateInfos(queueInfos);

    auto enabledPExtNames = enableExtNames | rgs::views::transform([](std::string_view name) { return name.data(); }) |
                            rgs::to<std::vector>();
    deviceInfo.setPEnabledExtensionNames(enabledPExtNames);

    vk::PhysicalDevice phyDevice = phyDeviceBox.getPhyDevice();
    const auto [deviceRes, device] = phyDevice.createDevice(deviceInfo);
    if (deviceRes != vk::Result::eSuccess) {
        return std::unexpected{Error{deviceRes}};
    }

    auto copiedQueueIndices = requiredQueueIndices | rgs::to<std::vector>();

    return DeviceBox{device, std::move(copiedQueueIndices)};
}

std::expected<vk::Queue, Error> DeviceBox::getQueue(vk::QueueFlags type) const noexcept {
    constexpr auto exposeType = [](QueueIndex queueIndex) { return queueIndex.type; };

    const auto queueIndexIt = rgs::find(queueIndices_, type, exposeType);
    if (queueIndexIt == queueIndices_.end()) {
        auto errMsg = std::format("no family index for type={}", (uint32_t)type);
        return std::unexpected{Error{-1, std::move(errMsg)}};
    }

    const uint32_t familyIndex = queueIndexIt->familyIndex;
    const vk::Queue queue = device_.getQueue(familyIndex, 0);

    return queue;
}

}  // namespace vkc
