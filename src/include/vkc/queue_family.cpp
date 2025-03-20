#include <cstdint>
#include <print>
#include <ranges>

#include <vulkan/vulkan.hpp>

#include "vkc/device/physical.hpp"
#include "vkc/helper/defines.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/queue_family.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

QueueFamilyManager::QueueFamilyManager(const PhyDeviceManager& phyDeviceMgr) : computeQFamilyIndex_(0) {
    const auto& physicalDevice = phyDeviceMgr.getPhysicalDevice();

    const auto isQueueFamilySuitable = [](const vk::QueueFamilyProperties& queueFamilyProp) {
        if (!(queueFamilyProp.queueFlags & vk::QueueFlagBits::eCompute)) return false;

        if constexpr (ENABLE_DEBUG) {
            if (queueFamilyProp.timestampValidBits == 0) return false;
        }

        return true;
    };

    const auto& queueFamilyProps = physicalDevice.getQueueFamilyProperties();
    for (const auto [idx, queueFamilyProp] : rgs::views::enumerate(queueFamilyProps)) {
        if (isQueueFamilySuitable(queueFamilyProp)) {
            computeQFamilyIndex_ = (uint32_t)idx;
            if constexpr (ENABLE_DEBUG) {
                std::println("Findout a sufficient queue family: {}", computeQFamilyIndex_);
            }
            break;
        }
    }
}

}  // namespace vkc
