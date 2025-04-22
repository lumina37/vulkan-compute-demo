#include <algorithm>
#include <cstdint>
#include <iostream>
#include <print>
#include <ranges>

#include <vulkan/vulkan.hpp>

#include "vkc/device/physical.hpp"
#include "vkc/helper/defines.hpp"
#include "vkc/helper/score.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/queue_family.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

uint32_t defaultComputeQFamilyIndex(const PhysicalDeviceManager& phyDeviceMgr) {
    const auto& physicalDevice = phyDeviceMgr.getPhysicalDevice();

    const auto isQueueFamilyOK = [](const vk::QueueFamilyProperties& queueFamilyProp) {
        if (!(queueFamilyProp.queueFlags & vk::QueueFlagBits::eCompute)) return false;

        if constexpr (ENABLE_DEBUG) {
            if (queueFamilyProp.timestampValidBits == 0) return false;
        }

        return true;
    };

    const auto& queueFamilyProps = physicalDevice.getQueueFamilyProperties();

    std::vector<Score<size_t>> scores;
    scores.reserve(queueFamilyProps.size());
    for (const auto [idx, queueFamilyProp] : rgs::views::enumerate(queueFamilyProps)) {
        if (!isQueueFamilyOK(queueFamilyProp)) {
            continue;
        }

        const int score = -(int)idx;
        scores.emplace_back(score, idx);

        if constexpr (ENABLE_DEBUG) {
            std::println("Candidate queue family: {}. Score: {}", idx, score);
        }
    }

    if (scores.empty()) {
        if constexpr (ENABLE_DEBUG) {
            std::println(std::cerr, "No sufficient queue family found!");
        }
        return 0;
    }

    const auto maxScoreIt = std::max_element(scores.begin(), scores.end());
    return (uint32_t)maxScoreIt->attachment;
}

}  // namespace vkc
