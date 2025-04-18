#include <algorithm>
#include <cstdint>
#include <iostream>
#include <print>
#include <ranges>

#include <vulkan/vulkan.hpp>

#include "vkc/device/instance.hpp"
#include "vkc/helper/defines.hpp"
#include "vkc/helper/score.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/physical.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

PhysicalDeviceManager::PhysicalDeviceManager(const InstanceManager& instMgr) {
    const auto& instance = instMgr.getInstance();

    const auto isPhysicalDeviceOK = [](const vk::PhysicalDeviceProperties& phyDeviceProp) {
        if constexpr (ENABLE_DEBUG) {
            if (phyDeviceProp.limits.timestampPeriod == 0) return false;
            if (!phyDeviceProp.limits.timestampComputeAndGraphics) return false;
        }
        return true;
    };

    const auto getPhysicalDeviceScore = [](const vk::PhysicalDeviceProperties& phyDeviceProp) {
        int score = 0;
        if (phyDeviceProp.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
            score += 1;
        } else if (phyDeviceProp.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
            score += 2;
        }
        return score;
    };

    const auto& physicalDevices = instance.enumeratePhysicalDevices();

    std::vector<ScoreWithIndex> scores;
    scores.reserve(physicalDevices.size());
    for (const auto [idx, physicalDevice] : rgs::views::enumerate(physicalDevices)) {
        const auto& phyDeviceProp = physicalDevice.getProperties();
        if (!isPhysicalDeviceOK(phyDeviceProp)) {
            continue;
        }

        const int score = getPhysicalDeviceScore(phyDeviceProp);
        scores.emplace_back(score, idx);

        if constexpr (ENABLE_DEBUG) {
            std::println("Candidate physical device: {}. Vk API version: {}.{}.{}. Score: {}",
                         phyDeviceProp.deviceName.data(), VK_API_VERSION_MAJOR(phyDeviceProp.apiVersion),
                         VK_API_VERSION_MINOR(phyDeviceProp.apiVersion), VK_API_VERSION_PATCH(phyDeviceProp.apiVersion),
                         score);
        }
    }

    if (scores.empty()) {
        if constexpr (ENABLE_DEBUG) {
            std::println(std::cerr, "No sufficient physical device found!");
        }
        return;
    }

    const auto maxScoreIt = std::max_element(scores.begin(), scores.end());
    const uint32_t physicalDeviceIdx = (uint32_t)maxScoreIt->index;

    physicalDevice_ = physicalDevices[physicalDeviceIdx];
    limits_ = physicalDevice_.getProperties().limits;
}

}  // namespace vkc
