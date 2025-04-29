#include <algorithm>
#include <cstdint>
#include <expected>
#include <print>
#include <ranges>
#include <utility>

#include "vkc/device/instance.hpp"
#include "vkc/helper/defines.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/score.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/physical.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

PhysicalDeviceManager::PhysicalDeviceManager(vk::PhysicalDevice physicalDevice,
                                             vk::PhysicalDeviceLimits limits) noexcept
    : physicalDevice_(physicalDevice), limits_(limits) {}

PhysicalDeviceManager::PhysicalDeviceManager(PhysicalDeviceManager&& rhs) noexcept
    : physicalDevice_(rhs.physicalDevice_), limits_(rhs.limits_) {}

std::expected<PhysicalDeviceManager, Error> PhysicalDeviceManager::create(const InstanceManager& instMgr) noexcept {
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

    const auto [physicalDevicesRes, physicalDevices] = instance.enumeratePhysicalDevices();
    if (physicalDevicesRes != vk::Result::eSuccess) {
        return std::unexpected{Error{physicalDevicesRes}};
    }

    std::vector<Score<size_t>> scores;
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
        return std::unexpected{Error{-1, "no sufficient physical device"}};
    }

    const auto maxScoreIt = std::max_element(scores.begin(), scores.end());
    const uint32_t physicalDeviceIdx = (uint32_t)maxScoreIt->attachment;

    const vk::PhysicalDevice physicalDevice = physicalDevices[physicalDeviceIdx];
    const vk::PhysicalDeviceLimits limits = physicalDevice.getProperties().limits;

    return PhysicalDeviceManager{physicalDevice, limits};
}

}  // namespace vkc
