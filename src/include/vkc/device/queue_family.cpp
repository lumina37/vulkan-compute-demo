#include <algorithm>
#include <bit>
#include <cstdint>
#include <expected>
#include <print>
#include <ranges>

#include "vkc/device/physical.hpp"
#include "vkc/device/score.hpp"
#include "vkc/helper/defines.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/queue_family.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

std::expected<uint32_t, Error> defaultComputeQFamilyIndex(const PhyDeviceBox& phyDeviceBox) noexcept {
    vk::PhysicalDevice physicalDevice = phyDeviceBox.getPhyDevice();

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

        const float score = -(float)std::popcount((uint32_t)queueFamilyProp.queueFlags);
        scores.emplace_back(score, idx);

        if constexpr (ENABLE_DEBUG) {
            std::println("Candidate queue family: {}. Score: {}", idx, score);
        }
    }

    if (scores.empty()) {
        return std::unexpected{Error{ECate::eVkC, ECode::eResourceInvalid, "no sufficient queue family"}};
    }

    const auto maxScoreIt = std::max_element(scores.begin(), scores.end());
    return (uint32_t)maxScoreIt->attachment;
}

}  // namespace vkc
