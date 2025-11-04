#include <ranges>

#include "vkc/device/physical/box.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/std.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/query_pool/perf/props.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

PerfCounterProp::PerfCounterProp(const vk::PerformanceCounterKHR& perfCounter,
                                 const vk::PerformanceCounterDescriptionKHR& perfCounterDesc) noexcept
    : unit_(perfCounter.unit),
      scope_(perfCounter.scope),
      storage_(perfCounter.storage),
      name_(perfCounterDesc.name),
      category_(perfCounterDesc.category),
      description_(perfCounterDesc.description) {}

std::expected<PerfCounterProps, Error> PerfCounterProps::create(const PhyDeviceBox& phyDeviceBox,
                                                                uint32_t queueFamilyIndex) noexcept {
    PerfCounterProps props;
    const vk::PhysicalDevice phyDevice = phyDeviceBox.getPhyDevice();

    auto [rawPerfCountersRes, rawPerfCounters] =
        phyDevice.enumerateQueueFamilyPerformanceQueryCountersKHR(queueFamilyIndex);
    if (rawPerfCountersRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, rawPerfCountersRes}};
    }

    const auto makeProp = [](const auto& pair) {
        const auto& [info, desc] = pair;
        return PerfCounterProp(info, desc);
    };

    props.rawPerfCounters_ = std::move(rawPerfCounters);
    props.perfCounters = rgs::views::zip(props.rawPerfCounters_.first, props.rawPerfCounters_.second) |
                         rgs::views::transform(makeProp) | rgs::to<std::vector>();

    return props;
}

}  // namespace vkc
