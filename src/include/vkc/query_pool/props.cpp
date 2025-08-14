#include <expected>
#include <utility>

#include "vkc/device/physical/box.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/query_pool/props.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

std::expected<PerfQueryProps, Error> PerfQueryProps::create(const PhyDeviceBox& phyDeviceBox,
                                                            uint32_t queueFamilyIndex) noexcept {
    PerfQueryProps props;
    const vk::PhysicalDevice phyDevice = phyDeviceBox.getPhyDevice();

    auto [perfCountersRes, perfCounterInfo] =
        phyDevice.enumerateQueueFamilyPerformanceQueryCountersKHR(queueFamilyIndex);
    if (perfCountersRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, perfCountersRes}};
    }
    auto [perfCounters, perfCounterDescs] = std::move(perfCounterInfo);

    props.descs = std::move(perfCounterDescs);

    return props;
}

}  // namespace vkc
