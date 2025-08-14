#pragma once

#include <expected>
#include <vector>

#include "vkc/device/physical/box.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class PerfQueryProps {
public:
    PerfQueryProps() noexcept = default;
    PerfQueryProps(const PerfQueryProps&) = delete;
    PerfQueryProps(PerfQueryProps&&) noexcept = default;

    [[nodiscard]] static std::expected<PerfQueryProps, Error> create(const PhyDeviceBox& phyDeviceBox,
                                                                     uint32_t queueFamilyIndex) noexcept;

    // Members
    std::vector<vk::PerformanceCounterDescriptionKHR> descs;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/query_pool/props.cpp"
#endif
