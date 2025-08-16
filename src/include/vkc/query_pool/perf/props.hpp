#pragma once

#include <expected>
#include <string>
#include <vector>

#include "vkc/device/physical/box.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class PerfCounterProp {
public:
    PerfCounterProp() noexcept = default;
    PerfCounterProp(const PerfCounterProp&) = delete;
    PerfCounterProp(PerfCounterProp&&) noexcept = default;

    PerfCounterProp(const vk::PerformanceCounterKHR& perfCounter,
                    const vk::PerformanceCounterDescriptionKHR& perfCounterDesc) noexcept;

    [[nodiscard]] vk::PerformanceCounterUnitKHR getUnit() const noexcept { return unit_; }
    [[nodiscard]] vk::PerformanceCounterScopeKHR getScope() const noexcept { return scope_; }
    [[nodiscard]] vk::PerformanceCounterStorageKHR getStorage() const noexcept { return storage_; }
    [[nodiscard]] std::string_view getName() const noexcept { return name_; }
    [[nodiscard]] std::string_view getCategory() const noexcept { return category_; }
    [[nodiscard]] std::string_view getDescription() const noexcept { return description_; }

private:
    vk::PerformanceCounterUnitKHR unit_;
    vk::PerformanceCounterScopeKHR scope_;
    vk::PerformanceCounterStorageKHR storage_;
    std::string_view name_;
    std::string_view category_;
    std::string_view description_;
};

class PerfCounterProps {
public:
    PerfCounterProps() noexcept = default;
    PerfCounterProps(const PerfCounterProps&) = delete;
    PerfCounterProps(PerfCounterProps&&) noexcept = default;

    [[nodiscard]] static std::expected<PerfCounterProps, Error> create(const PhyDeviceBox& phyDeviceBox,
                                                                       uint32_t queueFamilyIndex) noexcept;

    std::vector<PerfCounterProp> perfCounters;

private:
    std::pair<std::vector<vk::PerformanceCounterKHR>, std::vector<vk::PerformanceCounterDescriptionKHR>>
        rawPerfCounters_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/query_pool/perf/props.cpp"
#endif
