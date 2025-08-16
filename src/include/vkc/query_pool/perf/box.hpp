#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class PerfQueryPoolBox {
    PerfQueryPoolBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::QueryPool queryPool, int queryCount,
                     int perfCounterCount) noexcept;

public:
    PerfQueryPoolBox(const PerfQueryPoolBox&) = delete;
    PerfQueryPoolBox(PerfQueryPoolBox&& rhs) noexcept;
    ~PerfQueryPoolBox() noexcept;

    [[nodiscard]] static std::expected<PerfQueryPoolBox, Error> create(std::shared_ptr<DeviceBox> pDeviceBox,
                                                                       uint32_t queueFamilyIndex, int queryCount,
                                                                       std::span<uint32_t> perfCounterIndices) noexcept;

    [[nodiscard]] int getQueryIndex() const noexcept { return queryIndex_; }
    [[nodiscard]] std::expected<void, Error> addQueryIndex() noexcept;
    [[nodiscard]] int getQueryCount() const noexcept { return queryCount_; }
    [[nodiscard]] vk::QueryPool getQueryPool() const noexcept { return queryPool_; }

    template <typename... TRes>
        requires(std::is_arithmetic_v<TRes> && ...)
    [[nodiscard]] std::expected<std::vector<std::tuple<TRes...>>, Error> getResults() const noexcept;

    void resetQueryIndex() noexcept;
    void hostReset() noexcept;

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;
    vk::QueryPool queryPool_;
    int queryCount_;
    int queryIndex_;
    int perfCounterCount_;
};

template <typename... TRes>
    requires(std::is_arithmetic_v<TRes> && ...)
std::expected<std::vector<std::tuple<TRes...>>, Error> PerfQueryPoolBox::getResults() const noexcept {
    std::vector<std::tuple<TRes...>> results(queryIndex_);
    std::vector<vk::PerformanceCounterResultKHR> rawResults(results.size() * perfCounterCount_);

    vk::Device device = pDeviceBox_->getDevice();
    const vk::DeviceSize stride = perfCounterCount_ * sizeof(VkPerformanceCounterResultKHR);
    const auto queryRes = device.getQueryPoolResults(queryPool_, 0, queryIndex_, results.size() * stride,
                                                     rawResults.data(), stride, vk::QueryResultFlagBits::eWait);
    if (queryRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, queryRes}};
    }

    const auto extractValue = []<typename Tv>(const VkPerformanceCounterResultKHR& rawResult) {
        if constexpr (std::is_same_v<Tv, int32_t>) {
            return rawResult.int32;
        } else if constexpr (std::is_same_v<Tv, int64_t>) {
            return rawResult.int64;
        } else if constexpr (std::is_same_v<Tv, uint32_t>) {
            return rawResult.uint32;
        } else if constexpr (std::is_same_v<Tv, uint64_t>) {
            return rawResult.uint64;
        } else if constexpr (std::is_same_v<Tv, float>) {
            return rawResult.float32;
        } else if constexpr (std::is_same_v<Tv, double>) {
            return rawResult.float64;
        } else {
            return static_cast<Tv>(rawResult.uint64);
        }
    };

    const auto genTuple = [&]<size_t... Indices>(std::index_sequence<Indices...>, size_t i) {
        return std::make_tuple(extractValue.template operator()<std::tuple_element_t<Indices, std::tuple<TRes...>>>(
            rawResults[i * perfCounterCount_ + Indices])...);
    };

    for (size_t i = 0; i < queryIndex_; i++) {
        results[i] = genTuple(std::make_index_sequence<sizeof...(TRes)>{}, i);
    }

    return results;
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/query_pool/perf/box.cpp"
#endif
