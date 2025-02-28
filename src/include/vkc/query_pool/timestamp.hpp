#pragma once

#include <cstddef>
#include <ranges>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

namespace vkc {

namespace rgs = std::ranges;

class TimestampQueryPoolManager {
public:
    inline TimestampQueryPoolManager(DeviceManager& deviceMgr, const int queryCount, const float timestampPeriod);
    inline ~TimestampQueryPoolManager() noexcept;

    inline int getQueryIndex() const noexcept { return queryIndex_; }
    inline void addQueryIndex() noexcept { queryIndex_++; }
    inline void resetQueryIndex() noexcept { queryIndex_ = 0; }

    [[nodiscard]] inline int getQueryCount() const noexcept { return queryCount_; }

    template <typename Self>
    [[nodiscard]] auto&& getQueryPool(this Self&& self) noexcept {
        return std::forward_like<Self>(self).queryPool_;
    }

    [[nodiscard]] std::vector<float> getElaspedTimes() const noexcept;

private:
    DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::QueryPool queryPool_;
    int queryCount_;
    int queryIndex_;
    float timestampPeriod_;
};

TimestampQueryPoolManager::TimestampQueryPoolManager(DeviceManager& deviceMgr, const int queryCount,
                                                     const float timestampPeriod)
    : deviceMgr_(deviceMgr), queryCount_(queryCount), queryIndex_(0), timestampPeriod_(timestampPeriod) {
    vk::QueryPoolCreateInfo queryPoolInfo;
    queryPoolInfo.setQueryType(vk::QueryType::eTimestamp);
    queryPoolInfo.setQueryCount(queryCount);

    auto& device = deviceMgr.getDevice();
    queryPool_ = device.createQueryPool(queryPoolInfo);
}

TimestampQueryPoolManager::~TimestampQueryPoolManager() noexcept {
    auto& device = deviceMgr_.getDevice();
    device.destroyQueryPool(queryPool_);
}

std::vector<float> TimestampQueryPoolManager::getElaspedTimes() const noexcept {
    std::vector<uint64_t> timestamps(queryIndex_);
    std::vector<float> elapsedTimes(queryIndex_ / 2);
    constexpr size_t valueSize = sizeof(decltype(timestamps)::value_type);

    const auto& device = deviceMgr_.getDevice();
    vk::Result queryResult =
        device.getQueryPoolResults(queryPool_, 0, queryIndex_, timestamps.size() * valueSize, (void*)timestamps.data(),
                                   valueSize, vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);

    const float timestampPeriodMs = timestampPeriod_ * 1000000.0f;
    for (const auto [idx, pair] : rgs::views::enumerate(timestamps | rgs::views::chunk(2))) {
        elapsedTimes[idx] = ((float)pair[1] - (float)pair[0]) / timestampPeriodMs;
    }

    return elapsedTimes;
}

}  // namespace vkc
