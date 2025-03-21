#include <cstddef>
#include <cstdint>
#include <ranges>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/query_pool/timestamp.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

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
        elapsedTimes[idx] = (float)(pair[1] - pair[0]) / timestampPeriodMs;
    }

    return elapsedTimes;
}

}  // namespace vkc
