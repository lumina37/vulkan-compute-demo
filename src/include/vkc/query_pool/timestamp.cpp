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

TimestampQueryPoolManager::TimestampQueryPoolManager(const std::shared_ptr<DeviceManager>& pDeviceMgr,
                                                     const int queryCount, const float timestampPeriod)
    : pDeviceMgr_(pDeviceMgr), queryCount_(queryCount), queryIndex_(0), timestampPeriod_(timestampPeriod) {
    vk::QueryPoolCreateInfo queryPoolInfo;
    queryPoolInfo.setQueryType(vk::QueryType::eTimestamp);
    queryPoolInfo.setQueryCount(queryCount);

    auto& device = pDeviceMgr->getDevice();
    queryPool_ = device.createQueryPool(queryPoolInfo);
}

TimestampQueryPoolManager::~TimestampQueryPoolManager() noexcept {
    auto& device = pDeviceMgr_->getDevice();
    device.destroyQueryPool(queryPool_);
}

std::vector<float> TimestampQueryPoolManager::getElaspedTimes() const noexcept {
    std::vector<uint64_t> timestamps(queryIndex_);
    std::vector<float> elapsedTimes(queryIndex_ / 2);
    constexpr size_t valueSize = sizeof(decltype(timestamps)::value_type);

    const auto& device = pDeviceMgr_->getDevice();
    vk::Result queryResult =
        device.getQueryPoolResults(queryPool_, 0, queryIndex_, timestamps.size() * valueSize, (void*)timestamps.data(),
                                   valueSize, vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);

    for (const auto [idx, pair] : rgs::views::enumerate(timestamps | rgs::views::chunk(2))) {
        constexpr float ns2ms = 1e6;
        elapsedTimes[idx] = (float)(pair[1] - pair[0]) * timestampPeriod_ / ns2ms;
    }

    return elapsedTimes;
}

}  // namespace vkc
