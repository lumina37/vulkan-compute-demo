#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <ranges>
#include <utility>
#include <vector>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/query_pool/timestamp.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

TimestampQueryPoolBox::TimestampQueryPoolBox(std::shared_ptr<DeviceBox>&& pDeviceBox,
                                                     vk::QueryPool queryPool, int queryCount,
                                                     float timestampPeriod) noexcept
    : pDeviceBox_(std::move(pDeviceBox)),
      queryPool_(queryPool),
      queryCount_(queryCount),
      queryIndex_(0),
      timestampPeriod_(timestampPeriod) {}

TimestampQueryPoolBox::TimestampQueryPoolBox(TimestampQueryPoolBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)),
      queryPool_(std::exchange(rhs.queryPool_, nullptr)),
      queryCount_(rhs.queryCount_),
      queryIndex_(rhs.queryIndex_),
      timestampPeriod_(rhs.timestampPeriod_) {}

TimestampQueryPoolBox::~TimestampQueryPoolBox() noexcept {
    if (queryPool_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    device.destroyQueryPool(queryPool_);
    queryPool_ = nullptr;
}

std::expected<TimestampQueryPoolBox, Error> TimestampQueryPoolBox::create(
    std::shared_ptr<DeviceBox> pDeviceBox, int queryCount, float timestampPeriod) noexcept {
    vk::QueryPoolCreateInfo queryPoolInfo;
    queryPoolInfo.setQueryType(vk::QueryType::eTimestamp);
    queryPoolInfo.setQueryCount(queryCount);

    vk::Device device = pDeviceBox->getDevice();
    const auto [queryPoolRes, queryPool] = device.createQueryPool(queryPoolInfo);
    if (queryPoolRes != vk::Result::eSuccess) {
        return std::unexpected{Error{queryPoolRes}};
    }

    return TimestampQueryPoolBox{std::move(pDeviceBox), queryPool, queryCount, timestampPeriod};
}

std::expected<void, Error> TimestampQueryPoolBox::addQueryIndex() noexcept {
    queryIndex_++;
    if (queryIndex_ > queryCount_) {
        auto errMsg = std::format("query index exceeds limits. max={}", queryCount_);
        return std::unexpected{Error{-1, std::move(errMsg)}};
    }
    return {};
}

void TimestampQueryPoolBox::resetQueryIndex() noexcept { queryIndex_ = 0; }

std::expected<std::vector<float>, Error> TimestampQueryPoolBox::getElaspedTimes() const noexcept {
    std::vector<uint64_t> timestamps(queryIndex_);
    std::vector<float> elapsedTimes(queryIndex_ / 2);
    constexpr size_t valueSize = sizeof(decltype(timestamps)::value_type);

    vk::Device device = pDeviceBox_->getDevice();
    const auto queryRes =
        device.getQueryPoolResults(queryPool_, 0, queryIndex_, timestamps.size() * valueSize, timestamps.data(),
                                   valueSize, vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    if (queryRes != vk::Result::eSuccess) {
        return std::unexpected{Error{queryRes}};
    }

    for (const auto [idx, pair] : rgs::views::enumerate(timestamps | rgs::views::chunk(2))) {
        constexpr float ns2ms = 1e6;
        elapsedTimes[idx] = (float)(pair[1] - pair[0]) * timestampPeriod_ / ns2ms;
    }

    return elapsedTimes;
}

}  // namespace vkc
