#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <span>
#include <utility>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/query_pool/perf/box.hpp"
#endif

namespace vkc {

PerfQueryPoolBox::PerfQueryPoolBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::QueryPool queryPool, int queryCount,
                                   int perfCounterCount) noexcept
    : pDeviceBox_(std::move(pDeviceBox)),
      queryPool_(queryPool),
      queryCount_(queryCount),
      queryIndex_(0),
      perfCounterCount_(perfCounterCount) {}

PerfQueryPoolBox::PerfQueryPoolBox(PerfQueryPoolBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)),
      queryPool_(std::exchange(rhs.queryPool_, nullptr)),
      queryCount_(rhs.queryCount_),
      queryIndex_(rhs.queryIndex_),
      perfCounterCount_(rhs.perfCounterCount_) {}

PerfQueryPoolBox::~PerfQueryPoolBox() noexcept {
    if (queryPool_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    device.destroyQueryPool(queryPool_);
    queryPool_ = nullptr;
}

std::expected<PerfQueryPoolBox, Error> PerfQueryPoolBox::create(std::shared_ptr<DeviceBox> pDeviceBox,
                                                                uint32_t queueFamilyIndex, int queryCount,
                                                                std::span<uint32_t> perfCounterIndices) noexcept {
    vk::QueryPoolPerformanceCreateInfoKHR perfCreateInfo;
    perfCreateInfo.setQueueFamilyIndex(queueFamilyIndex);
    perfCreateInfo.setCounterIndices(perfCounterIndices);

    vk::QueryPoolCreateInfo queryPoolInfo;
    queryPoolInfo.setPNext(perfCreateInfo);
    queryPoolInfo.setQueryType(vk::QueryType::ePerformanceQueryKHR);
    queryPoolInfo.setQueryCount(queryCount);

    vk::Device device = pDeviceBox->getDevice();
    const auto [queryPoolRes, queryPool] = device.createQueryPool(queryPoolInfo);
    if (queryPoolRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, queryPoolRes}};
    }

    return PerfQueryPoolBox{std::move(pDeviceBox), queryPool, queryCount, (int)perfCounterIndices.size()};
}

std::expected<void, Error> PerfQueryPoolBox::addQueryIndex() noexcept {
    queryIndex_++;
    if (queryIndex_ > queryCount_) {
        auto errMsg = std::format("query index exceeds limits. max={}", queryCount_);
        return std::unexpected{Error{ECate::eVkC, ECode::eUnexValue, std::move(errMsg)}};
    }
    return {};
}

void PerfQueryPoolBox::resetQueryIndex() noexcept { queryIndex_ = 0; }

void PerfQueryPoolBox::hostReset() noexcept {
    // requires `VK_EXT_host_query_reset` extension
    vk::Device device = pDeviceBox_->getDevice();
    device.resetQueryPool(queryPool_, 0, queryCount_);
    resetQueryIndex();
}
}  // namespace vkc
