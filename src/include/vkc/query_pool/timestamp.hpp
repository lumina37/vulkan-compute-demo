#pragma once

#include <memory>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

namespace vkc {

class TimestampQueryPoolManager {
public:
    TimestampQueryPoolManager(const std::shared_ptr<DeviceManager>& pDeviceMgr, int queryCount, float timestampPeriod);
    ~TimestampQueryPoolManager() noexcept;

    int getQueryIndex() const noexcept { return queryIndex_; }
    void addQueryIndex() noexcept { queryIndex_++; }
    void resetQueryIndex() noexcept { queryIndex_ = 0; }

    [[nodiscard]] int getQueryCount() const noexcept { return queryCount_; }

    template <typename Self>
    [[nodiscard]] auto&& getQueryPool(this Self&& self) noexcept {
        return std::forward_like<Self>(self).queryPool_;
    }

    [[nodiscard]] std::vector<float> getElaspedTimes() const noexcept;

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    vk::QueryPool queryPool_;
    int queryCount_;
    int queryIndex_;
    float timestampPeriod_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/query_pool/timestamp.cpp"
#endif
