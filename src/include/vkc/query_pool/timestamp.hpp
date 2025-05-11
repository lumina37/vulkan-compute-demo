#pragma once

#include <expected>
#include <memory>
#include <utility>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class TimestampQueryPoolManager {
    TimestampQueryPoolManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::QueryPool queryPool, int queryCount,
                              float timestampPeriod) noexcept;

public:
    TimestampQueryPoolManager(TimestampQueryPoolManager&& rhs) noexcept;
    ~TimestampQueryPoolManager() noexcept;

    [[nodiscard]] static std::expected<TimestampQueryPoolManager, Error> create(
        std::shared_ptr<DeviceManager> pDeviceMgr, int queryCount, float timestampPeriod) noexcept;

    [[nodiscard]] int getQueryIndex() const noexcept { return queryIndex_; }
    [[nodiscard]] std::expected<void, Error> addQueryIndex() noexcept;
    void resetQueryIndex() noexcept;

    [[nodiscard]] int getQueryCount() const noexcept { return queryCount_; }

    template <typename Self>
    [[nodiscard]] auto&& getQueryPool(this Self&& self) noexcept {
        return std::forward_like<Self>(self).queryPool_;
    }

    [[nodiscard]] std::expected<std::vector<float>, Error> getElaspedTimes() const noexcept;

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
