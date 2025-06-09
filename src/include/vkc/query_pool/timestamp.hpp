#pragma once

#include <expected>
#include <memory>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class TimestampQueryPoolBox {
    TimestampQueryPoolBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::QueryPool queryPool, int queryCount,
                          float timestampPeriod) noexcept;

public:
    TimestampQueryPoolBox(const TimestampQueryPoolBox&) = delete;
    TimestampQueryPoolBox(TimestampQueryPoolBox&& rhs) noexcept;
    ~TimestampQueryPoolBox() noexcept;

    [[nodiscard]] static std::expected<TimestampQueryPoolBox, Error> create(std::shared_ptr<DeviceBox> pDeviceBox,
                                                                            int queryCount,
                                                                            float timestampPeriod) noexcept;

    [[nodiscard]] int getQueryIndex() const noexcept { return queryIndex_; }
    [[nodiscard]] std::expected<void, Error> addQueryIndex() noexcept;
    void resetQueryIndex() noexcept;

    [[nodiscard]] int getQueryCount() const noexcept { return queryCount_; }

    [[nodiscard]] vk::QueryPool getQueryPool() const noexcept { return queryPool_; }

    [[nodiscard]] std::expected<std::vector<float>, Error> getElaspedTimes() const noexcept;

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    vk::QueryPool queryPool_;
    int queryCount_;
    int queryIndex_;
    float timestampPeriod_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/query_pool/timestamp.cpp"
#endif
