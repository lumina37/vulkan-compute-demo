#pragma once

#include <expected>
#include <memory>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"

namespace vkc {

class ProfilingLockBox {
    ProfilingLockBox(std::shared_ptr<DeviceBox>&& pDeviceBox) noexcept;

public:
    ProfilingLockBox(const ProfilingLockBox&) = delete;
    ProfilingLockBox(ProfilingLockBox&& rhs) noexcept = default;
    ~ProfilingLockBox() noexcept;

    [[nodiscard]] static std::expected<ProfilingLockBox, Error> create(std::shared_ptr<DeviceBox> pDeviceBox) noexcept;

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/query_pool/perf/lock.cpp"
#endif
