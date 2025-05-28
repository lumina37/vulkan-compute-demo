#pragma once

#include <expected>
#include <memory>
#include <span>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class DescSetLayoutManager {
    DescSetLayoutManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::DescriptorSetLayout descSetlayout) noexcept;

public:
    DescSetLayoutManager(DescSetLayoutManager&& rhs) noexcept;
    ~DescSetLayoutManager() noexcept;

    [[nodiscard]] static std::expected<DescSetLayoutManager, Error> create(
        std::shared_ptr<DeviceManager> pDeviceMgr, std::span<const vk::DescriptorSetLayoutBinding> bindings) noexcept;

    [[nodiscard]] vk::DescriptorSetLayout getDescSetLayout() const noexcept { return descSetlayout_; }

private:
    std::shared_ptr<DeviceManager> pDdeviceMgr_;

    vk::DescriptorSetLayout descSetlayout_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/layout.cpp"
#endif
