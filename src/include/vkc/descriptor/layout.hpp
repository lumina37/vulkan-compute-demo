#pragma once

#include <memory>
#include <span>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/std.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class DescSetLayoutBox {
    DescSetLayoutBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::DescriptorSetLayout descSetlayout) noexcept;

public:
    DescSetLayoutBox(const DescSetLayoutBox&) = delete;
    DescSetLayoutBox(DescSetLayoutBox&& rhs) noexcept;
    ~DescSetLayoutBox() noexcept;

    [[nodiscard]] static std::expected<DescSetLayoutBox, Error> create(
        std::shared_ptr<DeviceBox> pDeviceBox, std::span<const vk::DescriptorSetLayoutBinding> bindings) noexcept;

    [[nodiscard]] vk::DescriptorSetLayout getDescSetLayout() const noexcept { return descSetlayout_; }
    [[nodiscard]] static vk::DescriptorSetLayout exposeDescSetLayout(const DescSetLayoutBox& box) noexcept {
        return box.getDescSetLayout();
    }

private:
    std::shared_ptr<DeviceBox> pDdeviceBox_;

    vk::DescriptorSetLayout descSetlayout_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/layout.cpp"
#endif
