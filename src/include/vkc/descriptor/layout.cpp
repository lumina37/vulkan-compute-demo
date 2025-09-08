#include <expected>
#include <memory>
#include <span>
#include <utility>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/layout.hpp"
#endif

namespace vkc {

DescSetLayoutBox::DescSetLayoutBox(std::shared_ptr<DeviceBox>&& pDeviceBox,
                                   vk::DescriptorSetLayout descSetlayout) noexcept
    : pDdeviceBox_(std::move(pDeviceBox)), descSetlayout_(descSetlayout) {}

DescSetLayoutBox::DescSetLayoutBox(DescSetLayoutBox&& rhs) noexcept
    : pDdeviceBox_(std::move(rhs.pDdeviceBox_)), descSetlayout_(std::exchange(rhs.descSetlayout_, nullptr)) {}

DescSetLayoutBox::~DescSetLayoutBox() noexcept {
    if (descSetlayout_ == nullptr) return;
    vk::Device device = pDdeviceBox_->getDevice();
    device.destroyDescriptorSetLayout(descSetlayout_);
    descSetlayout_ = nullptr;
}

std::expected<DescSetLayoutBox, Error> DescSetLayoutBox::create(
    std::shared_ptr<DeviceBox> pDeviceBox, std::span<const vk::DescriptorSetLayoutBinding> bindings) noexcept {
    vk::DescriptorSetLayoutCreateInfo layoutInfo;
    layoutInfo.setBindings(bindings);

    vk::Device device = pDeviceBox->getDevice();
    const auto [descSetlayoutRes, descSetlayout] = device.createDescriptorSetLayout(layoutInfo);
    if (descSetlayoutRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, descSetlayoutRes}};
    }

    return DescSetLayoutBox{std::move(pDeviceBox), descSetlayout};
}

}  // namespace vkc
