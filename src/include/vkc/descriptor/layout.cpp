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

DescSetLayoutManager::DescSetLayoutManager(std::shared_ptr<DeviceManager>&& pDeviceMgr,
                                           vk::DescriptorSetLayout descSetlayout) noexcept
    : pDdeviceMgr_(std::move(pDeviceMgr)), descSetlayout_(descSetlayout) {}

DescSetLayoutManager::DescSetLayoutManager(DescSetLayoutManager&& rhs) noexcept
    : pDdeviceMgr_(std::move(rhs.pDdeviceMgr_)), descSetlayout_(std::exchange(rhs.descSetlayout_, nullptr)) {}

DescSetLayoutManager::~DescSetLayoutManager() noexcept {
    if (descSetlayout_ == nullptr) return;
    vk::Device device = pDdeviceMgr_->getDevice();
    device.destroyDescriptorSetLayout(descSetlayout_);
    descSetlayout_ = nullptr;
}

std::expected<DescSetLayoutManager, Error> DescSetLayoutManager::create(
    std::shared_ptr<DeviceManager> pDeviceMgr, std::span<const vk::DescriptorSetLayoutBinding> bindings) noexcept {
    vk::DescriptorSetLayoutCreateInfo layoutInfo;
    layoutInfo.setBindings(bindings);

    vk::Device device = pDeviceMgr->getDevice();
    const auto [descSetlayoutRes, descSetlayout] = device.createDescriptorSetLayout(layoutInfo);
    if (descSetlayoutRes != vk::Result::eSuccess) {
        return std::unexpected{Error{descSetlayoutRes}};
    }

    return DescSetLayoutManager{std::move(pDeviceMgr), descSetlayout};
}

}  // namespace vkc
