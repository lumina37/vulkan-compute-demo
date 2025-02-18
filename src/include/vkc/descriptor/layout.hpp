#pragma once

#include <span>
#include <utility>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/binding.hpp"
#include "vkc/device/logical.hpp"

namespace vkc {

class DescSetLayoutManager {
public:
    inline DescSetLayoutManager(const DeviceManager& deviceMgr,
                                const std::span<DescSetLayoutBindingManager> bindingMgrs);
    inline ~DescSetLayoutManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getDescSetLayout(this Self& self) noexcept {
        return std::forward_like<Self>(self).descSetlayout_;
    }

private:
    const DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::DescriptorSetLayout descSetlayout_;
};

DescSetLayoutManager::DescSetLayoutManager(const DeviceManager& deviceMgr,
                                           const std::span<DescSetLayoutBindingManager> bindingMgrs)
    : deviceMgr_(deviceMgr) {
    vk::DescriptorSetLayoutCreateInfo layoutInfo;
    const auto& bindings =
        bindingMgrs |
        std::views::transform([](const DescSetLayoutBindingManager& bindingMgr) { return bindingMgr.getBinding(); }) |
        std::ranges::to<std::vector>();
    layoutInfo.setBindings(bindings);

    const auto& device = deviceMgr.getDevice();
    descSetlayout_ = device.createDescriptorSetLayout(layoutInfo);
}

DescSetLayoutManager::~DescSetLayoutManager() noexcept {
    const auto& device = deviceMgr_.getDevice();
    device.destroyDescriptorSetLayout(descSetlayout_);
}

}  // namespace vkc
