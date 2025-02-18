#pragma once

#include <utility>

#include <vulkan/vulkan.hpp>

namespace vkc {

class DescSetLayoutBindingManager {
public:
    inline DescSetLayoutBindingManager(const int index, const vk::DescriptorType type) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getBinding(this Self& self) noexcept {
        return std::forward_like<Self>(self).binding_;
    }

private:
    vk::DescriptorSetLayoutBinding binding_;
};

DescSetLayoutBindingManager::DescSetLayoutBindingManager(const int index, const vk::DescriptorType type) noexcept {
    binding_.setBinding(index);
    binding_.setDescriptorCount(1);
    binding_.setDescriptorType(type);
    binding_.setStageFlags(vk::ShaderStageFlagBits::eCompute);
}

}  // namespace vkc
