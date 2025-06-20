#pragma once

#include "vkc/helper/vulkan.hpp"

namespace vkc {

template <typename TPc_>
class PushConstantBox {
public:
    using TPc = TPc_;

    constexpr PushConstantBox(TPc pushConstant,
                              vk::ShaderStageFlags stage = vk::ShaderStageFlagBits::eCompute) noexcept;

    template <typename Self>
    [[nodiscard]] constexpr auto&& getPushConstantRange(this Self&& self) noexcept {
        return std::forward_like<Self>(self).pushConstantRange_;
    }

    [[nodiscard]] const void* getPPushConstant() const noexcept { return (void*)&pushConstant_; }

private:
    TPc pushConstant_;
    vk::PushConstantRange pushConstantRange_;
};

template <typename TPc>
constexpr PushConstantBox<TPc>::PushConstantBox(const TPc pushConstant, const vk::ShaderStageFlags stage) noexcept
    : pushConstant_(pushConstant) {
    pushConstantRange_.setStageFlags(stage);
    pushConstantRange_.setSize(sizeof(TPc));
}

}  // namespace vkc
