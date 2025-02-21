#pragma once

#include <concepts>
#include <cstddef>
#include <span>

#include <vulkan/vulkan.hpp>

namespace vkc {

template <typename TPc_>
    requires std::is_trivially_copyable_v<TPc_>
class PushConstantManager {
public:
    using TPc = TPc_;

    constexpr inline PushConstantManager(const TPc pushConstant);

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
    requires std::is_trivially_copyable_v<TPc>
constexpr PushConstantManager<TPc>::PushConstantManager(const TPc pushConstant) : pushConstant_(pushConstant) {
    pushConstantRange_.setStageFlags(vk::ShaderStageFlagBits::eCompute);
    pushConstantRange_.setSize(sizeof(TPc));
}

}  // namespace vkc
