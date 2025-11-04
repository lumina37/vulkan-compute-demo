#include <memory>
#include <span>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/std.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/shader.hpp"
#endif

namespace vkc {

ShaderBox::ShaderBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::ShaderModule shader) noexcept
    : pDeviceBox_(std::move(pDeviceBox)), shader_(shader) {}

ShaderBox::ShaderBox(ShaderBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)), shader_(std::exchange(rhs.shader_, nullptr)) {}

ShaderBox::~ShaderBox() noexcept {
    if (shader_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    device.destroyShaderModule(shader_);
    shader_ = nullptr;
}

std::expected<ShaderBox, Error> ShaderBox::create(std::shared_ptr<DeviceBox> pDeviceBox,
                                                  std::span<const std::byte> code) noexcept {
    vk::ShaderModuleCreateInfo shaderInfo;
    shaderInfo.setPCode((uint32_t*)code.data());
    shaderInfo.setCodeSize(code.size());

    vk::Device device = pDeviceBox->getDevice();
    const auto [shaderRes, shader] = device.createShaderModule(shaderInfo);
    if (shaderRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, shaderRes}};
    }

    return ShaderBox{std::move(pDeviceBox), shader};
}

}  // namespace vkc
