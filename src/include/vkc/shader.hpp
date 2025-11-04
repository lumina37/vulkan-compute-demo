#pragma once

#include <memory>
#include <span>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/std.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class ShaderBox {
    ShaderBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::ShaderModule shader) noexcept;

public:
    ShaderBox(const ShaderBox&) = delete;
    ShaderBox(ShaderBox&& rhs) noexcept;
    ~ShaderBox() noexcept;

    [[nodiscard]] static std::expected<ShaderBox, Error> create(std::shared_ptr<DeviceBox> pDeviceBox,
                                                                std::span<const std::byte> code) noexcept;

    [[nodiscard]] vk::ShaderModule getShaderModule() const noexcept { return shader_; }

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    vk::ShaderModule shader_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/shader.cpp"
#endif
