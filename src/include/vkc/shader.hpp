#pragma once

#include <cstddef>
#include <expected>
#include <memory>
#include <span>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class ShaderManager {
    ShaderManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::ShaderModule shader) noexcept;

public:
    ShaderManager(ShaderManager&& rhs) noexcept;
    ~ShaderManager() noexcept;

    [[nodiscard]] static std::expected<ShaderManager, Error> create(std::shared_ptr<DeviceManager> pDeviceMgr,
                                                                    std::span<const std::byte> code) noexcept;

    [[nodiscard]] vk::ShaderModule getShaderModule() const noexcept { return shader_; }

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    vk::ShaderModule shader_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/shader.cpp"
#endif
