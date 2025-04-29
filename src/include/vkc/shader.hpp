#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <utility>

#include "vkc/helper/vulkan.hpp"

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"

namespace vkc {

namespace fs = std::filesystem;

class ShaderManager {
    ShaderManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::ShaderModule shader) noexcept;

public:
    ShaderManager(ShaderManager&& rhs) noexcept;
    ~ShaderManager() noexcept;

    [[nodiscard]] static std::expected<ShaderManager, Error> create(std::shared_ptr<DeviceManager> pDeviceMgr,
                                                                    std::span<const std::byte> code) noexcept;
    [[nodiscard]] static std::expected<ShaderManager, Error> createFromPath(std::shared_ptr<DeviceManager> pDeviceMgr,
                                                                            const fs::path& path) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getShaderModule(this Self&& self) noexcept {
        return std::forward_like<Self>(self).shader_;
    }

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    vk::ShaderModule shader_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/shader.cpp"
#endif
