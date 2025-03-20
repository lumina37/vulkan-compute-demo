#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

namespace vkc {

namespace fs = std::filesystem;

class ShaderManager {
public:
    ShaderManager(DeviceManager& deviceMgr, const fs::path& path);
    ShaderManager(DeviceManager& deviceMgr, std::span<std::byte> code);
    ~ShaderManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getShaderModule(this Self&& self) noexcept {
        return std::forward_like<Self>(self).shader_;
    }

private:
    DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::ShaderModule shader_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/shader.cpp"
#endif
