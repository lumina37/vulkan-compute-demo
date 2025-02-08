#pragma once

#include <cstdint>
#include <filesystem>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/helper/readfile.hpp"

namespace vkc {

namespace fs = std::filesystem;

class ShaderManager {
public:
    inline ShaderManager(const DeviceManager& deviceMgr, const fs::path& path);
    inline ~ShaderManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getShaderModule(this Self& self) noexcept {
        return std::forward_like<Self>(self).shader_;
    }

private:
    const DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::ShaderModule shader_;
};

ShaderManager::ShaderManager(const DeviceManager& deviceMgr, const fs::path& path) : deviceMgr_(deviceMgr) {
    const auto& code = readFile(path);

    vk::ShaderModuleCreateInfo shaderInfo;
    shaderInfo.setPCode((uint32_t*)code.data());
    shaderInfo.setCodeSize(code.size());

    const auto& device = deviceMgr.getDevice();
    shader_ = device.createShaderModule(shaderInfo);
}

ShaderManager::~ShaderManager() noexcept {
    const auto& device = deviceMgr_.getDevice();
    device.destroyShaderModule(shader_);
}

}  // namespace vkc
