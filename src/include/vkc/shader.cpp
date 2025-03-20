#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/helper/readfile.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/shader.hpp"
#endif

namespace vkc {

namespace fs = std::filesystem;

ShaderManager::ShaderManager(DeviceManager& deviceMgr, const fs::path& path) : deviceMgr_(deviceMgr) {
    const auto& code = readFile(path);

    vk::ShaderModuleCreateInfo shaderInfo;
    shaderInfo.setPCode((uint32_t*)code.data());
    shaderInfo.setCodeSize(code.size());

    auto& device = deviceMgr.getDevice();
    shader_ = device.createShaderModule(shaderInfo);
}

ShaderManager::ShaderManager(DeviceManager& deviceMgr, const std::span<std::byte> code) : deviceMgr_(deviceMgr) {
    vk::ShaderModuleCreateInfo shaderInfo;
    shaderInfo.setPCode((uint32_t*)code.data());
    shaderInfo.setCodeSize(code.size());

    auto& device = deviceMgr.getDevice();
    shader_ = device.createShaderModule(shaderInfo);
}

ShaderManager::~ShaderManager() noexcept {
    auto& device = deviceMgr_.getDevice();
    device.destroyShaderModule(shader_);
}

}  // namespace vkc
