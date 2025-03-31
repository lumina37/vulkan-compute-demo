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

ShaderManager::ShaderManager(const std::shared_ptr<DeviceManager>& pDeviceMgr, const fs::path& path)
    : pDeviceMgr_(pDeviceMgr) {
    const auto& code = readFile(path);

    vk::ShaderModuleCreateInfo shaderInfo;
    shaderInfo.setPCode((uint32_t*)code.data());
    shaderInfo.setCodeSize(code.size());

    auto& device = pDeviceMgr->getDevice();
    shader_ = device.createShaderModule(shaderInfo);
}

ShaderManager::ShaderManager(const std::shared_ptr<DeviceManager>& pDeviceMgr, const std::span<std::byte> code)
    : pDeviceMgr_(pDeviceMgr) {
    vk::ShaderModuleCreateInfo shaderInfo;
    shaderInfo.setPCode((uint32_t*)code.data());
    shaderInfo.setCodeSize(code.size());

    auto& device = pDeviceMgr->getDevice();
    shader_ = device.createShaderModule(shaderInfo);
}

ShaderManager::~ShaderManager() noexcept {
    auto& device = pDeviceMgr_->getDevice();
    device.destroyShaderModule(shader_);
}

}  // namespace vkc
