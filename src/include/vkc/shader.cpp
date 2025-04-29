#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <utility>

#include "vkc/helper/vulkan.hpp"

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/readfile.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/shader.hpp"
#endif

namespace vkc {

namespace fs = std::filesystem;

ShaderManager::ShaderManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::ShaderModule shader) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)), shader_(shader) {}

ShaderManager::ShaderManager(ShaderManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)), shader_(std::exchange(rhs.shader_, nullptr)) {}

ShaderManager::~ShaderManager() noexcept {
    if (shader_ == nullptr) return;
    auto& device = pDeviceMgr_->getDevice();
    device.destroyShaderModule(shader_);
    shader_ = nullptr;
}

std::expected<ShaderManager, Error> ShaderManager::create(std::shared_ptr<DeviceManager> pDeviceMgr,
                                                          std::span<const std::byte> code) noexcept {
    vk::ShaderModuleCreateInfo shaderInfo;
    shaderInfo.setPCode((uint32_t*)code.data());
    shaderInfo.setCodeSize(code.size());

    auto& device = pDeviceMgr->getDevice();
    const auto [shaderRes, shader] = device.createShaderModule(shaderInfo);
    if (shaderRes != vk::Result::eSuccess) {
        return std::unexpected{Error{shaderRes}};
    }

    return ShaderManager{std::move(pDeviceMgr), shader};
}

std::expected<ShaderManager, Error> ShaderManager::createFromPath(std::shared_ptr<DeviceManager> pDeviceMgr,
                                                                  const fs::path& path) noexcept {
    const auto& code = readFile(path);

    vk::ShaderModuleCreateInfo shaderInfo;
    shaderInfo.setPCode((uint32_t*)code.data());
    shaderInfo.setCodeSize(code.size());

    auto& device = pDeviceMgr->getDevice();
    const auto [shaderRes, shader] = device.createShaderModule(shaderInfo);
    if (shaderRes != vk::Result::eSuccess) {
        return std::unexpected{Error{shaderRes}};
    }

    return ShaderManager{std::move(pDeviceMgr), shader};
}

}  // namespace vkc
