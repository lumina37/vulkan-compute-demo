#include <array>
#include <expected>
#include <ranges>
#include <string>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/helper/defines.hpp"
#include "vkc/helper/error.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/instance.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

InstanceManager::InstanceManager(vk::Instance&& instance) noexcept : instance_(std::move(instance)) {}

InstanceManager::InstanceManager(InstanceManager&& rhs) noexcept : instance_(std::exchange(rhs.instance_, nullptr)) {}

InstanceManager::~InstanceManager() noexcept {
    if (instance_ == nullptr) return;
    instance_.destroy();
    instance_ = nullptr;
}

std::expected<InstanceManager, Error> InstanceManager::create() noexcept {
    vk::ApplicationInfo appInfo;
    appInfo.setPApplicationName("vk-compute-demo");
    appInfo.setApiVersion(VK_API_VERSION_1_0);

    vk::InstanceCreateInfo instInfo;
    instInfo.setPApplicationInfo(&appInfo);

    if constexpr (ENABLE_DEBUG) {
        const auto hasValidationLayer = [](const auto& layerProp) {
            return VALIDATION_LAYER_NAME == layerProp.layerName;
        };

        const bool hasValLayer = rgs::any_of(vk::enumerateInstanceLayerProperties(), hasValidationLayer);
        if (hasValLayer) {
            const std::array enabledLayers{VALIDATION_LAYER_NAME.data()};
            instInfo.setPEnabledLayerNames(enabledLayers);
        }
    }

    vk::Instance instance = vk::createInstance(instInfo);
    return InstanceManager{std::move(instance)};
}

}  // namespace vkc
