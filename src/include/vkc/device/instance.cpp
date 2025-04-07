#include <cstring>
#include <ranges>

#include <vulkan/vulkan.hpp>

#include "vkc/helper/defines.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/instance.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

InstanceManager::InstanceManager() {
    constexpr bool ENABLE_VALIDATION_LAYER = ENABLE_DEBUG;

    vk::ApplicationInfo appInfo;
    appInfo.setPApplicationName("vk-compute-demo");
    appInfo.setApiVersion(VK_API_VERSION_1_1);

    vk::InstanceCreateInfo instInfo;
    instInfo.setPApplicationInfo(&appInfo);

    if constexpr (ENABLE_VALIDATION_LAYER) {
        const auto hasValidationLayer = [](const auto& layerProp) {
            return std::strcmp(VALIDATION_LAYER_NAME, layerProp.layerName) != 0;
        };

        const bool hasValLayer = rgs::any_of(vk::enumerateInstanceLayerProperties(), hasValidationLayer);
        if (hasValLayer) {
            instInfo.setPEnabledLayerNames({VALIDATION_LAYER_NAME});
        }
    }

    instance_ = vk::createInstance(instInfo);
};

InstanceManager::~InstanceManager() noexcept { instance_.destroy(); }

}  // namespace vkc
