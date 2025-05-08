#include <array>
#include <expected>
#include <ranges>
#include <string>
#include <utility>

#include "vkc/device/extensions.hpp"
#include "vkc/helper/defines.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/instance.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

InstanceManager::InstanceManager(vk::Instance instance) noexcept : instance_(instance) {}

InstanceManager::InstanceManager(InstanceManager&& rhs) noexcept : instance_(std::exchange(rhs.instance_, nullptr)) {}

InstanceManager::~InstanceManager() noexcept {
    if (instance_ == nullptr) return;
    instance_.destroy();
    instance_ = nullptr;
}

std::expected<InstanceManager, Error> InstanceManager::create() noexcept {
    vk::ApplicationInfo appInfo;
    appInfo.setPApplicationName("vk-compute-demo");
    appInfo.setApiVersion(VK_API_VERSION_1_1);

    vk::InstanceCreateInfo instInfo;
    instInfo.setPApplicationInfo(&appInfo);

    if constexpr (ENABLE_DEBUG) {
        auto [layerPropsRes, layerProps] = vk::enumerateInstanceLayerProperties();
        if (layerPropsRes != vk::Result::eSuccess) {
            return std::unexpected{Error{layerPropsRes}};
        }

        auto layerEntriesRes = ExtEntries_<vk::LayerProperties>::create(std::move(layerProps));
        if (!layerEntriesRes) return std::unexpected{std::move(layerEntriesRes.error())};
        auto layerEntries = std::move(layerEntriesRes.value());

        constexpr std::string_view VALIDATION_LAYER_NAME{"VK_LAYER_KHRONOS_validation"};
        const bool hasValLayer = layerEntries.has(VALIDATION_LAYER_NAME);
        if (hasValLayer) {
            const std::array enabledLayers{VALIDATION_LAYER_NAME.data()};
            instInfo.setPEnabledLayerNames(enabledLayers);
        }
    }

    const auto [instanceRes, instance] = vk::createInstance(instInfo);
    if (instanceRes != vk::Result::eSuccess) {
        return std::unexpected{Error{instanceRes}};
    }

    return InstanceManager{instance};
}

}  // namespace vkc
