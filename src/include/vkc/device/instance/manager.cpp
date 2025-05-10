#include <array>
#include <expected>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "vkc/helper/defines.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/instance/manager.hpp"
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
    constexpr std::string_view validationLayerName{"VK_LAYER_KHRONOS_validation"};
    if constexpr (ENABLE_DEBUG) {
        constexpr std::array enableLayerNames{validationLayerName};
        return createWithExts({}, enableLayerNames);
    } else {
        return createWithExts({}, {});
    }
}

std::expected<InstanceManager, Error> InstanceManager::createWithExts(
    std::span<const std::string_view> enableExtNames, std::span<const std::string_view> enableLayerNames) noexcept {
    vk::ApplicationInfo appInfo;
    appInfo.setPApplicationName("vk-compute-demo");
    appInfo.setApiVersion(vk::ApiVersion11);

    vk::InstanceCreateInfo instInfo;
    instInfo.setPApplicationInfo(&appInfo);

    auto enabledPExtNames = enableExtNames | rgs::views::transform([](std::string_view name) { return name.data(); }) |
                            rgs::to<std::vector>();
    instInfo.setPEnabledExtensionNames(enabledPExtNames);

    auto enabledPLayerNames = enableLayerNames |
                              rgs::views::transform([](std::string_view name) { return name.data(); }) |
                              rgs::to<std::vector>();
    instInfo.setPEnabledLayerNames(enabledPLayerNames);

    const auto [instanceRes, instance] = vk::createInstance(instInfo);
    if (instanceRes != vk::Result::eSuccess) {
        return std::unexpected{Error{instanceRes}};
    }

    return InstanceManager{instance};
}

}  // namespace vkc
