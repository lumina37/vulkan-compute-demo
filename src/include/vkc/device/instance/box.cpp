#include <array>
#include <ranges>
#include <span>
#include <string>
#include <vector>

#include "vkc/helper/defines.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/std.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/instance/box.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

InstanceBox::InstanceBox(vk::Instance instance) noexcept : instance_(instance) {}

InstanceBox::InstanceBox(InstanceBox&& rhs) noexcept : instance_(std::exchange(rhs.instance_, nullptr)) {}

InstanceBox::~InstanceBox() noexcept {
    if (instance_ == nullptr) return;
    instance_.destroy();
    instance_ = nullptr;
}

std::expected<InstanceBox, Error> InstanceBox::create() noexcept {
    constexpr std::string_view validationLayerName{"VK_LAYER_KHRONOS_validation"};
    if constexpr (DEBUG_ENABLED) {
        constexpr std::array enableLayerNames{validationLayerName};
        return createWithExts({}, enableLayerNames);
    } else {
        return createWithExts({}, {});
    }
}

std::expected<InstanceBox, Error> InstanceBox::createWithExts(
    std::span<const std::string_view> enableExtNames, std::span<const std::string_view> enableLayerNames) noexcept {
    vk::ApplicationInfo appInfo;
    appInfo.setPApplicationName("vk-demo");
    appInfo.setApiVersion(vk::ApiVersion13);

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
        return std::unexpected{Error{ECate::eVk, instanceRes}};
    }

    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);

    return InstanceBox{instance};
}

}  // namespace vkc
