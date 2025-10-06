#include "vkc/helper/vulkan.hpp"
#include "vkc/helper/error.hpp"

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

namespace vkc {

std::expected<void, Error> initVulkan() noexcept {
    static vk::detail::DynamicLoader dl;
    auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    if (vkGetInstanceProcAddr == nullptr) {
        return std::unexpected{Error{ECate::eVkC, ECode::eNoSupport, "Vulkan dynlib not exists"}};
    }
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
    return {};
}

}  // namespace vkc
