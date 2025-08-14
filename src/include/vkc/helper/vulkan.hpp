#pragma once

#define VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vulkan/vulkan.hpp>

#include "vkc/helper/error.hpp"

namespace vkc {

std::expected<void, Error> initVulkan() noexcept;

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/helper/vulkan.cpp"
#endif
