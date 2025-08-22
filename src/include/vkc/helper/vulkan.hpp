#pragma once

#include <vulkan/vulkan.hpp>

#include "vkc/helper/error.hpp"

namespace vkc {

std::expected<void, Error> initVulkan() noexcept;

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/helper/vulkan.cpp"
#endif
