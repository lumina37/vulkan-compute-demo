#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/extensions.hpp"
#endif

namespace vkc {

template class ExtEntry_<vk::ExtensionProperties>;
template class ExtEntry_<vk::LayerProperties>;

template class ExtEntries_<vk::ExtensionProperties>;
template class ExtEntries_<vk::LayerProperties>;

}  // namespace vkc
