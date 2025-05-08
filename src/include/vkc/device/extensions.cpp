#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/extensions.hpp"
#endif

namespace vkc {

template class ExtEntry_<vk::ExtensionProperties>;
template class ExtEntry_<vk::LayerProperties>;

template class OrderedExtEntries_<vk::ExtensionProperties>;
template class OrderedExtEntries_<vk::LayerProperties>;

}  // namespace vkc
