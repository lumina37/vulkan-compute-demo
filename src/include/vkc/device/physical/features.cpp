#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/physical/features.hpp"
#endif

namespace vkc {

template class PhyDeviceFeatures_<vk::PhysicalDeviceFeatures2, vk::PhysicalDevicePerformanceQueryFeaturesKHR,
                                  vk::PhysicalDeviceShaderFloat16Int8Features,
                                  vk::PhysicalDeviceHostQueryResetFeatures>;

}  // namespace vkc
