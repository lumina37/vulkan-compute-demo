#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/physical/features.hpp"
#endif

namespace vkc {

template class PhyDeviceFeatures_<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
                                  vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
                                  vk::PhysicalDeviceCooperativeMatrixFeaturesKHR>;

}  // namespace vkc
