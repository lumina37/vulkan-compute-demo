#pragma once

#include "vkc/device/concepts.hpp"
#include "vkc/device/extensions.hpp"
#include "vkc/device/instance.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/device/physical.hpp"
#include "vkc/device/queue_family.hpp"

namespace vkc {

using PhyDeviceWithProps = PhyDeviceWithProps_<DefaultPhyDeviceProps>;
using PhyDeviceSet = PhyDeviceSet_<DefaultPhyDeviceProps>;

}  // namespace vkc
