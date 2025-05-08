#pragma once

#include "vkc/device/concepts.hpp"
#include "vkc/device/extensions.hpp"
#include "vkc/device/instance.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/device/physical.hpp"
#include "vkc/device/props.hpp"
#include "vkc/device/queue.hpp"
#include "vkc/device/queue_family.hpp"
#include "vkc/device/set.hpp"

namespace vkc {

using PhyDeviceWithProps = PhyDeviceWithProps_<PhyDeviceProps>;
using PhyDeviceSet = PhyDeviceSet_<PhyDeviceProps>;

}  // namespace vkc
