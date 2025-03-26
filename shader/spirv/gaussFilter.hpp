#pragma once

#include <cstddef>
#include <span>

namespace shader {

namespace _spirv::gaussFilter {

#include "spirv/gaussFilter.hlsl.h"

}

static const std::span gaussFilterSpirvCode{(std::byte*)_spirv::gaussFilter::g_main,
                                             sizeof(_spirv::gaussFilter::g_main)};

}  // namespace vkc
