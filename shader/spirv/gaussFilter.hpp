#pragma once

#include <cstddef>
#include <span>

namespace shader {

namespace _spirv::gaussFilterV1 {

#include "spirv/gaussFilterV1.hlsl.h"

}

static const std::span gaussFilterV1SpirvCode{(std::byte*)_spirv::gaussFilterV1::g_main,
                                              sizeof(_spirv::gaussFilterV1::g_main)};

}  // namespace shader
