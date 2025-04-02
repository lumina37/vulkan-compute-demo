#pragma once

#include <cstddef>
#include <span>

namespace shader {

namespace _spirv::grayscale {

#include "spirv/grayscale.hlsl.h"

}

static const std::span grayscaleSpirvCode{(std::byte*)_spirv::grayscale::g_main, sizeof(_spirv::grayscale::g_main)};

}  // namespace shader
