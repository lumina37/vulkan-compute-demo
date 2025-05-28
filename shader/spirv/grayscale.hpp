#pragma once

#include <cstddef>
#include <span>

namespace shader::grayscale {

namespace _detail {
#include "spirv/grayscale.h"
}

static const std::span code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace shader::grayscale
