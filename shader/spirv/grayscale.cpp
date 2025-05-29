#include <cstddef>
#include <span>

#include "spirv/grayscale.hpp"

namespace shader::grayscale {

namespace ro {

namespace _detail {
#include "spirv/grayscale/ro.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace ro

namespace rw {

namespace _detail {
#include "spirv/grayscale/rw.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace rw

}  // namespace shader::grayscale
