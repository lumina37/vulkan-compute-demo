#include <cstddef>
#include <span>

#include "spirv/flash_attention2.hpp"

namespace shader::flash_attention2 {

namespace v0 {

namespace _detail {
#include "spirv/flash_attention2/v0.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v0

}  // namespace shader::flash_attention2
