#include <cstddef>
#include <span>

#include "spirv/sgemm.hpp"

namespace shader::sgemm {

namespace v0 {

namespace _detail {
#include "spirv/sgemm/v0.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v0

}  // namespace shader::sgemm
