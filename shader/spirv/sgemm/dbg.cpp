#include <cstddef>
#include <span>

#include "spirv/sgemm/dbg.hpp"

namespace shader::sgemm::dbg {

namespace wt0 {

namespace _detail {
#include "spirv/sgemm/dbg/wt0.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace wt0

namespace wt1 {

namespace _detail {
#include "spirv/sgemm/dbg/wt1.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace wt1

}  // namespace shader::sgemm::dbg
