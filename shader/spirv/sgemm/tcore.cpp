#include <cstddef>
#include <span>

#include "spirv/sgemm/tcore.hpp"

namespace shader::sgemm::tcore {

namespace v0 {

namespace _detail {
#include "spirv/sgemm/tcore/v0.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v0

namespace v1 {

namespace _detail {
#include "spirv/sgemm/tcore/v1.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v1

namespace v2 {

namespace _detail {
#include "spirv/sgemm/tcore/v2.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v2

}  // namespace shader::sgemm::tcore
