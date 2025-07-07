#include <source_location>
#include <string>
#include <utility>

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/helper/error.hpp"
#endif

namespace vkc {

Error::Error() noexcept : cate(ECate::eSuccess), code(0) {}

}  // namespace vkc
