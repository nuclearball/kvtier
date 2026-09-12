#ifndef SC_HOT_HPP
#define SC_HOT_HPP

#include "radix.hpp"

namespace sc {

bool hot_check_path(const radix_tree *t, const uint64_t *hashes, int depth,
                    uint32_t threshold);

void hot_decay(radix_tree *t);

} // namespace sc

#endif
