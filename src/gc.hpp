#ifndef SC_GC_HPP
#define SC_GC_HPP

#include "cache.hpp"

namespace sc {

int gc_start(cache *c);
void gc_stop(cache *c);

} // namespace sc

#endif
