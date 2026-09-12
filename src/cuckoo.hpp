#ifndef SC_CUCKOO_HPP
#define SC_CUCKOO_HPP

#include <cstddef>
#include <cstdint>

namespace sc {

struct kv_cuckoo;

int kv_cuckoo_init(kv_cuckoo **out, uint32_t n_entries);
void kv_cuckoo_destroy(kv_cuckoo *cf);

int kv_cuckoo_insert(kv_cuckoo *cf, uint64_t key);
bool kv_cuckoo_lookup(const kv_cuckoo *cf, uint64_t key);
bool kv_cuckoo_delete(kv_cuckoo *cf, uint64_t key);

uint32_t kv_cuckoo_count(const kv_cuckoo *cf);
uint32_t kv_cuckoo_capacity(const kv_cuckoo *cf);

} // namespace sc

#endif
