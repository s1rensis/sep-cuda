#include <stdlib.h>
#include <string.h>

#include <atomic>

#include "sep_internal.h"

namespace {

thread_local sep_cuda_background_profile g_last_background_profile{};
std::atomic<int> g_profile_enabled{-1};

int parse_profile_env() {
  const char *value = getenv("SEPCUDA_PROFILE");
  if (value == nullptr || value[0] == '\0') {
    return 0;
  }
  return !(strcmp(value, "0") == 0 || strcmp(value, "false") == 0 || strcmp(value, "FALSE") == 0);
}

}  // namespace

extern "C" int sepcuda_profile_enabled(void) {
  int cached = g_profile_enabled.load(std::memory_order_acquire);
  if (cached >= 0) {
    return cached;
  }

  const int enabled = parse_profile_env();
  int expected = -1;
  if (g_profile_enabled.compare_exchange_strong(expected, enabled, std::memory_order_acq_rel)) {
    return enabled;
  }
  return g_profile_enabled.load(std::memory_order_acquire);
}

extern "C" void sepcuda_profile_reset_background(sep_cuda_background_profile *profile) {
  if (profile == nullptr) {
    return;
  }
  memset(profile, 0, sizeof(*profile));
  profile->enabled = sepcuda_profile_enabled();
}

extern "C" void sepcuda_profile_commit_background(const sep_cuda_background_profile *profile) {
  if (profile == nullptr) {
    memset(&g_last_background_profile, 0, sizeof(g_last_background_profile));
    return;
  }
  g_last_background_profile = *profile;
}

extern "C" SEP_API int sep_cuda_profile_get_last_background(sep_cuda_background_profile *profile) {
  if (profile == nullptr) {
    return 0;
  }
  *profile = g_last_background_profile;
  return profile->enabled;
}
