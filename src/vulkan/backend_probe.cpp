#include "ggml-backend.h"
#include "ggml-vulkan.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace {

std::string env_device_hint() {
    const char* value = std::getenv("STRATA_VULKAN_DEVICE");
    return value && *value ? value : "6800 XT";
}

std::string escape_json(const char* input) {
    std::string out;
    for (const char* p = input; *p; ++p) {
        switch (*p) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += *p; break;
        }
    }
    return out;
}

}  // namespace

int main() {
    const int count = ggml_backend_vk_get_device_count();
    if (count <= 0) {
        std::cerr << "STRATA_VULKAN_PROBE=FAIL no Vulkan devices\n";
        return 2;
    }

    const std::string hint = env_device_hint();
    int selected = -1;
    char description[512] = {};
    for (int i = 0; i < count; ++i) {
        char current[512] = {};
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_vk_get_device_description(i, current, sizeof(current));
        ggml_backend_vk_get_device_memory(i, &free_bytes, &total_bytes);
        std::cout << "VULKAN_DEVICE index=" << i
                  << " description=\"" << current << "\""
                  << " free_bytes=" << free_bytes
                  << " total_bytes=" << total_bytes << "\n";
        if (selected < 0 && std::strstr(current, hint.c_str()) != nullptr) {
            selected = i;
            std::strncpy(description, current, sizeof(description) - 1);
        }
    }

    if (selected < 0) {
        std::cerr << "STRATA_VULKAN_PROBE=FAIL target device hint not found: " << hint << "\n";
        return 3;
    }

    ggml_backend_t backend = ggml_backend_vk_init(static_cast<size_t>(selected));
    if (!backend || !ggml_backend_is_vk(backend)) {
        std::cerr << "STRATA_VULKAN_PROBE=FAIL backend init failed\n";
        if (backend) ggml_backend_free(backend);
        return 4;
    }

    constexpr size_t probe_bytes = 64ull * 1024ull * 1024ull;
    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(backend, probe_bytes);
    if (!buffer || ggml_backend_buffer_get_size(buffer) < probe_bytes) {
        std::cerr << "STRATA_VULKAN_PROBE=FAIL 64MiB device buffer allocation failed\n";
        if (buffer) ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        return 5;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_vk_get_device_memory(selected, &free_bytes, &total_bytes);
    std::cout << "{\"status\":\"PASS\",\"backend\":\"ggml-vulkan\","
              << "\"device_index\":" << selected << ","
              << "\"device\":\"" << escape_json(description) << "\","
              << "\"free_bytes\":" << free_bytes << ","
              << "\"total_bytes\":" << total_bytes << ","
              << "\"probe_buffer_bytes\":" << ggml_backend_buffer_get_size(buffer)
              << "}\n";
    std::cout << "STRATA_VULKAN_PROBE=PASS\n";

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return 0;
}
