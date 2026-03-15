// Stub for OpenMesh IO - not needed for mesh kernel operations
#ifndef OPENMESH_STORERESTORE_HH
#define OPENMESH_STORERESTORE_HH
#include <iosfwd>
#include <cstddef>
#include <cstdint>
#include <string>
#include <numeric>
namespace OpenMesh { namespace IO {

// Standard integer type aliases expected by PropertyCreator
using int8_t   = ::int8_t;
using int16_t  = ::int16_t;
using int32_t  = ::int32_t;
using int64_t  = ::int64_t;
using uint8_t  = ::uint8_t;
using uint16_t = ::uint16_t;
using uint32_t = ::uint32_t;
using uint64_t = ::uint64_t;

const size_t UnknownSize = size_t(-1);

template<typename T>
bool is_streamable(void) { return false; }

template<typename T>
inline size_t store(std::ostream&, const T&, bool = false, bool = false) { return 0; }

template<typename T>
inline size_t restore(std::istream&, T&, bool = false, bool = false) { return 0; }

template<typename T>
inline size_t size_of(const T&) { return sizeof(T); }

template<typename T>
inline size_t size_of() { return sizeof(T); }

template<typename T>
struct binary {
    enum { is_streamable = 0 };
    static size_t size_of(void) { return sizeof(T); }
    static size_t size_of(const T&) { return sizeof(T); }
    static size_t store(std::ostream&, const T&, bool = false) { return 0; }
    static size_t restore(std::istream&, T&, bool = false) { return 0; }
    static std::string type_identifier() { return "unknown"; }
};

}} // namespace OpenMesh::IO
#endif
