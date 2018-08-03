# ----------------------------------------------------------------------------
# Regarde la taille des différents types (short,int,long,long long) du C++

include(CheckTypeSize)
check_type_size(short ARCCORE_SIZEOF_SHORT BUILTIN_TYPES_ONLY LANGUAGE CXX)
check_type_size(int ARCCORE_SIZEOF_INT BUILTIN_TYPES_ONLY LANGUAGE CXX)
check_type_size(long ARCCORE_SIZEOF_LONG BUILTIN_TYPES_ONLY LANGUAGE CXX)
check_type_size("long long" ARCCORE_SIZEOF_LONGLONG BUILTIN_TYPES_ONLY LANGUAGE CXX)
message(STATUS "Size of built-in types:"
  " short=${ARCCORE_SIZEOF_SHORT}"
  " int=${ARCCORE_SIZEOF_INT}"
  " long=${ARCCORE_SIZEOF_LONG}"
  " long long=${ARCCORE_SIZEOF_LONGLONG}")

# ----------------------------------------------------------------------------
# Définit les types 'Int16', 'Int32' et 'Int64' si ce n'est pas déjà fait.

if(NOT ARCCORE_TYPE_INT16)
  # Cherche un type C pour Int16
  # Toutes les implémentations actuelles ont un short sur 2 octets
  if (ARCCORE_SIZEOF_SHORT EQUAL 2)
    set(ARCCORE_TYPE_INT16 "short")
  else()
  endif()
endif()

if(NOT ARCCORE_TYPE_INT32)
  # Cherche un type C pour Int32
  # Pour les implémentations actuelles, c'est soit un 'int', soit un 'long'.
  # (a priori je crois que 'int' est 32 bits sur toutes les implémentations
  # actuelles)
  if (ARCCORE_SIZEOF_INT EQUAL 4)
    set(ARCCORE_TYPE_INT32 "int")
    set(ARCCORE_INT32_MAX "2147483647")
  elseif(ARCCORE_SIZEOF_LONG EQUAL 4)
    set(ARCCORE_TYPE_INT32 "long")
    set(ARCCORE_INT32_MAX "2147483647L")
  endif()
endif()

if(NOT ARCCORE_TYPE_INT64)
  # Cherche un type C pour Int64
  # Pour les implémentations actuelles, c'est soit un 'long', soit un 'long long'.
  # (a priori je crois que 'long' est 64 bits sur toutes les implémentations
  # actuelles Linux et 32 bits sur Win64)
  if (ARCCORE_SIZEOF_LONG EQUAL 8)
    set(ARCCORE_TYPE_INT64 "long")
    set(ARCCORE_INT64_MAX "9223372036854775807L")
  elseif(ARCCORE_SIZEOF_LONGLONG EQUAL 8)
    set(ARCCORE_TYPE_INT64 "long long")
    set(ARCCORE_INT64_MAX "9223372036854775807LL")
  endif()
endif()

message(STATUS "Arcane types Int16=${ARCCORE_TYPE_INT16} Int32=${ARCCORE_TYPE_INT32} Int64=${ARCCORE_TYPE_INT64}")
if (NOT ARCCORE_TYPE_INT16)
  message(FATAL_ERROR "Can not find a valid type for 'Int16'")
endif()
if (NOT ARCCORE_TYPE_INT32)
  message(FATAL_ERROR "Can not find a valid type for 'Int32'")
endif()
if (NOT ARCCORE_TYPE_INT64)
  message(FATAL_ERROR "Can not find a valid type for 'Int64'")
endif()

message(STATUS "Arcane types Int32_MAX=${ARCCORE_INT32_MAX} Int64_MAX=${ARCCORE_INT64_MAX}")

set(ARCCORE_TYPE_INT16 ${ARCCORE_TYPE_INT16} CACHE STRING "C++ built-in type for Int16" FORCE)
set(ARCCORE_TYPE_INT32 ${ARCCORE_TYPE_INT32} CACHE STRING "C++ built-in type for Int32" FORCE)
set(ARCCORE_TYPE_INT64 ${ARCCORE_TYPE_INT64} CACHE STRING "C++ built-in type for Int64" FORCE)

set(ARCCORE_INT32_MAX ${ARCCORE_INT32_MAX} CACHE STRING "Max value for Int32" FORCE)
set(ARCCORE_INT64_MAX ${ARCCORE_INT64_MAX} CACHE STRING "Max value for Int64" FORCE)
