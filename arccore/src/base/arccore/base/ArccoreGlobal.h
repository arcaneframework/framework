// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArccoreGlobal.h                                             (C) 2000-2025 */
/*                                                                           */
/* General declarations for Arccore.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARCCOREGLOBAL_H
#define ARCCORE_BASE_ARCCOREGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <cstdint>

#include "arccore/arccore_config.h"

#ifdef ARCCORE_VALID_TARGET
#  undef ARCCORE_VALID_TARGET
#endif

// Determine the OS type.
#if defined(__linux)
#  define ARCCORE_OS_LINUX
#elif defined(__APPLE__) && defined(__MACH__)
#  define ARCCORE_OS_MACOS
#elif defined(_AIX)
#  define ARCCORE_OS_AIX
#elif defined(__WIN32__) || defined(__NT__) || defined(WIN32) || defined(_WIN32) || defined(WIN32) || defined(_WINDOWS)
#  define ARCCORE_OS_WIN32
#elif defined(__CYGWIN__)
#  define ARCCORE_OS_CYGWIN
#endif

#ifdef ARCCORE_OS_WIN32
#  define ARCCORE_VALID_TARGET
#  define ARCCORE_EXPORT     __declspec(dllexport)
#  define ARCCORE_IMPORT     __declspec(dllimport)

/* Suppresses certain Microsoft compiler warnings */
#  ifdef _MSC_VER
#    pragma warning(disable: 4251) // class 'A' needs to have dll interface for to be used by clients of class 'B'.
#    pragma warning(disable: 4275) // non - DLL-interface classkey 'identifier' used as base for DLL-interface classkey 'identifier'
#    pragma warning(disable: 4800) // 'type' : forcing value to bool 'true' or 'false' (performance warning)
#    pragma warning(disable: 4355) // 'this' : used in base member initializer list
#  endif

#endif

// On Unix, this indicates that symbols in each .so are hidden by default.
// You must then explicitly mark the
// symbols you wish to export, like on Windows.
// The only difference is that for gcc with explicit template instantiations
// you must specify the export during explicit instantiation
// whereas on Windows it is done in the class.
#ifndef ARCCORE_OS_WIN32
#  define ARCCORE_EXPORT __attribute__ ((visibility("default")))
#  define ARCCORE_IMPORT __attribute__ ((visibility("default")))
#  define ARCCORE_TEMPLATE_EXPORT ARCCORE_EXPORT
#endif

#ifdef ARCCORE_OS_CYGWIN
#  define ARCCORE_VALID_TARGET
#endif

#ifdef ARCCORE_OS_LINUX
#  define ARCCORE_VALID_TARGET
#endif

#ifdef ARCCORE_OS_MACOS
#  define ARCCORE_VALID_TARGET
#endif

#ifndef ARCCORE_VALID_TARGET
#error "This target is not supported"
#endif

#ifndef ARCCORE_EXPORT
#define ARCCORE_EXPORT
#endif

#ifndef ARCCORE_IMPORT
#define ARCCORE_IMPORT
#endif

#ifndef ARCCORE_TEMPLATE_EXPORT
#define ARCCORE_TEMPLATE_EXPORT
#endif

#ifndef ARCCORE_RESTRICT
#define ARCCORE_RESTRICT
#endif

#define ARCCORE_STD std

// Tag var as a voluntary unused variable.
// Works with any compiler but might be improved by using attribute.
#define ARCCORE_UNUSED(var) do { (void)(var) ; } while(false)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef DOXYGEN_DOC
typedef ARCCORE_TYPE_INT16 Int16;
typedef ARCCORE_TYPE_INT32 Int32;
typedef ARCCORE_TYPE_INT64 Int64;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Macros for heterogeneous programming support (CPU/GPU)
 - ARCCORE_DEVICE_CODE: indicates a part of code compiled only on the device
 - ARCCORE_HOST_DEVICE: indicates that the method/variable is accessible both
   on the device and the host
 - ARCCORE_DEVICE: indicates that the method/variable is accessible only on
   the device.
*/

#if defined(__SYCL_DEVICE_ONLY__)
#  define ARCCORE_DEVICE_CODE
#  define ARCCORE_DEVICE_TARGET_SYCL
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#  define ARCCORE_DEVICE_CODE
#  if defined(__HIP_DEVICE_COMPILE__)
#    define ARCCORE_DEVICE_TARGET_HIP
#  endif
#  if defined(__CUDA_ARCH__)
#    define ARCCORE_DEVICE_TARGET_CUDA
// Necessary for assert(), for example in arccoreCheckAt()
// TODO: check if this is also necessary for AMD HIP.
#include <cassert>
#  endif
#endif

#if defined(__CUDACC__) || defined(__HIP__)
#define ARCCORE_HOST_DEVICE __host__ __device__
#define ARCCORE_DEVICE __device__
#endif


#ifndef ARCCORE_HOST_DEVICE
#define ARCCORE_HOST_DEVICE
#endif

#ifndef ARCCORE_DEVICE
#define ARCCORE_DEVICE
#endif

#if defined(ARCCORE_HAS_CUDA) && defined(__CUDACC__)

/*!
 * \brief Macro to indicate that %Arcane is compiled with support
 * for CUDA and that the CUDA compiler is used.
 */
#define ARCCORE_COMPILING_CUDA
//! \deprecated
#define ARCANE_COMPILING_CUDA
#endif
#if defined(ARCCORE_HAS_HIP) && defined(__HIP__)

/*!
 * \brief Macro to indicate that %Arcane is compiled with support
 * for HIP and that the HIP compiler is used.
 */
#define ARCCORE_COMPILING_HIP
//! \deprecated
#define ARCANE_COMPILING_HIP
#endif

#if defined(ARCCORE_HAS_SYCL)
#  if defined(SYCL_LANGUAGE_VERSION) || defined(__ADAPTIVECPP__)

/*!
 * \brief Macro to indicate that %Arcane is compiled with support
 * for SYCL and that the SYCL compiler is used.
 */
#    define ARCCORE_COMPILING_SYCL
//! \deprecated
#    define ARCANE_COMPILING_SYCL
#  endif
#endif

#if defined(ARCCORE_COMPILING_CUDA) || defined(ARCCORE_COMPILING_HIP)

/*!
 * \brief Macro to indicate that compilation is done with support
 * for CUDA or HIP.
 */
#define ARCCORE_COMPILING_CUDA_OR_HIP
//! \deprecated
#define ARCANE_COMPILING_CUDA_OR_HIP
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPONENT_arccore_base)
#define ARCCORE_BASE_EXPORT ARCCORE_EXPORT
#define ARCCORE_BASE_EXTERN_TPL
#else
#define ARCCORE_BASE_EXPORT ARCCORE_IMPORT
#define ARCCORE_BASE_EXTERN_TPL extern
#endif

#ifdef ARCCORE_REAL_USE_APFLOAT
#  include <apfloat.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Definition of Arccore types Int16, Int32, and Int64.
 */
//! Signed integer type of 8 bits
using Int8 = std::int8_t;
//! Signed integer type of 16 bits
using Int16 = std::int16_t;
//! Signed integer type of 32 bits
using Int32 = std::int32_t;
//! Signed integer type of 64 bits
using Int64 = std::int64_t;
//! Unsigned integer type of 32 bits
using UInt32 = std::uint32_t;
//! Unsigned integer type of 64 bits
using UInt64 = std::uint64_t;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Type representing a pointer.
 *
 * It must be used everywhere an object of any pointer type is expected.
 */
using Pointer = void*;

#ifdef ARCCORE_REAL_USE_APFLOAT
#  define ARCCORE_REAL(val) (Real(#val,1000))
#  define ARCCORE_REAL_NOT_BUILTIN
using Real = apfloat;
using APReal = apfloat;
#else
#  ifdef ARCCORE_REAL_LONG
#    define ARCCORE_REAL(val) val##L

/*!
 * \brief Type representing a real number.
 *
 * It must be used everywhere a real number object is expected.
 */
using long double Real;
#  else
#    define ARCCORE_REAL(val) val
#    define ARCCORE_REAL_IS_DOUBLE

/*!
 * \brief Type representing a real number.
 *
 * It must be used everywhere a real number object is expected.
 */
using Real = double;
#  endif
//! Emulation of real number in arbitrary precision.
class APReal
{
 public:
  Real v[4];
};
#endif

#ifdef ARCCORE_64BIT
#  define ARCCORE_INTEGER_MAX ARCCORE_INT64_MAX
using Short = Int32;
using Integer = Int64;
#else
#  define ARCCORE_INTEGER_MAX ARCCORE_INT32_MAX
using Short = Int32;
using Integer = Int32;
#endif

/*!
 * \def ARCCORE_INTEGER_MAX
 * \brief Macro indicating the maximum value that the #Integer type can take
 */


/*!
 * \typedef Int64
 * \brief Signed integer type of 64 bits.
 */
/*!
 * \typedef Int32
 * \brief Signed integer type of 32 bits.
 */
/*!
 * \typedef Int16
 * \brief Signed integer type of 16 bits.
 */
/*!
 * \typedef Integer
 * \brief Type representing an integer
 *
 * If the ARCCORE_64BIT macro is defined, the Integer type corresponds to an
 * Int64 integer, otherwise to an Int32 integer.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Brain Float16
class BFloat16;

//! Float 16 bit
class Float16;

//! IEEE-753 single-precision floating-point type
using Float32 = float;

//! Float 128 bit
class Float128;

//! Integer 128 bit
class Int128;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Structure equivalent to the boolean value \a true
 */
struct TrueType  {};

/*!
  \internal
  \brief Structure equivalent to the boolean value \a true
*/
struct FalseType {};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef __GNUG__
#  define ARCCORE_DEPRECATED __attribute__ ((deprecated))
#endif

#ifdef _MSC_VER
#  if _MSC_VER >= 1300
#    define ARCCORE_DEPRECATED __declspec(deprecated)
#  endif
#endif

#define ARCCORE_DEPRECATED_2017 ARCCORE_DEPRECATED
#define ARCCORE_DEPRECATED_2018 ARCCORE_DEPRECATED
#define ARCCORE_DEPRECATED_2019(reason) [[deprecated(reason)]]
#define ARCCORE_DEPRECATED_2020(reason) [[deprecated(reason)]]
#define ARCCORE_DEPRECATED_REASON(reason) [[deprecated(reason)]]

// Define this macro if you wish to suppress obsolete
// methods and types from compilation.
#define ARCCORE_NO_DEPRECATED

#ifndef ARCCORE_DEPRECATED
#  define ARCCORE_DEPRECATED
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Compatibility macros with different C++ standards.
// Now (in 2021) all compilers with which Arccore compiles
// support C++17, so most of these macros are no longer useful.
// We keep them only for compatibility with existing code.

// The ARCCORE_NORETURN macro uses the C++11 [[noreturn]] attribute to
// indicate that a function does not return.
#define ARCCORE_NORETURN [[noreturn]]

//! Macro allowing specification of the C++11 'constexpr' keyword
#define ARCCORE_CONSTEXPR constexpr

// Macro to indicate that no exceptions are thrown.
#define ARCCORE_NOEXCEPT noexcept

// Macros to indicate that no exceptions are thrown.
#define ARCCORE_NOEXCEPT_FALSE noexcept(false)

// Support for operator[](a,b,...)
#ifdef __cpp_multidimensional_subscript
#define ARCCORE_HAS_MULTI_SUBSCRIPT
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Macros for the [[no_unique_address]] attribute
// With VS2022, this attribute is not taken into account and you must
// use [[msvc::no_unique_address]]
#ifdef _MSC_VER
#define ARCCORE_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define ARCCORE_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Support for alignment.
// C++11 uses the alignas keyword to specify alignment.
// This works with GCC 4.9+ and Visual Studio 2015. It does not work
// with Visual Studio 2013. So for Visual Studio we use __declspec which always works.
// Under Linux, __attribute__ also always works, so we use that. Note that Simd structures need
// the 'packed' attribute, which only exists with GCC and Intel. There seems to be no
// equivalent with MSVC.
#ifdef _MSC_VER
//! Macro to guarantee the packing and alignment of a class to \a value bytes
#  define ARCCORE_ALIGNAS(value) __declspec(align(value))
//! Macro to guarantee the alignment of a class to \a value bytes
#  define ARCCORE_ALIGNAS_PACKED(value) __declspec(align(value))
#else
//! Macro to guarantee the packing and alignment of a class to \a value bytes
#  define ARCCORE_ALIGNAS_PACKED(value) __attribute__ ((aligned (value),packed))
//! Macro to guarantee the alignment of a class to \a value bytes
#  define ARCCORE_ALIGNAS(value) __attribute__ ((aligned (value)))
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_CHECK) || defined(ARCCORE_DEBUG)
#  ifndef ARCCORE_DEBUG_ASSERT
#    define ARCCORE_DEBUG_ASSERT
#  endif
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief True if in check mode.
 *
 * This mode is active if the ARCCORE_CHECK macro is defined
 * or if the arccoreSetCheck() method has been set to true.
 */
extern "C++" ARCCORE_BASE_EXPORT 
bool arccoreIsCheck();

/*!
 * \brief Activates or deactivates check mode.
 *
 * Check mode is always active if the ARCCORE_CHECK macro is defined.
 * Otherwise, it is possible to activate it using this method. This allows
 * certain tests to be enabled even in optimized mode.
 */
extern "C++" ARCCORE_BASE_EXPORT 
void arccoreSetCheck(bool v);

/*!
 * \brief True if the ARCCORE_DEBUG macro is defined
 */
extern "C++" ARCCORE_BASE_EXPORT
bool arccoreIsDebug();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Encapsulation of the C printf function
extern "C++" ARCCORE_BASE_EXPORT void
arccorePrintf(const char*,...);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enters pause mode or throws a fatal error.
 *
 * If arccoreSetPauseOnError() is called with the argument \a true,
 * it pauses the program
 * to potentially attach a debugger.
 *
 * Otherwise, it throws a FatalErrorException with the message
 * \a msg as an argument.
 */
extern "C++" ARCCORE_BASE_EXPORT void
arccoreDebugPause(const char* msg);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Indicates whether calling arccoreDebugPause() results in a pause.
 *
 * \sa arccoreDebugPause()
 */
extern "C++" ARCCORE_BASE_EXPORT void
arccoreSetPauseOnError(bool v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Macro to throw an exception with formatting.
 *
 * \a exception_class is the type of the exception. The following arguments of
 * the macro are used to format an error message via the
 * String::format() method.
 */
#define ARCCORE_THROW(exception_class,...) \
  throw exception_class (A_FUNCINFO,Arccore::String::format(__VA_ARGS__))

/*!
 * \brief Macro to throw an exception with formatting if \a cond is true.
 *
 * \a exception_class is the type of the exception. The following arguments of
 * the macro are used to format an error message via the
 * String::format() method.
 *
 * \sa ARCCORE_THROW
 */
#define ARCCORE_THROW_IF(cond, exception_class, ...) \
  if (cond) [[unlikely]] \
    ARCCORE_THROW(exception_class,__VA_ARGS__)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Macro throwing a FatalErrorException.
 *
 * The macro arguments are used to format an error message
 * via the String::format() method.
 */
#define ARCCORE_FATAL(...)\
  ARCCORE_THROW(::Arccore::FatalErrorException,__VA_ARGS__)

/*!
 * \brief Macro throwing a FatalErrorException if \a cond is true
 *
 * The macro arguments are used to format an error message
 * via the String::format() method.
 *
 * \sa ARCCORE_FATAL
 */
#define ARCCORE_FATAL_IF(cond, ...) \
  ARCCORE_THROW_IF(cond, ::Arccore::FatalErrorException,__VA_ARGS__)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Signals the use of a null pointer.
 *
 * Signals an attempt to use a null pointer.
 * Displays a message, calls arccoreDebugPause(), and throws a FatalErrorException.
 */
extern "C++" ARCCORE_BASE_EXPORT void
arccoreNullPointerError ARCCORE_NORETURN ();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Signals that a value is not within the desired range.
 *
 * Indicates that the assertion `min_value_inclusive <= i < max_value_exclusive`
 * is false.
 * Calls arccoreDebugPause() then throws an IndexOutOfRangeException.
 *
 * \param i invalid value.
 * \param min_value_inclusive allowed minimum inclusive value.
 * \param max_value_exclusive allowed maximum exclusive value.
 */
extern "C++" ARCCORE_BASE_EXPORT void
arccoreRangeError ARCCORE_NORETURN (Int64 i,Int64 min_value_inclusive,
                                    Int64 max_value_exclusive);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Signals that a value is not within the desired range.
 *
 * Indicates that the assertion `0 <= i < max_value` is false.
 * Throws an IndexOutOfRangeException.
 *
 * \param i invalid index
 * \param max_size number of elements in the array
 */
extern "C++" ARCCORE_BASE_EXPORT void
arccoreRangeError ARCCORE_NORETURN (Int64 i,Int64 max_size);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Checks that `min_value_inclusive <= i < max_value_exclusive`.
 *
 * If this is not the case, calls arccoreRangeError() to throw an
 * exception.
 */
inline ARCCORE_HOST_DEVICE void
arccoreCheckRange(Int64 i,Int64 min_value_inclusive,Int64 max_value_exclusive)
{
  if (i>=min_value_inclusive && i<max_value_exclusive)
    return;
#ifndef ARCCORE_DEVICE_CODE
  arccoreRangeError(i,min_value_inclusive,max_value_exclusive);
#elif defined(ARCCORE_DEVICE_TARGET_CUDA)
  // Code for the device.
  // assert() is available for CUDA.
  // TODO: check if a similar function exists for HIP
  assert(false);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Checks for potential array overflow.
 *
 * Calls arccoreCheckRange(i,0,max_size).
 */
inline ARCCORE_HOST_DEVICE void
arccoreCheckAt(Int64 i,Int64 max_size)
{
  arccoreCheckRange(i,0,max_size);
}

#if defined(ARCCORE_CHECK) || defined(ARCCORE_DEBUG)
#define ARCCORE_CHECK_AT(a,b) ::Arccore::arccoreCheckAt((a),(b))
#define ARCCORE_CHECK_RANGE(a,b,c) ::Arccore::arccoreCheckRange((a),(b),(c))
#else
#define ARCCORE_CHECK_AT(a,b)
#define ARCCORE_CHECK_RANGE(a,b,c)
#endif

#define ARCCORE_CHECK_AT2(a0,a1,b0,b1) \
  ARCCORE_CHECK_AT(a0,b0); ARCCORE_CHECK_AT(a1,b1)
#define ARCCORE_CHECK_AT3(a0,a1,a2,b0,b1,b2) \
  ARCCORE_CHECK_AT(a0,b0); ARCCORE_CHECK_AT(a1,b1); ARCCORE_CHECK_AT(a2,b2)
#define ARCCORE_CHECK_AT4(a0,a1,a2,a3,b0,b1,b2,b3) \
  ARCCORE_CHECK_AT(a0,b0); ARCCORE_CHECK_AT(a1,b1); ARCCORE_CHECK_AT(a2,b2); ARCCORE_CHECK_AT(a3,b3)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCCORE_CAST_SMALL_SIZE(a) ((Integer)(a))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT void
_doAssert(const char*,const char*,const char*,int);
template<typename T> inline T*
_checkPointer(const T* t,const char* file,const char* func,int line)
{
  if (!t){
    _doAssert("ARCCORE_ASSERT",file,func,line);
    arccorePrintf("Bad Pointer");
  }
  return t;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Macro to obtain the current function name via the preprocessor

#if defined(__GNUG__)
#  define ARCCORE_MACRO_FUNCTION_NAME __PRETTY_FUNCTION__
#elif defined( _MSC_VER)
#  define ARCCORE_MACRO_FUNCTION_NAME __FUNCTION__
#else
#  define ARCCORE_MACRO_FUNCTION_NAME __func__
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Macros used for debugging.
 */
#ifdef ARCCORE_DEBUG_ASSERT
#  define ARCCORE_D_WHERE(a) ::Arcane::_doAssert(a, __FILE__, ARCCORE_MACRO_FUNCTION_NAME, __LINE__)
#  define ARCCORE_DCHECK_POINTER(a) ::Arcane::_checkPointer((a), __FILE__, ARCCORE_MACRO_FUNCTION_NAME, __LINE__);
#  define ARCCORE_CHECK_PTR(a) \
  { \
    if (!(a)) { \
      ::Arcane::arccorePrintf("Null value"); \
      ARCCORE_D_WHERE("ARCCORE_ASSERT"); \
    } \
  }

#  define ARCCORE_ASSERT(a,b) \
  { \
    if (!(a)) { \
      ::Arcane::arccorePrintf("Assertion '%s' fails:", #a); \
      ::Arcane::arccorePrintf b; \
      ARCCORE_D_WHERE("ARCCORE_ASSERT"); \
    } \
  }
#  define ARCCORE_WARNING(a) \
  { \
    ::Arcane::arccorePrintf a; \
    ARCCORE_D_WHERE("ARCCORE_WARNING"); \
  }
#else
#  define ARCCORE_CHECK_PTR(a)
#  define ARCCORE_ASSERT(a,b)
#  define ARCCORE_WARNING(a)
#  define ARCCORE_DCHECK_POINTER(a) (a);
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Signals the use of a null pointer by throwing an exception.
 *
 * Signals an attempt to use a null pointer.
 * Throws a FatalErrorException.
 *
 * In the exception, displays \a text if not null, otherwise displays \a ptr_name.
 *
 * Normally this method should not be called directly but
 * via the ARCCORE_CHECK_POINTER macro.
 */
extern "C++" ARCCORE_BASE_EXPORT void
arccoreThrowNullPointerError [[noreturn]] (const char* ptr_name,const char* text);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Checks that a pointer is not null.
 *
 * If the pointer is null, calls arccoreThrowNullPointerError().
 * Otherwise, returns the pointer.
 */
inline void*
arccoreThrowIfNull(void* ptr,const char* ptr_name,const char* text)
{
  if (!ptr)
    arccoreThrowNullPointerError(ptr_name,text);
  return ptr;
}

/*!
 * \brief Checks that a pointer is not null.
 *
 * If the pointer is null, calls arccoreThrowNullPointerError().
 * Otherwise, returns the pointer.
 */
inline const void*
arccoreThrowIfNull(const void* ptr,const char* ptr_name,const char* text)
{
  if (!ptr)
    arccoreThrowNullPointerError(ptr_name,text);
  return ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Macro that returns the pointer \a ptr if it is not null
 * or throws an exception if it is null.
 *
 * \sa arccoreThrowIfNull().
 */
#define ARCCORE_CHECK_POINTER(ptr) \
  arccoreThrowIfNull(ptr,#ptr,nullptr)

/*!
 * \brief Macro that returns the pointer \a ptr if it is not null
 * or throws an exception if it is null.
 *
 * \sa arccoreThrowIfNull().
 */
#define ARCCORE_CHECK_POINTER2(ptr,text)\
  arccoreThrowIfNull(ptr,#ptr,text)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// The following macros allow creating an identifier by suffixing
// the file line number. This allows for a unique identifier
// for a file and is used, for example, to generate names
// of global variables for service registration.
// The macro to use is ARCANE_JOIN_WITH_LINE(name).
#define ARCCORE_JOIN_HELPER2(a,b) a ## b
#define ARCCORE_JOIN_HELPER(a,b) ARCCORE_JOIN_HELPER2(a,b)
#define ARCCORE_JOIN_WITH_LINE(a) ARCCORE_JOIN_HELPER(a,__LINE__)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Definitions of base types.
class String;
class StringView;
class StringFormatterArg;
class StringBuilder;
// Not in this component but included here for compatibility with existing
// code
class ITraceMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
using Arcane::ITraceMng;
using Arcane::String;
using Arcane::StringBuilder;
using Arcane::StringFormatterArg;
using Arcane::StringView;
using Arcane::UInt32;
using Arcane::UInt64;

using Arcane::APReal;
using Arcane::Integer;
using Arcane::Pointer;
using Arcane::Real;
using Arcane::Short;

//! Type 'Brain Float16'
using BFloat16 = Arcane::BFloat16;

//! Type 'Float16' (binary16)
using Float16 = Arcane::Float16;

//! IEEE-753 single precision floating point type (binary32)
using Float32 = float;

//! Type representing an 8-bit integer
using Int8 = Arcane::Int8;

//! Type representing a 128-bit float
using Float128 = Arcane::Float128;

//! Type representing a 128-bit integer
using Int128 = Arcane::Int128;
using Int16 = Arcane::Int16;
using Int32 = Arcane::Int32;
using Int64 = Arcane::Int64;

using Arcane::arccoreCheckAt;
using Arcane::arccoreCheckRange;
using Arcane::arccoreDebugPause;
using Arcane::arccoreIsCheck;
using Arcane::arccoreIsDebug;
using Arcane::arccoreNullPointerError;
using Arcane::arccorePrintf;
using Arcane::arccoreRangeError;
using Arcane::arccoreSetCheck;
using Arcane::arccoreSetPauseOnError;
using Arcane::arccoreThrowIfNull;
using Arcane::arccoreThrowNullPointerError;

using Arcane::FalseType;
using Arcane::TrueType;
using Arcane::_doAssert;
using Arcane::_checkPointer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
