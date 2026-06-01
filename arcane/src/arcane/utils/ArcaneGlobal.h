// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneGlobal.h                                              (C) 2000-2025 */
/*                                                                           */
/* General declarations for Arcane.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ARCANEGLOBAL_H
#define ARCANE_UTILS_ARCANEGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

// Global information about compilation options such as
// threads, debug mode, etc.
#include "arcane_core_config.h"

#ifdef ARCCORE_OS_LINUX
#  define ARCANE_OS_LINUX
#  include <cstddef>
#endif

#ifdef ARCCORE_OS_WIN32
#  define ARCANE_OS_WIN32
#endif

#ifdef ARCCORE_OS_MACOS
#  define ARCANE_OS_MACOS
#endif

#define ARCANE_EXPORT ARCCORE_EXPORT
#define ARCANE_IMPORT ARCCORE_IMPORT
#define ARCANE_TEMPLATE_EXPORT ARCCORE_TEMPLATE_EXPORT
#define ARCANE_RESTRICT ARCCORE_RESTRICT

#define ARCANE_STD std

// Tag var as a voluntary unused variable.
// Works with any compiler but might be improved by using attribute.
#define ARCANE_UNUSED(var) ARCCORE_UNUSED(var)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: remove the inclusion of <iosfwd> and the using statements.
// For now (2022), these inclusions are removed only for Arcane.

#ifndef ARCANE_NO_USING_FOR_STREAM
#include <iosfwd>
using std::istream;
using std::ostream;
using std::ios;
using std::ifstream;
using std::ofstream;
using std::ostringstream;
using std::istringstream;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef DOXYGEN_DOC
typedef ARCANE_TYPE_INT16 Int16;
typedef ARCANE_TYPE_INT32 Int32;
typedef ARCANE_TYPE_INT64 Int64;
#endif

#define ARCANE_BEGIN_NAMESPACE  namespace Arcane {
#define ARCANE_END_NAMESPACE    }
#define NUMERICS_BEGIN_NAMESPACE  namespace Numerics {
#define NUMERICS_END_NAMESPACE    }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_FULL
#define ARCANE_COMPONENT_arcane_utils
#define ARCANE_COMPONENT_arcane
#define ARCANE_COMPONENT_arcane_mesh
#define ARCANE_COMPONENT_arcane_std
#define ARCANE_COMPONENT_arcane_impl
#define ARCANE_COMPONENT_arcane_script
#endif

#if defined(ARCANE_COMPONENT_arcane) || defined(ARCANE_COMPONENT_arcane_core)
#define ARCANE_CORE_EXPORT ARCANE_EXPORT
#define ARCANE_EXPR_EXPORT ARCANE_EXPORT
#define ARCANE_DATATYPE_EXPORT ARCANE_EXPORT
#define ARCANE_CORE_EXTERN_TPL
#else
#define ARCANE_CORE_EXPORT ARCANE_IMPORT
#define ARCANE_EXPR_EXPORT ARCANE_IMPORT
#define ARCANE_DATATYPE_EXPORT ARCANE_IMPORT
#define ARCANE_CORE_EXTERN_TPL extern
#endif

#ifdef ARCANE_COMPONENT_arcane_utils
#define ARCANE_UTILS_EXPORT ARCANE_EXPORT
#define ARCANE_UTILS_EXTERN_TPL
#else
#define ARCANE_UTILS_EXPORT ARCANE_IMPORT
#define ARCANE_UTILS_EXTERN_TPL extern
#endif

#ifdef ARCANE_COMPONENT_arcane_impl
#define ARCANE_IMPL_EXPORT ARCANE_EXPORT
#else
#define ARCANE_IMPL_EXPORT ARCANE_IMPORT
#endif

#ifdef ARCANE_COMPONENT_arcane_mesh
#define ARCANE_MESH_EXPORT ARCANE_EXPORT
#else
#define ARCANE_MESH_EXPORT ARCANE_IMPORT
#endif

#ifdef ARCANE_COMPONENT_arcane_std
#define ARCANE_STD_EXPORT ARCANE_EXPORT
#else
#define ARCANE_STD_EXPORT ARCANE_IMPORT
#endif

#ifdef ARCANE_COMPONENT_arcane_script
#define ARCANE_SCRIPT_EXPORT ARCANE_EXPORT
#else
#define ARCANE_SCRIPT_EXPORT ARCANE_IMPORT
#endif

#ifdef ARCANE_COMPONENT_arcane_solvers
#define ARCANE_SOLVERS_EXPORT ARCANE_EXPORT
#else
#define ARCANE_SOLVERS_EXPORT ARCANE_IMPORT
#endif

#ifdef ARCANE_COMPONENT_arcane_geometry
#define ARCANE_GEOMETRY_EXPORT ARCANE_EXPORT
#else
#define ARCANE_GEOMETRY_EXPORT ARCANE_IMPORT
#endif

#ifdef ARCANE_COMPONENT_arcane_thread
#define ARCANE_THREAD_EXPORT ARCANE_EXPORT
#else
#define ARCANE_THREAD_EXPORT ARCANE_IMPORT
#endif

#ifdef ARCANE_COMPONENT_arcane_mpi
#define ARCANE_MPI_EXPORT ARCANE_EXPORT
#else
#define ARCANE_MPI_EXPORT ARCANE_IMPORT
#endif

#ifdef ARCANE_COMPONENT_arcane_hyoda
#define ARCANE_HYODA_EXPORT ARCANE_EXPORT
#else
#define ARCANE_HYODA_EXPORT ARCANE_IMPORT
#endif

#ifdef ARCANE_REAL_USE_APFLOAT
#include <apfloat.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_HAS_LONG_LONG

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const double cgrEPSILON_DELTA = 1.0e-2;
const double cgrPI = 3.14159265358979323846;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_REAL(val) ARCCORE_REAL(val)

#ifdef ARCCORE_REAL_NOT_BUILTIN
#  define ARCANE_REAL_NOT_BUILTIN
#endif

#ifdef ARCCORE_REAL_LONG
#  define ARCANE_REAL_LONG
#endif

#ifdef ARCCORE_REAL_IS_DOUBLE
#  define ARCANE_REAL_IS_DOUBLE
#endif

/*!
 * \brief Type of integers used to store local identifiers
 * of entities.
 *
 * The values this type can take indicate how many entities
 * can be present on a subdomain.
 */
using LocalIdType = Int32;

/*!
 * \brief Type of integers used to store unique
 * (global) identifiers of entities.
 *
 * The values this type can take indicate how many entities
 * can be present on the initial domain.
 */
using UniqueIdType = Int64;

/*!
 * \def ARCANE_INTEGER_MAX
 * \brief Macro indicating the maximum value that the #Integer type can take
 */

/*!
 * \typedef Int64
 * \brief 64-bit signed integer type.
 */
/*!
 * \typedef Int32
 * \brief 32-bit signed integer type.
 */
/*!
 * \typedef Int16
 * \brief 16-bit signed integer type.
 */
/*!
 * \typedef Integer
 * \brief Type representing an integer
 *
 * If the ARCANE_64BIT macro is defined, the Integer type corresponds to an
 * Int64 integer, otherwise to an Int32 integer.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Encapsulation of the C printf function
extern "C++" ARCANE_UTILS_EXPORT void
arcanePrintf(const char*,...);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enters pause mode or throws a fatal error.
 *
 * If the code is compiled in \a debug mode (ARCANE_DEBUG is defined) or
 * in \a check mode (ARCANE_CHECK is defined), it pauses the program
 * potentially allowing a debugger to connect to it.
 *
 * In normal mode, it throws a FatalErrorException with the message
 * \a msg as an argument.
 */
extern "C++" ARCANE_UTILS_EXPORT void
arcaneDebugPause(const char* msg);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
_internalArcaneMathError(long double arg_value,const char* func_name);

extern "C++" ARCANE_UTILS_EXPORT void
_internalArcaneMathError(long double arg_value1,long double arg_value2,const char* func_name);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Signals an invalid argument in a mathematical function.
 *
 * After displaying the message, it calls arcaneDebugPause()
 *
 * \param arg_value value of the invalid argument.
 * \param func_name name of the mathematical function.
 */
ARCCORE_HOST_DEVICE inline void
arcaneMathError(long double arg_value,const char* func_name)
{
#ifndef ARCCORE_DEVICE_CODE
  _internalArcaneMathError(arg_value,func_name);
#else
  ARCANE_UNUSED(arg_value);
  ARCANE_UNUSED(func_name);
#endif
}

/*!
 * \brief Signals an invalid argument in a mathematical function.
 *
 * After displaying the message, it calls arcaneDebugPause()
 *
 * \param arg_value1 value of the first invalid argument.
 * \param arg_value2 value of the second invalid argument.
 * \param func_name name of the mathematical function.
 */
ARCCORE_HOST_DEVICE inline void
arcaneMathError(long double arg_value1,long double arg_value2,const char* func_name)
{
#ifndef ARCCORE_DEVICE_CODE
  _internalArcaneMathError(arg_value1,arg_value2,func_name);
#else
  ARCANE_UNUSED(arg_value1);
  ARCANE_UNUSED(arg_value2);
  ARCANE_UNUSED(func_name);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Signals an unimplemented function.
 *
 * After displaying the message, it calls arcaneDebugPause()
 *
 * \param file name of the file containing the function
 * \param func name of the function
 * \param line number
 * \param msg optional message to display (0 if none)
 */
extern "C++" ARCANE_UTILS_EXPORT void
arcaneNotYetImplemented(const char* file,const char* func,unsigned long line,const char* msg);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Signals the use of a deprecated function
extern "C++" ARCANE_UTILS_EXPORT void
arcaneDeprecated(const char* file,const char* func,unsigned long line,const char* text);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Use of an unreferenced object.
 *
 * Signals an attempt to use an object that should no longer be
 * referenced. Displays a message and calls arcaneDebugPause() if requested,
 * and then throws a FatalErrorException.
 *
 * \param ptr address of the object
 */
extern "C++" ARCANE_UTILS_EXPORT void
arcaneNoReferenceError(const void* ptr);

/*!
 * \brief Use of an unreferenced object.
 *
 * Signals an attempt to use an object that should no longer be
 * referenced. Displays a message and calls arcaneDebugPause() if requested,
 * and then calls std::terminate().
 *
 * \param ptr address of the object
 */
extern "C++" ARCANE_UTILS_EXPORT void
arcaneNoReferenceErrorCallTerminate(const void* ptr);

/*!
 * \brief Checks that \a size can be converted into an 'Integer' to serve
 * as the size of an array. If possible, returns \a size converted to an 'Integer'.
 * Otherwise, throws an ArgumentException.
 */
extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCheckArraySize(unsigned long long size);

/*!
 * \brief Checks that \a size can be converted into an 'Integer' to serve
 * as the size of an array. If possible, returns \a size converted to an 'Integer'.
 * Otherwise, throws an ArgumentException.
 */
extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCheckArraySize(long long size);

/*!
 * \brief Checks that \a size can be converted into an 'Integer' to serve
 * as the size of an array. If possible, returns \a size converted to an 'Integer'.
 * Otherwise, throws an ArgumentException.
 */
extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCheckArraySize(unsigned long size);

/*!
 * \brief Checks that \a size can be converted into an 'Integer' to serve
 * as the size of an array. If possible, returns \a size converted to an 'Integer'.
 * Otherwise, throws an ArgumentException.
 */
extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCheckArraySize(long size);

/*!
 * \brief Checks that \a size can be converted into an 'Integer' to serve
 * as the size of an array. If possible, returns \a size converted to an 'Integer'.
 * Otherwise, throws an ArgumentException.
 */
extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCheckArraySize(unsigned int size);

/*!
 * \brief Checks that \a size can be converted into an 'Integer' to serve
 * as the size of an array. If possible, returns \a size converted to an 'Integer'.
 * Otherwise, throws an ArgumentException.
 */
extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCheckArraySize(int size);

/*!
 * \brief Checks that \a ptr is aligned on \a alignment bytes.
 * If not, throws a BadAlignmentException.
 */
extern "C++" ARCANE_UTILS_EXPORT void
arcaneCheckAlignment(const void* ptr,Integer alignment);

/*!
 * \brief True if running in check mode.
 *
 * This mode is active if the ARCANE_CHECK macro is defined
 * or if the arcaneSetCheck() method has been set to true.
 */
extern "C++" ARCANE_UTILS_EXPORT 
bool arcaneIsCheck();

/*!
 * \brief Activates or deactivates verification mode.
 *
 * Verification mode is always active if the ARCANE_CHECK macro is defined.
 * Otherwise, it is possible to activate it using this method. This allows
 * certain tests to be activated even in optimized mode.
 */
extern "C++" ARCANE_UTILS_EXPORT 
void arcaneSetCheck(bool v);

/*!
 * \brief True if the ARCANE_DEBUG macro is defined
 */
extern "C++" ARCANE_UTILS_EXPORT
bool arcaneIsDebug();

/*!
 * \brief True if arcane is compiled with thread support AND they are active
 */
extern "C++" ARCANE_UTILS_EXPORT 
bool arcaneHasThread();

/*!
 * \brief Activates or deactivates thread support.
 *
 * This function should only be called during application initialization
 * (or before) and must not be modified afterward.
 * Thread activation is only possible if a thread implementation
 * exists on the platform and Arcane was compiled with this support.
 */
extern "C++" ARCANE_UTILS_EXPORT 
void arcaneSetHasThread(bool v);

/*!
 * \brief Returns the ID of the current thread.
 *
 * Always returns 0 if arcaneHasThread() is false.
 */
extern "C++" ARCANE_UTILS_EXPORT
Int64 arcaneCurrentThread();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_DEBUG
extern "C++" ARCANE_UTILS_EXPORT bool _checkDebug(size_t);
#define ARCANE_DEBUGP(a,b)     if (_checkDebug(a)) { arcanePrintf b; }
#else
#define ARCANE_DEBUGP(a,b)
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef __GNUG__
#  define ARCANE_NOT_YET_IMPLEMENTED(a) \
{ arcaneNotYetImplemented(__FILE__,__PRETTY_FUNCTION__,__LINE__,(a)); }
#else
#  define ARCANE_NOT_YET_IMPLEMENTED(a) \
{ arcaneNotYetImplemented(__FILE__,"(NoInfo)",__LINE__,(a)); }
#endif

#define ARCANE_DEPRECATED ARCCORE_DEPRECATED

#define ARCANE_DEPRECATED_112 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_114 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_116 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_118 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_120 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_122 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_200 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_220 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_240 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_260 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_280 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_2018 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_2018_R(reason) [[deprecated(reason)]]

#ifndef ARCCORE_DEPRECATED_2021
#define ARCCORE_DEPRECATED_2021(reason) [[deprecated(reason)]]
#endif

#define ARCANE_DEPRECATED_REASON(reason) [[deprecated(reason)]]

#ifdef ARCANE_NO_DEPRECATED_LONG_TERM
#define ARCANE_DEPRECATED_LONG_TERM(reason)
#else
/*!
 * \brief Macro for long-term 'deprecated' attribute.
 *
 * This macro is used to indicate types or functions
 * that are obsolete and therefore preferably should not be used, but
 * will not be removed for several versions.
 */
#define ARCANE_DEPRECATED_LONG_TERM(reason) [[deprecated(reason)]]
#endif

// Define this macro if you wish to remove obsolete methods and types from compilation.
#define ARCANE_NO_DEPRECATED

// If the macro is defined, do not notify about deprecated methods of old
// array classes.
#ifdef ARCANE_NO_NOTIFY_DEPRECATED_ARRAY
#define ARCANE_DEPRECATED_ARRAY
#else
#define ARCANE_DEPRECATED_ARRAY ARCANE_DEPRECATED
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// The following macros allow creating an identifier by suffixing the file line number.
// This allows having a unique identifier for a file and is used, for example,
// to generate global variable names for service registration.
// The macro to use is ARCANE_JOIN_WITH_LINE(name).
#define ARCANE_JOIN_HELPER2(a,b) a ## b
#define ARCANE_JOIN_HELPER(a,b) ARCANE_JOIN_HELPER2(a,b)
#define ARCANE_JOIN_WITH_LINE(a) ARCANE_JOIN_HELPER(a,__LINE__)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// The ARCANE_NORETURN macro uses the C++11 [[noreturn]] attribute to indicate that a function does not return.
#define ARCANE_NORETURN ARCCORE_NORETURN

//! Macro allowing specification of the C++11 'constexpr' keyword
#define ARCANE_CONSTEXPR ARCCORE_CONSTEXPR

// C++11 defines a 'noexcept' keyword to indicate that a method does not throw exceptions.
// Unfortunately, since C++11 support is partial across compilers, this does not work
// for everyone. In particular, icc 13, 14, and 15 do not support this, nor do
// Visual Studio 2013 and earlier.
#define ARCANE_NOEXCEPT ARCCORE_NOEXCEPT
#define ARCANE_NOEXCEPT_FALSE ARCCORE_NOEXCEPT_FALSE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Support for alignment.
// C++11 uses the alignas keyword to specify alignment.
// This works with GCC 4.9+ and Visual Studio 2015. It does not work with Visual Studio 2013.
// Therefore, for Visual Studio, we always use __declspec, which always works.
// Under Linux, __attribute__ also always works, so we use that. Note that SIMD structures
// require the 'packed' attribute, which only exists with GCC and Intel. There seems
// to be no equivalent with MSVC.
#ifdef _MSC_VER
//! Macro to guarantee the packing and alignment of a class to \a value bytes
#  define ARCANE_ALIGNAS(value) __declspec(align(value))
//! Macro to guarantee the alignment of a class to \a value bytes
#  define ARCANE_ALIGNAS_PACKED(value) __declspec(align(value))
#else
//! Macro to guarantee the packing and alignment of a class to \a value bytes
#  define ARCANE_ALIGNAS_PACKED(value) __attribute__ ((aligned (value),packed))
//! Macro to guarantee the alignment of a class to \a value bytes
#  define ARCANE_ALIGNAS(value) __attribute__ ((aligned (value)))
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_SWIG
#ifdef ARCANE_DEPRECATED
#undef ARCANE_DEPRECATED
#endif
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_CHECK) || defined(ARCANE_DEBUG)
#ifndef ARCANE_DEBUG_ASSERT
#define ARCANE_DEBUG_ASSERT
#endif
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Signals the use of a null pointer.
 *
 * Signals an attempt to use a null pointer.
 * Displays a message, calls arcaneDebugPause(), and throws a FatalErrorException.
 */
extern "C++" ARCANE_UTILS_EXPORT void
arcaneNullPointerError [[noreturn]] ();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Signals the use of a null pointer by throwing an exception
 *
 * Signals an attempt to use a null pointer.
 * Throws a FatalErrorException.
 *
 * In the exception, displays \a text if not null, otherwise displays \a ptr_name.
 *
 * Normally, this method should not be called directly but via the ARCANE_CHECK_POINTER macro.
 */
extern "C++" ARCANE_UTILS_EXPORT void
arcaneThrowNullPointerError [[noreturn]] (const char* ptr_name,const char* text);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Checks that a pointer is not null.
 */
static inline void
arcaneCheckNull(const void* ptr)
{
  if (!ptr)
    arcaneNullPointerError();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Returns the size with padding for a size \a size.
 *
 * The returned value is a multiple of SIMD_PADDING_SIZE and is:
 * - 0 if \a size is less than or equal to 0.
 * - \a size if \a size is a multiple of SIMD_PADDING_SIZE.
 * - the multiple of SIMD_PADDING_SIZE immediately greater than \a size otherwise.
 */
extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneSizeWithPadding(Integer size);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Macros used for debugging.
 */
#ifdef ARCANE_DEBUG_ASSERT
extern "C++" ARCANE_UTILS_EXPORT void _doAssert(const char*,const char*,const char*,size_t);
template<typename T> inline T*
_checkPointer(T* t,const char* file,const char* func,size_t line)
{
  if (!t){
    _doAssert("ARCANE_ASSERT",file,func,line);
    arcanePrintf("Bad Pointer");
  }
  return t;
}
#  ifdef __GNUG__
#    define ARCANE_D_WHERE(a)  Arcane::_doAssert(a,__FILE__,__PRETTY_FUNCTION__,__LINE__)
#    define ARCANE_DCHECK_POINTER(a) Arcane::_checkPointer((a),__FILE__,__PRETTY_FUNCTION__,__LINE__);
#  else
#    define ARCANE_D_WHERE(a)  Arcane::_doAssert(a,__FILE__,"(NoInfo)",__LINE__)
#    define ARCANE_DCHECK_POINTER(a) Arcane::_checkPointer((a),__FILE__,"(NoInfo"),__LINE__);
#  endif
#  define ARCANE_CHECK_PTR(a) \
   {if (!(a)){Arcane::arcanePrintf("Null value");ARCANE_D_WHERE("ARCANE_ASSERT");}}

#  define ARCANE_ASSERT(a,b) \
  {if (!(a)){ Arcane::arcanePrintf("Assertion '%s' fails:",#a); Arcane::arcanePrintf b; ARCANE_D_WHERE("ARCANE_ASSERT");}}
#  define ARCANE_WARNING(a) \
   { Arcane::arcanePrintf a; ARCANE_D_WHERE("ARCANE_WARNING"); }
#else
#  define ARCANE_CHECK_PTR(a)
#  define ARCANE_ASSERT(a,b)
#  define ARCANE_WARNING(a)
#  define ARCANE_DCHECK_POINTER(a) (a);
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro for throwing an exception with formatting.
 *
 * \a exception_class is the exception type. The following arguments of
 * the macro are used to format an error message via the
 * String::format() method.
 */
#define ARCANE_THROW(exception_class,...) \
  ARCCORE_THROW(exception_class,__VA_ARGS__)

/*!
 * \brief Macro for throwing an exception with formatting if \a cond is true.
 *
 * \a exception_class is the exception type. The following arguments of
 * the macro are used to format an error message via the
 * String::format() method.
 *
 * \sa ARCANE_THROW
 */
#define ARCANE_THROW_IF(const, exception_class, ...)    \
  ARCCORE_THROW_IF(const, exception_class, __VA_ARGS__)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro throwing a FatalErrorException.
 *
 * The macro arguments are used to format an error message
 * via the String::format() method.
 */
#define ARCANE_FATAL(...) \
  ARCCORE_FATAL(__VA_ARGS__)

/*!
 * \brief Macro throwing a FatalErrorException if \a cond is true
 *
 * The macro arguments are used to format an error message
 * via the String::format() method.
 *
 * \sa ARCANE_FATAL
 */
#define ARCANE_FATAL_IF(const, ...) \
  ARCCORE_FATAL_IF(const, __VA_ARGS__)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Checks that a pointer is not null.
 *
 * If the pointer is null, it calls arcaneThrowNullPointerError().
 * Otherwise, it returns the pointer.
 */
static inline void*
arcaneThrowIfNull(void* ptr,const char* ptr_name,const char* text)
{
  if (!ptr)
    arcaneThrowNullPointerError(ptr_name,text);
  return ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Checks that a pointer is not null.
 *
 * If the pointer is null, it calls arcaneThrowNullPointerError().
 * Otherwise, it returns the pointer.
 */
static inline const void*
arcaneThrowIfNull(const void* ptr,const char* ptr_name,const char* text)
{
  if (!ptr)
    arcaneThrowNullPointerError(ptr_name,text);
  return ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Checks that a pointer is not null.
 *
 * If the pointer is null, it calls arcaneThrowNullPointerError().
 * Otherwise, it returns the pointer.
 */
template<typename T> inline T*
arcaneThrowIfNull(T* ptr,const char* ptr_name,const char* text)
{
  if (!ptr)
    arcaneThrowNullPointerError(ptr_name,text);
  return ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro returning the pointer \a ptr if it is not null
 * or throwing an exception if it is null.
 *
 * \sa arcaneThrowIfNull().
 */
#define ARCANE_CHECK_POINTER(ptr) \
  arcaneThrowIfNull(ptr,#ptr,nullptr)

/*!
 * \brief Macro returning the pointer \a ptr if it is not null
 * or throwing an exception if it is null.
 *
 * \sa arcaneThrowIfNull().
 */
#define ARCANE_CHECK_POINTER2(ptr,text)\
  arcaneThrowIfNull(ptr,#ptr,text)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Signals an overflow error.
 *
 * Signals an array overflow. Displays a message and calls arcaneDebugPause().
 *
 * \param i invalid index
 * \param max_size number of elements in the array
 */
extern "C++" ARCANE_UTILS_EXPORT void
arcaneRangeError [[noreturn]] (Int64 i,Int64 max_size);

/*!
 * \brief Checks for a possible array overflow.
 */
static inline constexpr ARCCORE_HOST_DEVICE void
arcaneCheckAt(Int64 i,Int64 max_size)
{
#ifndef ARCCORE_DEVICE_CODE
  if (i<0 || i>=max_size)
    arcaneRangeError(i,max_size);
#else
  ARCANE_UNUSED(i);
  ARCANE_UNUSED(max_size);
#endif
}

#if defined(ARCANE_CHECK) || defined(ARCANE_DEBUG)
#define ARCANE_CHECK_AT(a,b) ::Arcane::arcaneCheckAt((a),(b))
#else
#define ARCANE_CHECK_AT(a,b)
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
