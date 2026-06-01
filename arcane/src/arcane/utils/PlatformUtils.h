// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PlatformUtils.h                                             (C) 2000-2026 */
/*                                                                           */
/* Platform-dependent utility functions.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_PLATFORMUTILS_H
#define ARCANE_UTILS_PLATFORMUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/PlatformUtils.h"
#include "arcane/utils/UtilsTypes.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IOnlineDebuggerService;
class IProfilingService;
class IProcessorAffinityService;
class IDynamicLibraryLoader;
class ISymbolizerService;
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Namespace for platform-dependent functions.
 *
 *  This namespace contains all platform-dependent functions.
 */
namespace Arcane::platform
{

/*!
 * \brief Platform-specific initialization.
 *
 This routine is called when the architecture is initialized.
 It allows certain platform-dependent processes to be performed.
 */
extern "C++" ARCANE_UTILS_EXPORT void platformInitialize();

/*!
 * \brief Platform-specific program termination routines.
 *
 This routine is called just before exiting the program.
 */
extern "C++" ARCANE_UTILS_EXPORT void platformTerminate();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Service used for managing processor affinity.
 *
 * May return null if no service is available.
 */
extern "C++" ARCANE_UTILS_EXPORT IProcessorAffinityService*
getProcessorAffinityService();

/*!
 * \brief Sets the service used for managing processor affinity.
 *
 * Returns the previously used service.
 */
extern "C++" ARCCORE_DEPRECATED_REASON("Y2025: This method is internal to Arcane")
ARCANE_UTILS_EXPORT IProcessorAffinityService* setProcessorAffinityService(IProcessorAffinityService* service);

/*!
 * \brief Service used to obtain profiling information.
 *
 * May return null if no service is available.
 */
extern "C++" ARCANE_UTILS_EXPORT IProfilingService*
getProfilingService();

/*!
 * \brief Sets the service used to obtain profiling information.
 *
 * Returns the previously used service.
 */
extern "C++" ARCCORE_DEPRECATED_REASON("Y2025: This method is internal to Arcane")
ARCANE_UTILS_EXPORT IProfilingService* setProfilingService(IProfilingService* service);

/*!
 * \brief Service used to set up an online debug architecture.
 *
 * May return null if no service is available.
 */
extern "C++" ARCANE_UTILS_EXPORT IOnlineDebuggerService*
getOnlineDebuggerService();

/*!
 * \brief Sets the service to be used for the online debug architecture.
 *
 * Returns the previously used service.
 */
extern "C++" ARCCORE_DEPRECATED_REASON("Y2025: This method is internal to Arcane")
ARCANE_UTILS_EXPORT IOnlineDebuggerService* setOnlineDebuggerService(IOnlineDebuggerService* service);

/*!
 * \brief Service used to manage threads.
 *
 * May return null if no service is available.
 */
extern "C++" ARCANE_UTILS_EXPORT IThreadImplementation*
getThreadImplementationService();

/*!
 * \brief Sets the service used to manage threads.
 *
 * Returns the previously used service.
 */
extern "C++" ARCCORE_DEPRECATED_REASON("Y2025: This method is internal to Arcane")
ARCANE_UTILS_EXPORT IThreadImplementation* setThreadImplementationService(IThreadImplementation* service);

/*!
 * \brief Sets the service used to manage internal processor counters.
 *
 * Returns the previously used service.
 */
extern "C++" ARCCORE_DEPRECATED_REASON("Y2025: This method is internal to Arcane")
ARCANE_UTILS_EXPORT IPerformanceCounterService* setPerformanceCounterService(IPerformanceCounterService* service);

/*!
 * \brief Service used to obtain internal processor counters.
 *
 * May return null if no service is available.
 */
extern "C++" ARCANE_UTILS_EXPORT IPerformanceCounterService*
getPerformanceCounterService();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Resets the alarm timer to \a nb_second.
 *
 * The timer will trigger a signal (SIGALRM) after \a nb_second.
 */
extern "C++" ARCANE_UTILS_EXPORT void
resetAlarmTimer(Integer nb_second);

/*!
 * \brief True if the code is running with the .NET runtime.
 */
extern "C++" ARCANE_UTILS_EXPORT bool
hasDotNETRuntime();

/*!
 * \brief Sets whether the code is running with the .NET runtime.
 *
 * This function can only be set at the start of the calculation before
 * arcaneInitialize().
 */
extern "C++" ARCANE_UTILS_EXPORT void
setHasDotNETRuntime(bool v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Calls the .Net Garbage Collector if available
extern "C++" ARCANE_UTILS_EXPORT void
callDotNETGarbageCollector();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allocator specific to accelerators.
 *
 * \deprecated Use MemoryUtils::getDefaultDataAllocator() instead.
 */
extern "C++" ARCANE_DEPRECATED_REASON("Y2024: Use MemoryUtils::getDefaultDataAllocator() instead.")
ARCANE_UTILS_EXPORT IMemoryAllocator* getAcceleratorHostMemoryAllocator();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Sets the allocator specific to accelerators.
 *
 * Returns the previously used allocator. The specified allocator must remain
 * valid throughout the application's lifetime.
 *
 * \deprecated This method is internal to Arcane.
 */
extern "C++" ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane")
ARCANE_UTILS_EXPORT IMemoryAllocator* setAcceleratorHostMemoryAllocator(IMemoryAllocator* a);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Default allocator for data.
 *
 * This allocator uses the one returned by getAcceleratorHostMemoryAllocator()
 * if available; otherwise, it uses an aligned allocator.
 *
 * It is guaranteed that the returned allocator will allow the data
 * to be used on the accelerator if available.
 *
 * It is guaranteed that the alignment is at least that returned by
 * AlignedMemoryAllocator::Simd().
 */
extern "C++" ARCANE_UTILS_EXPORT IMemoryAllocator*
getDefaultDataAllocator();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Sets the memory resource manager for data.
 *
 * The manager must remain valid throughout the program's execution.
 *
 * Returns the previously used manager.
 *
 * \deprecated This method is internal to Arcane.
 */
extern "C++" ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane")
ARCANE_UTILS_EXPORT IMemoryRessourceMng* setDataMemoryRessourceMng(IMemoryRessourceMng* mng);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Memory resource manager for data.
 *
 * It is guaranteed that the alignment is at least that returned by
 * AlignedMemoryAllocator::Simd().
 *
 * \deprecated This method is internal to Arcane. Use methods from MemoryUtils
 * instead.
 */
extern "C++" ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane. Use methods from MemoryUtils instead.")
ARCANE_UTILS_EXPORT IMemoryRessourceMng* getDataMemoryRessourceMng();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Reads the content of a file and stores it in \a out_bytes.
 *
 * Reads the file named \a filename and fills \a out_bytes with the content
 * of this file. If \a is_binary is true, the file is opened in binary mode.
 * Otherwise, it is opened in text mode.
 *
 * \retval true in case of error
 * \retval false otherwise.
 */
extern "C++" ARCANE_UTILS_EXPORT bool
readAllFile(StringView filename, bool is_binary, ByteArray& out_bytes);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Reads the content of a file and stores it in \a out_bytes.
 *
 * Reads the file named \a filename and fills \a out_bytes with the content
 * of this file. If \a is_binary is true, the file is opened in binary mode.
 * Otherwise, it is opened in text mode.
 *
 * \retval true in case of error
 * \retval false otherwise.
 */
extern "C++" ARCANE_UTILS_EXPORT bool
readAllFile(StringView filename, bool is_binary, Array<std::byte>& out_bytes);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Returns the full name with the executable path.
 */
extern "C++" ARCANE_UTILS_EXPORT String
getExeFullPath();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Fills \a arg_list with command line arguments.
 *
 * This function fills \a arg_list with the arguments used in the call to
 * main().
 *
 * Currently, this method only works on Linux. For other platforms, it
 * returns an empty list.
 */
extern "C++" ARCANE_UTILS_EXPORT void
fillCommandLineArguments(StringList& arg_list);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Retrieves the call stack via gdb.
 *
 * This method only works on Linux and if GDB is installed. In other cases,
 * the null string is returned.
 *
 * This method calls the std::system() command to launch gdb, which must be
 * in the PATH. Since gdb then loads the debug symbols, the command can take
 * a long time to execute.
 */
extern "C++" ARCANE_UTILS_EXPORT String
getGDBStack();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Retrieves the call stack via lldb.
 *
 * This method is similar to getGDBStack() but uses 'lldb' to retrieve the
 * call stack. If `dotnet-sos` is installed, it also allows retrieving
 * information about 'dotnet' runtime methods.
 */
extern "C++" ARCANE_UTILS_EXPORT String
getLLDBStack();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Definition of the pragma to indicate iteration independence

/*!
 * \def ARCANE_PRAGMA_IVDEP
 * Pragma to indicate to the compiler that loop iterations are
 * independent. This pragma is placed before a 'for' loop.
 */

/*!
 * \def ARCANE_PRAGMA_IVDEP_VALUE
 * Value of the ARCANE_PRAGMA_IVDEP pragma
 */

// For definitions, GCC must be last because Clang and ICC define
// the macro __GNU__
// For CLANG, there is no equivalent to the ICC pragma ivdep yet.
// The closest one is:
//   #pragma clang loop vectorize(enable)
// but it does not force vectorization.
#ifdef __clang__
#define ARCANE_PRAGMA_IVDEP_VALUE "clang loop vectorize(enable)"
#else
#ifdef __INTEL_COMPILER
#define ARCANE_PRAGMA_IVDEP_VALUE "ivdep"
#else
#ifdef __GNUC__
#if (__GNUC__ >= 5)
#define ARCANE_PRAGMA_IVDEP_VALUE "GCC ivdep"
#endif
#endif
#endif
#endif

#ifdef ARCANE_PRAGMA_IVDEP_VALUE
#define ARCANE_PRAGMA_IVDEP _Pragma(ARCANE_PRAGMA_IVDEP_VALUE)
#else
#define ARCANE_PRAGMA_IVDEP
#define ARCANE_PRAGMA_IVDEP_VALUE ""
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::platform

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
