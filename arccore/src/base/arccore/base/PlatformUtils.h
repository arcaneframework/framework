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
#ifndef ARCCORE_BASE_PLATFORMUTILS_H
#define ARCCORE_BASE_PLATFORMUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IStackTraceService;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Namespace for platform-dependent functions.
 * 
 * This namespace contains all platform-dependent functions.
 */
namespace Arcane::Platform
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Platform-specific initialization.
 *
 * This routine is called when the architecture is initialized.
 * It allows certain operations that depend on the platform to be performed.
 *
 * Activates floating exceptions if they are available.
 */
extern "C++" ARCCORE_BASE_EXPORT void
platformInitialize();

/*!
 * \brief Platform-specific initialization.
 *
 * This routine is called when the architecture is initialized.
 * It allows certain operations that depend on the platform to be performed.
 *
 * If \a enable_fpe is true, floating exceptions are enabled if they are
 * available (via the call to enableFloatingException().
 */
extern "C++" ARCCORE_BASE_EXPORT void
platformInitialize(bool enable_fpe);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Platform-specific program termination routines.
 *
 Cette routine is called just before exiting the program.
 */
extern "C++" ARCCORE_BASE_EXPORT void
platformTerminate();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Current date.
 *
 * The string is returned in the format day/month/year.
 */
extern "C++" ARCCORE_BASE_EXPORT String
getCurrentDate();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Current time.
 *
 * Returns the current date, expressed in seconds elapsed
 * since January 1, 1970.
 */
extern "C++" ARCCORE_BASE_EXPORT long
getCurrentTime();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Clock time in nanoseconds.
 */
extern "C++" ARCCORE_BASE_EXPORT Int64
getRealTimeNS();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Current date and time in ISO 8601 format.
 *
 * The string is returned in the format YYYY-MM-DDTHH:MM:SS.
 */
extern "C++" ARCCORE_BASE_EXPORT String
getCurrentDateTime();

/*!
 * \brief Name of the machine running the process.
 */
extern "C++" ARCCORE_BASE_EXPORT String
getHostName();

/*!
 * \brief Current directory path.
 */
extern "C++" ARCCORE_BASE_EXPORT String
getCurrentDirectory();

/*!
 * \brief Process ID.
 */
extern "C++" ARCCORE_BASE_EXPORT int
getProcessId();

/*!
 * \brief Username.
 */
extern "C++" ARCCORE_BASE_EXPORT String
getUserName();

/*!
 * \brief Directory containing user documents.
 *
 * This corresponds to the HOME environment variable on Unix,
 * or the 'My Documents' directory under Win32.
 */
extern "C++" ARCCORE_BASE_EXPORT String
getHomeDirectory();

/*!
 * \brief Length of the file \a filename.
 * If the file is not readable or does not exist, returns 0.
 */
extern "C++" ARCCORE_BASE_EXPORT long unsigned int
getFileLength(const String& filename);

/*!
 * \brief Environment variable named \a name.
 *
 * If no variable named \a name is defined,
 * an empty string is returned.
 */
extern "C++" ARCCORE_BASE_EXPORT String
getEnvironmentVariable(const String& name);

/*!
 * \brief Create a directory.
 *
 * Creates the directory named \a dir_name. If necessary, creates the
 * required parent directories.
 *
 * \retval true in case of failure,
 * \retval false in case of success or if the directory already exists.
 */
extern "C++" ARCCORE_BASE_EXPORT bool
recursiveCreateDirectory(const String& dir_name);

/*!
 * \brief Create a directory.
 *
 * Creates a directory named \a dir_name. This function assumes
 * that the parent directory already exists.
 *
 * \retval true in case of failure,
 * \retval false in case of success or if the directory already exists.
 */
extern "C++" ARCCORE_BASE_EXPORT bool
createDirectory(const String& dir_name);

/*!
 * \brief Delete the file \a file_name.
 *
 * \retval true in case of failure,
 * \retval false in case of success or if the file does not exist.
 */
extern "C++" ARCCORE_BASE_EXPORT bool
removeFile(const String& file_name);

/*!
 * \brief Checks if the file \a file_name is accessible and readable.
 *
 * \retval true if the file is readable,
  * \retval false otherwise.
 */
extern "C++" ARCCORE_BASE_EXPORT bool
isFileReadable(const String& file_name);

/*!
 * \brief Returns the directory name of a file.
 *
 * Returns the directory name where the file
 * named \a file_name is located.
 * For example, if \a file_name is "/tmp/toto.cc", it returns "/tmp".
 * If the file does not contain directories, it returns \c ".".
 */
extern "C++" ARCCORE_BASE_EXPORT String
getFileDirName(const String& file_name);

/*!
 * \brief Memory block copy
 *
 * Copies \a len bytes from address \a from to address \a to.
 */
extern "C++" ARCCORE_BASE_EXPORT void
stdMemcpy(void* to, const void* from, ::size_t len);

/*!
 * \brief Memory used in bytes
 *
 * \return the memory used or a negative number if unknown
 */
extern "C++" ARCCORE_BASE_EXPORT double
getMemoryUsed();

/*!
 * \brief CPU time used in microseconds.
 *
 * The origin of the CPU time is taken when calling platformInitialize().
 *
 * \return the CPU time used in microseconds.
 */
extern "C++" ARCCORE_BASE_EXPORT Int64
getCPUTime();

/*!
 * \brief Real time used in seconds.
 *
 * \return the time used in seconds.
 */
extern "C++" ARCCORE_BASE_EXPORT Real
getRealTime();

/*!
 * \brief Returns time in hours, minutes, and seconds format.
 *
 * Converts \a t, expressed in seconds, into the format AhBmCs
 * where A is hours, B is minutes, and C is seconds.
 * For example, 3732 becomes 1h2m12s.
 */
extern "C++" ARCCORE_BASE_EXPORT String
timeToHourMinuteSecond(Real t);

/*!
 * \brief Returns \a true if \a v is denormalized (invalid float).
 *
 * If the platform does not support this concept, it always returns \a false.
 */
extern "C++" ARCCORE_BASE_EXPORT bool
isDenormalized(Real v);

/*!
 * \brief Service used to obtain the call stack.
 *
 * May return null if no service is available.
 */
extern "C++" ARCCORE_BASE_EXPORT IStackTraceService*
getStackTraceService();

/*!
 * \brief Sets the service used to obtain the call stack.
 *
 * Returns the previously used service.
 */
extern "C++" ARCCORE_DEPRECATED_REASON("Y2025: This method is internal to Arcane")
ARCCORE_BASE_EXPORT IStackTraceService* setStackTraceService(IStackTraceService* service);

/*!
 * \brief Returns a string containing the call stack.
 *
 * If no call stack management service is present
 * (getStackTraceService()==0), the returned string is null.
 */
extern "C++" ARCCORE_BASE_EXPORT String
getStackTrace();

/*!
 * \brief Service used to obtain information
 * about source code symbols.
 *
 * May return null if no service is available.
 */
extern "C++" ARCCORE_BASE_EXPORT ISymbolizerService*
getSymbolizerService();

/*!
 * \brief Sets the service to obtain information
 * about source code symbols.
 *
 * Returns the previously used service.
 */
extern "C++" ARCCORE_DEPRECATED_REASON("Y2025: This method is internal to Arcane")
ARCCORE_BASE_EXPORT ISymbolizerService* setSymbolizerService(ISymbolizerService* service);

/*
 * \brief Copies a character string with overflow check.
 *
 * \param input string to copy.
 * \param output pointer where the string will be copied.
 * \param output_len memory allocated for \a output.
 */
extern "C++" ARCCORE_BASE_EXPORT void
safeStringCopy(char* output, Integer output_len, const char* input);

/*!
 * \brief Puts the process to sleep for \a nb_second seconds.
 */
extern "C++" ARCCORE_BASE_EXPORT void
sleep(Integer nb_second);

/*!
 * \brief Enables or disables exceptions during a floating-point calculation.
 * This operation is not supported on all platforms. If it is not supported, nothing happens.
 */
extern "C++" ARCCORE_BASE_EXPORT void
enableFloatingException(bool active);

//! Indicates if processor floating exceptions are enabled.
extern "C++" ARCCORE_BASE_EXPORT bool
isFloatingExceptionEnabled();

/*!
 * \brief Raises a floating exception.
 *
 * This method does nothing if hasFloatingExceptionSupport()==false.
 * Generally under Linux, this translates to sending a signal
 * of type SIGFPE. By default, %Arccore catches this signal and
 * raises an 'ArithmeticException'.
 */
extern "C++" ARCCORE_BASE_EXPORT void
raiseFloatingException();

/*!
 * \brief Indicates if the implementation allows modifying
 * the floating exception activation state.
 *
 * If this method returns \a false, then the methods
 * enableFloatingException() and isFloatingExceptionEnabled()
 * have no effect.
 */
extern "C++" ARCCORE_BASE_EXPORT bool
hasFloatingExceptionSupport();

/*!
 * \brief Dumps the call stack to the stream \a ostr.
 */
extern "C++" ARCCORE_BASE_EXPORT void
dumpStackTrace(std::ostream& ostr);

/*!
 * \brief Indicates if the console supports colors.
 */
extern "C++" ARCCORE_BASE_EXPORT bool
getConsoleHasColor();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Character string used to identify the compiler
 * used to compile %Arccore.
 */
extern "C++" ARCCORE_BASE_EXPORT String
getCompilerId();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Host system page size in bytes
 */
extern "C++" ARCCORE_BASE_EXPORT Int64
getPageSize();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Returns the full path of a loaded dynamic library.
 *
 * Returns the full path of the dynamic library named
 * \a dll_name. \a dll_name must only contain the library name
 * without platform-specific extensions. For example, on Linux,
 * do not use 'libtoto.so' but just 'toto'.
 *
 * Returns a null string if the full path cannot
 * be determined.
 */
extern "C++" ARCCORE_BASE_EXPORT String
getLoadedSharedLibraryFullPath(const String& dll_name);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Platform

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
namespace Platform = Arcane::Platform;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::platform
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using Arcane::Platform::createDirectory;
using Arcane::Platform::getCPUTime;
using Arcane::Platform::getCurrentDate;
using Arcane::Platform::getCurrentDateTime;
using Arcane::Platform::getCurrentDirectory;
using Arcane::Platform::getCurrentTime;
using Arcane::Platform::getEnvironmentVariable;
using Arcane::Platform::getFileDirName;
using Arcane::Platform::getFileLength;
using Arcane::Platform::getHomeDirectory;
using Arcane::Platform::getHostName;
using Arcane::Platform::getMemoryUsed;
using Arcane::Platform::getPageSize;
using Arcane::Platform::getProcessId;
using Arcane::Platform::getRealTime;
using Arcane::Platform::getRealTimeNS;
using Arcane::Platform::getUserName;
using Arcane::Platform::isDenormalized;
using Arcane::Platform::isFileReadable;
using Arcane::Platform::recursiveCreateDirectory;
using Arcane::Platform::removeFile;
using Arcane::Platform::safeStringCopy;
using Arcane::Platform::sleep;
using Arcane::Platform::stdMemcpy;
using Arcane::Platform::timeToHourMinuteSecond;

using Arcane::Platform::enableFloatingException;
using Arcane::Platform::hasFloatingExceptionSupport;
using Arcane::Platform::isFloatingExceptionEnabled;
using Arcane::Platform::raiseFloatingException;

using Arcane::Platform::dumpStackTrace;
using Arcane::Platform::getStackTrace;
using Arcane::Platform::getStackTraceService;
using Arcane::Platform::getSymbolizerService;
using Arcane::Platform::setStackTraceService;
using Arcane::Platform::setSymbolizerService;

using Arcane::Platform::getCompilerId;
using Arcane::Platform::getConsoleHasColor;

using Arcane::Platform::getLoadedSharedLibraryFullPath;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::platform

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
