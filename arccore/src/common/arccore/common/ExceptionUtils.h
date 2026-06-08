// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExceptionUtils.h                                            (C) 2000-2025 */
/*                                                                           */
/* Utility functions for exception handling.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_EXCEPTIONUTILS_H
#define ARCCORE_COMMON_EXCEPTIONUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"
#include "arccore/common/CommonGlobal.h"

#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::ExceptionUtils
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Prints a message for an unknown exception.
 *
 * This function is used for `catch(...)` expressions.
 *
 * If \a trace_mng is not null, it will be used for printing.
 * If \a is_no_continue is true, it displays a message indicating that execution cannot
 * continue.
 *
 * \retval 1
 */
extern "C++" ARCCORE_COMMON_EXPORT Int32
print(ITraceMng* tm, bool is_no_continue = true);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Prints a message for the standard exception \a ex.
 *
 * If \a trace_mng is not null, it will be used for printing.
 * If \a is_no_continue is true, it displays a message indicating that execution cannot
 * continue.
 *
 * \retval 2
 */
extern "C++" ARCCORE_COMMON_EXPORT Int32
print(const std::exception& ex, ITraceMng* tm, bool is_no_continue = true);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Prints a message for the standard exception \a ex.
 *
 * If \a tm is not null, it will be used for printing.
 * If \a is_no_continue is true, it displays a message indicating that execution cannot
 * continue.
 *
 * \retval 3
 */
extern "C++" ARCCORE_COMMON_EXPORT Int32
print(const Exception& ex, ITraceMng* tm, bool is_no_continue = true);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calls a function while catching and displaying exceptions.
 *
 * Executes the function \a function and catches any exceptions.
 * In case of an exception, the print() function is called to display a
 * message, and the return code is that of the print() function.
 *
 * Usage:
 *
 * \code
 * callWithTryCatch([&]() { std::cout << "Hello\n"});
 * \endcode
 *
 * \return 0 if no exception is thrown, and a positive value otherwise.
 */
extern "C++" ARCCORE_COMMON_EXPORT Int32
callWithTryCatch(std::function<void()> function, ITraceMng* tm = nullptr);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calls a function and terminates the program if an exception occurs.
 *
 * Calls the function \a function via callWithTryCatch() and calls
 * std::terminate() in case of an exception.
 */
extern "C++" ARCCORE_COMMON_EXPORT void
callAndTerminateIfThrow(std::function<void()> function, ITraceMng* tm = nullptr);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::ExceptionUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
