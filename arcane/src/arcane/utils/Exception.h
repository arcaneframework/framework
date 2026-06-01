// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Exception.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Declarations and definitions related to exceptions.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_EXCEPTION_H
#define ARCANE_UTILS_EXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Exception.h"
#include "arccore/base/TraceInfo.h"
#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/Atomic.h"

#include <functional>

// TODO: Make these methods obsolete by the end of 2026 and indicate that
// ExceptionUtils should be used instead.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT Integer
arcanePrintAnyException(ITraceMng* msg,bool is_no_continue = true);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT Integer
arcanePrintStdException(const std::exception& ex,ITraceMng* msg,bool is_no_continue = true);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT Integer
arcanePrintArcaneException(const Exception& ex,ITraceMng* msg,bool is_no_continue = true);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calls the function \a function while catching potential exceptions.
 *
 * Usage:
 *
 * \code
 * arcaneCallAndCatchException([&]() { std::cout << "Hello\n"});
 * \endcode
 *
 * \return 0 if no exception is caught and a positive value otherwise.
 */
extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCallFunctionAndCatchException(std::function<void()> function);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calls the function \a function and calls std::terminate() if an exception occurs.
 */
extern "C++" ARCANE_UTILS_EXPORT void
arcaneCallFunctionAndTerminateIfThrow(std::function<void()> function);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
