// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Exception.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Déclarations et définitions liées aux exceptions.                         */
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

// TODO: Rendre ces méthode obsolète fin 2026 et indiquer qu'il faut
// utiliser ExceptionUtils à la place.

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
 * \ brief Appelle la fonction \a function en récupérant les éventuelles exceptions.
 *
 * Usage:
 *
 * \code
 * arcaneCallAndCatchException([&]() { std::cout << "Hello\n"});
 * \endcode
 *
 * \return 0 si aucune exception n'est récupérée et une valeur positive dans
 * le cas contraire.
 */
extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCallFunctionAndCatchException(std::function<void()> function);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ brief Appelle la fonction \a function et en cas d'exception appelle std::terminate().
 */
extern "C++" ARCANE_UTILS_EXPORT void
arcaneCallFunctionAndTerminateIfThrow(std::function<void()> function);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
