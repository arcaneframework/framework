// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExceptionUtils.h                                            (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires pour la gestion des exceptions.                     */
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
 * \brief Imprime un message pour une exception inconnue.
 *
 * Cette fonction sert pour les expressions du type `catch(...)`.
 *
 * Si \a trace_mng est non nul, il sera utilisé pour l'impression.
 * Si \a is_no_continue est vrai, affiche un message indiquant qu'on ne peut
 * plus continuer l'exécution.
 *
 * \retval 1
 */
extern "C++" ARCCORE_COMMON_EXPORT Int32
print(ITraceMng* tm, bool is_no_continue = true);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Imprime un message pour l'exception standard \a ex.
 *
 * Si \a trace_mng est non nul, il sera utilisé pour l'impression.
 * Si \a is_no_continue est vrai, affiche un message indiquant qu'on ne peut
 * plus continuer l'exécution.
 *
 * \retval 2
 */
extern "C++" ARCCORE_COMMON_EXPORT Int32
print(const std::exception& ex, ITraceMng* tm, bool is_no_continue = true);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Imprime un message pour l'exception standard \a ex.
 *
 * Si \a tm est non nul, il sera utilisé pour l'impression.
 * Si \a is_no_continue est vrai, affiche un message indiquant qu'on ne peut
 * plus continuer l'exécution.
 *
 * \retval 3
 */
extern "C++" ARCCORE_COMMON_EXPORT Int32
print(const Exception& ex, ITraceMng* tm, bool is_no_continue = true);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Appelle une fonction en récupérant et affichant les exceptions.
 *
 * Applique la fonction \a function et récupère les éventuelles exceptions.
 * En cas d'exception, la fonction print() est appelée pour afficher un
 * message et le code de retour est celui de la fonction print().
 *
 * Usage:
 *
 * \code
 * callWithTryCatch([&]() { std::cout << "Hello\n"});
 * \endcode
 *
 * \return 0 si aucune exception n'est lancée et une valeur positive dans
 * le cas contraire.
 */
extern "C++" ARCCORE_COMMON_EXPORT Int32
callWithTryCatch(std::function<void()> function, ITraceMng* tm = nullptr);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Appelle une fonction et termine le programme en cas d'exception.
 *
 * Appelle la fonction \a function via callWithTryCatch() et appelle
 * std::terminate() en cas d'exception.
 */
extern "C++" ARCCORE_COMMON_EXPORT void
callAndTerminateIfThrow(std::function<void()> function, ITraceMng* tm = nullptr);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::ExceptionUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
