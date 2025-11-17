// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HostKernelRemainingArgsHelper.h                             (C) 2000-2025 */
/*                                                                           */
/* Classe pour exécuter une méthode en début et fin de noyau.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_HOSTKERNELREMAININGARGSHELPER_H
#define ARCCORE_COMMON_HOSTKERNELREMAININGARGSHELPER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe pour appliquer des méthodes des arguments additionnels
 * en début et fin de kernel.
 */
class HostKernelRemainingArgsHelper
{
 public:

  //! Applique les functors des arguments additionnels au début de l'itération.
  template <typename... RemainingArgs> static void
  applyRemainingArgsAtBegin(RemainingArgs&... remaining_args)
  {
    (_doOneAtBegin(remaining_args), ...);
  }

  //! Applique les functors des arguments additionnels à la fin de l'itération.
  template <typename... RemainingArgs> static void
  applyRemainingArgsAtEnd(RemainingArgs&... remaining_args)
  {
    (_doOneAtEnd(remaining_args), ...);
  }

 private:

  template <typename OneArg> static void _doOneAtBegin(OneArg& one_arg)
  {
    //if constexpr (requires { one_arg._internalHostExecWorkItemAtBegin(); })
    one_arg._internalHostExecWorkItemAtBegin();
  }
  template <typename OneArg> static void _doOneAtEnd(OneArg& one_arg)
  {
    //if constexpr (requires { one_arg._internalHostExecWorkItemAtEnd(); })
    one_arg._internalHostExecWorkItemAtEnd();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
