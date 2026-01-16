// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Exception.cc                                                (C) 2000-2025 */
/*                                                                           */
/* Déclarations et définitions liées aux exceptions.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"
#include "arccore/common/ExceptionUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT Integer
arcanePrintAnyException(ITraceMng* msg, bool is_no_continue)
{
  return ExceptionUtils::print(msg, is_no_continue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT Integer
arcanePrintStdException(const std::exception& ex, ITraceMng* msg, bool is_no_continue)
{
  return ExceptionUtils::print(ex, msg, is_no_continue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT Integer
arcanePrintArcaneException(const Exception& ex, ITraceMng* msg, bool is_no_continue)
{
  return ExceptionUtils::print(ex, msg, is_no_continue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCallFunctionAndCatchException(std::function<void()> function)
{
  return ExceptionUtils::callWithTryCatch(function);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
arcaneCallFunctionAndTerminateIfThrow(std::function<void()> function)
{
  ExceptionUtils::callAndTerminateIfThrow(function);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
