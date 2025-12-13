// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExceptionUtils.cc                                           (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires pour la gestion des exceptions.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/ExceptionUtils.h"

#include "arccore/base/Exception.h"
#include "arccore/trace/ITraceMng.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*!
 * \namespace ExceptionUtils
 *
 * \brief Fonctions utilitaires pour la gestion des exceptions.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  const char* _noContinueString(bool is_no_continue)
  {
    return (is_no_continue) ? "** Can't continue with the execution.\n" : "";
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ExceptionUtils::
print(ITraceMng* msg, bool is_no_continue)
{
  const char* nc = _noContinueString(is_no_continue);
  const char* msg_str = "** An unknowed error occured...\n";
  if (msg) {
    msg->error() << msg_str << nc;
  }
  else {
    std::cerr << msg_str << nc;
  }
  return 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ExceptionUtils::
print(const std::exception& ex, ITraceMng* msg, bool is_no_continue)
{
  const char* nc = _noContinueString(is_no_continue);
  if (msg) {
    msg->error() << "** A standard exception occured: " << ex.what() << ".\n"
                 << nc;
  }
  else {
    std::cerr << "** A standard exception occured: " << ex.what() << ".\n"
              << nc;
  }
  return 2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ExceptionUtils::
print(const Exception& ex, ITraceMng* msg, bool is_no_continue)
{
  const char* nc = _noContinueString(is_no_continue);
  if (msg) {
    if (!ex.isCollective() || msg->isMaster())
      msg->error() << ex << '\n'
                   << nc;
  }
  else {
    std::cerr << "EXCEPTION: " << ex << '\n'
              << nc;
  }
  return 3;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ExceptionUtils::
callWithTryCatch(std::function<void()> function, ITraceMng* tm)
{
  try {
    function();
  }
  catch (const Exception& ex) {
    return print(ex, tm, false);
  }
  catch (const std::exception& ex) {
    return print(ex, tm, false);
  }
  catch (...) {
    return print(tm, false);
  }
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExceptionUtils::
callAndTerminateIfThrow(std::function<void()> function, ITraceMng* tm)
{
  int r = callWithTryCatch(function, tm);
  if (r != 0) {
    std::cerr << "Exception catched in arcaneCallFunctionAndTerminateIfThrow: calling std::terminate()\n";
    std::terminate();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
