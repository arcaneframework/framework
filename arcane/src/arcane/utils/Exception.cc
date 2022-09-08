// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Exception.cc                                                (C) 2000-2022 */
/*                                                                           */
/* Déclarations et définitions liées aux exceptions.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"
#include "arcane/utils/ITraceMng.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
const char* _noContinueString(bool is_no_continue)
{
  return (is_no_continue) ? "** Can't continue with the execution.\n" : "";
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT Integer
arcanePrintAnyException(ITraceMng* msg,bool is_no_continue)
{
  const char* nc = _noContinueString(is_no_continue);
  const char* msg_str = "** An unknowed error occured...\n";
  if (msg){
    msg->error() << msg_str << nc;
  }
  else{
    std::cerr << msg_str << nc;
  }
  return 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT Integer
arcanePrintStdException(const std::exception& ex,ITraceMng* msg,bool is_no_continue)
{
  const char* nc = _noContinueString(is_no_continue);
  if (msg){
    msg->error() << "** A standard exception occured: " << ex.what() << ".\n" << nc;
  }
  else{
    std::cerr << "** A standard exception occured: " << ex.what() << ".\n" << nc;
  }
  return 2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT Integer
arcanePrintArcaneException(const Exception& ex,ITraceMng* msg,bool is_no_continue)
{
  const char* nc = _noContinueString(is_no_continue);
  if (msg){
    if (!ex.isCollective() || msg->isMaster())
      msg->error() << ex << '\n' << nc;
  }
  else{
    std::cerr << "EXCEPTION: " << ex << '\n' << nc;
  }
  return 3;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
