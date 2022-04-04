// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DirectExecutionFunctor.h                                    (C) 2000-2022 */
/*                                                                           */
/* Fonctor pour l'exécution directe.                                         */
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/internal/DirectExecutionFunctor.h"

#include "arcane/launcher/ArcaneLauncher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int DirectExecutionWrapper::
run()
{
  return ArcaneLauncher::run();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int DirectExecutionWrapper::
run(IDirectExecutionFunctor* f)
{
  std::cout << "DIRECT CALL (DirectExecutionContext)\n";
  if (!f)
    return (-1);
  auto f2 = [=](DirectExecutionContext& ctx2) -> int
           {
             return f->execute(&ctx2);
           };
  return ArcaneLauncher::run(f2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int DirectExecutionWrapper::
run(IDirectSubDomainExecutionFunctor* f)
{
  std::cout << "DIRECT CALL (DirectSubDomainExecutionContext)\n";
  if (!f)
    return (-1);
  auto f2 = [=](DirectSubDomainExecutionContext& ctx2) -> int
           {
             return f->execute(&ctx2);
           };
  return ArcaneLauncher::run(f2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
