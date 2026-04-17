// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlinaSamplesCommon.cc                                       (C) 2000-2026 */
/*                                                                           */
/* Utilitary functions used by all samples.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "./AlinaSamplesCommon.h"

#include "arcane/utils/Exception.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/launcher/ArcaneLauncher.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/IProfilingService.h"

#include "arccore/base/ConcurrencyBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int SampleMainContext::
execMain(MainFunction f, int argc, char* argv[])
{
  if (!f) {
    std::cerr << "Invalid null functor 'f'\n";
    return 1;
  }

  auto func = [&] {
    CommandLineArguments cmd_line_args(&argc, &argv);
    ArcaneLauncher::init(cmd_line_args);
    StandaloneSubDomain launcher{ ArcaneLauncher::createStandaloneSubDomain({}) };
    IProfilingService* ps = platform::getProfilingService();
    ISubDomain* sd = launcher.subDomain();
    ITraceMng* tm = sd->traceMng();
    SampleMainContext ctx(tm, sd->acceleratorMng(), sd->parallelMng()->messagePassingMng());
    std::cout << "ConcurrencyLevel=" << ConcurrencyBase::maxAllowedThread() << "\n";
    {
      ProfilingSentryWithInitialize ps_sentry(ps);
      ps_sentry.setPrintAtEnd(true);
      (*f)(ctx, argc, argv);
    }
  };
  return arcaneCallFunctionAndCatchException(func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
