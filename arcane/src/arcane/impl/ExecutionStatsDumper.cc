// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExecutionStatsDumper.cc                                     (C) 2000-2022 */
/*                                                                           */
/* Ecriture des statistiques d'exécution.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/ExecutionStatsDumper.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/IMemoryInfo.h"
#include "arcane/utils/Profiling.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/ISubDomain.h"
#include "arcane/IVariableMng.h"
#include "arcane/IPropertyMng.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/IMesh.h"
#include "arcane/ITimeStats.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExecutionStatsDumper::
dumpStats(ISubDomain* sd, ITimeStats* time_stat)
{
  {
    // Statistiques sur la mémoire
    double mem = platform::getMemoryUsed();
    info() << "Memory consumption (Mo): " << mem / 1.e6;
  }
  if (sd) {
    {
      // Affiche les valeurs des propriétés
      OStringStream postr;
      postr() << "Properties:\n";
      sd->propertyMng()->print(postr());
      plog() << postr.str();
    }
    // Affiche les statistiques sur les variables
    IVariableMng* vm = sd->variableMng();
    OStringStream ostr_full;
    vm->dumpStats(ostr_full(), true);
    OStringStream ostr;
    vm->dumpStats(ostr(), false);
    plog() << ostr_full.str();
    info() << ostr.str();
  }
  {
    // Affiche les statistiques sur les variables
    IMemoryInfo* mem_info = arcaneGlobalMemoryInfo();
    OStringStream ostr;
    mem_info->printInfos(ostr());
    info() << ostr.str();
  }
  // Affiche les statistiques sur les temps d'exécution
  Integer nb_loop = 1;
  Integer nb_cell = 1;
  if (sd) {
    nb_loop = sd->timeLoopMng()->nbLoop();
    if (sd->defaultMesh())
      nb_cell = sd->defaultMesh()->nbCell();
  }
  Real n = ((Real)nb_cell);
  info() << "NB_CELL=" << nb_cell << " nb_loop=" << nb_loop;
  {
    OStringStream ostr;
    ProfilingRegistry::printExecutionStats(ostr());
    String str = ostr.str();
    if (!str.empty())
      info() << "TaskStatistics:\n"
             << str;
  }
  {
    bool use_elapsed_time = true;
    if (!platform::getEnvironmentVariable("ARCANE_USE_REAL_TIMER").null())
      use_elapsed_time = true;
    if (!platform::getEnvironmentVariable("ARCANE_USE_VIRTUAL_TIMER").null())
      use_elapsed_time = false;
    OStringStream ostr_full;
    time_stat->dumpStats(ostr_full(), true, n, "cell", use_elapsed_time);
    OStringStream ostr;
    time_stat->dumpStats(ostr(), false, n, "cell", use_elapsed_time);
    info() << ostr.str();
    plog() << ostr_full.str();
    traceMng()->flush();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
