// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExecutionStatsDumper.cc                                     (C) 2000-2026 */
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
#include "arcane/utils/JSONWriter.h"
#include "arcane/utils/FloatingPointExceptionSentry.h"

#include "arccore/base/internal/ProfilingInternal.h"

#include "arcane/core/ISubDomain.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IVariableSynchronizerMng.h"
#include "arcane/core/IPropertyMng.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/ITimeStats.h"
#include "arcane/core/Directory.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/ISimpleTableOutput.h"

#include <iostream>
#include <iomanip>
#include <fstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExecutionStatsDumper::
_dumpProfilingJSON(JSONWriter& json_writer)
{
  json_writer.write("Version", "1");
  auto f = [&](const Impl::ForLoopStatInfoList& stat_list) {
    JSONWriter::Object jo(json_writer);
    json_writer.writeKey("Loops");
    json_writer.beginArray();
    for (const auto& x : stat_list._internalImpl()->m_stat_map) {
      JSONWriter::Object jo2(json_writer);
      const auto& s = x.second;
      json_writer.write("Name", x.first);
      json_writer.write("TotalTime", s.execTime());
      json_writer.write("NbLoop", s.nbCall());
      json_writer.write("NbChunk", s.nbChunk());
    }
    json_writer.endArray();
  };

  json_writer.writeKey("AllLoops");
  json_writer.beginArray();
  ProfilingRegistry::visitLoopStat(f);
  json_writer.endArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExecutionStatsDumper::
_dumpProfilingJSON(const String& filename)
{
  JSONWriter json_writer;
  {
    JSONWriter::Object jo(json_writer);
    _dumpProfilingJSON(json_writer);
  }
  {
    ofstream ofile(filename.localstr());
    ofile << json_writer.getBuffer();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExecutionStatsDumper::
_dumpProfilingTable(ISimpleTableOutput* table)
{
  table->init("Profiling");

  table->addColumn("TotalTime");
  table->addColumn("NbLoop");
  table->addColumn("NbChunk");

  Integer list_index = 0;
  auto f = [&](const Impl::ForLoopStatInfoList& stat_list) {
    for (const auto& x : stat_list._internalImpl()->m_stat_map) {
      const auto& s = x.second;
      String row_name = x.first;
      if (list_index > 0)
        row_name = String::format("{0}_{1}", x.first, list_index);
      Integer row = table->addRow(row_name);
      table->addElementInRow(row, static_cast<Real>(s.execTime()));
      table->addElementInRow(row, static_cast<Real>(s.nbCall()));
      table->addElementInRow(row, static_cast<Real>(s.nbChunk()));
    }
    ++list_index;
  };

  ProfilingRegistry::visitLoopStat(f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExecutionStatsDumper::
_dumpProfiling(std::ostream& o)
{
  Impl::dumpProfilingStatistics(o);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExecutionStatsDumper::
dumpStats(ISubDomain* sd, ITimeStats* time_stat)
{
  FloatingPointExceptionSentry fp_sentry(false);
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
    IVariableMng* vm = sd->variableMng();
    // Affiche les statistiques sur les synchronisations
    IVariableSynchronizerMng* vsm = vm->synchronizerMng();
    OStringStream ostr_vsm;
    vsm->dumpStats(ostr_vsm());
    info() << ostr_vsm.str();
    // Affiche les statistiques sur les variables
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

  // Informations de profiling des boucles
  {
    OStringStream ostr;
    _dumpProfiling(ostr());
    String str = ostr.str();
    if (!str.empty()) {
      info() << "LoopStatistics:\n"
             << str;
      if (sd) {
        Directory dir = sd->listingDirectory();
        Int32 rank = sd->parallelMng()->commRank();
        {
          String filename = String::format("loop_profiling.{0}.json", rank);
          String full_filename(dir.file(filename));
          _dumpProfilingJSON(full_filename);
        }
        {
          ServiceBuilder<ISimpleTableOutput> sb(sd);
          Ref<ISimpleTableOutput> table(sb.createReference("SimpleCsvOutput"));
          _dumpProfilingTable(table.get());
          table->writeFile(0);
        }
      }
      else
        _dumpProfilingJSON("loop_profiling.json");
    }
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
