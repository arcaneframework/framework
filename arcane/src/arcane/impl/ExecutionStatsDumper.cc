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
#include "arcane/utils/JSONWriter.h"

#include "arcane/utils/internal/ProfilingInternal.h"

#include "arcane/ISubDomain.h"
#include "arcane/IVariableMng.h"
#include "arcane/IPropertyMng.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/IMesh.h"
#include "arcane/ITimeStats.h"
#include "arcane/Directory.h"
#include "arcane/IParallelMng.h"

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
  json_writer.write("Version","1");
  auto f = [&](const impl::ForLoopStatInfoList& stat_list){
    JSONWriter::Object jo(json_writer);
    json_writer.writeKey("Loops");
    json_writer.beginArray();
    for (const auto& x : stat_list._internalImpl()->m_stat_map) {
      JSONWriter::Object jo2(json_writer);
      const auto& s = x.second;
      json_writer.write("Name",x.first);
      json_writer.write("TotalTime",s.execTime());
      json_writer.write("NbLoop",s.nbCall());
      json_writer.write("NbChunk",s.nbChunk());
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
  JSONWriter json_writer; //(JSONWriter::FormatFlags::None);
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
_dumpProfiling(std::ostream& o)
{
  // Affiche les informations de profiling sur \a o
  _printGlobalLoopInfos(o,ProfilingRegistry::globalLoopStat());
  auto f = [&](const impl::ForLoopStatInfoList& stat_list){
    _dumpOneLoopListStat(o,stat_list);
  };
  ProfilingRegistry::visitLoopStat(f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExecutionStatsDumper::
_dumpOneLoopListStat(std::ostream& o,const impl::ForLoopStatInfoList& stat_list)
{
  o << "ProfilingStat\n";
  o << std::setw(10) << "Ncall" << std::setw(10) << "Nchunk"
    << std::setw(10) << " T (us)" << std::setw(11) << "Tck (ns)\n";
  Int64 cumulative_total = 0;
  for (const auto& x : stat_list._internalImpl()->m_stat_map) {
    const auto& s = x.second;
    Int64 nb_loop = s.nbCall();
    Int64 nb_chunk = s.nbChunk();
    Int64 total_time = s.execTime();
    Int64 time_per_chunk = (nb_chunk == 0) ? 0 : (total_time / nb_chunk);
    o << std::setw(10) << nb_loop << std::setw(10) << nb_chunk
      << std::setw(10) << total_time / 1000 << std::setw(10) << time_per_chunk << "  " << x.first << "\n";
    cumulative_total += total_time;
  }
  o << "TOTAL=" << cumulative_total / 1000000 << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExecutionStatsDumper::
_printGlobalLoopInfos(std::ostream& o,const impl::ForLoopCumulativeStat& cumulative_stat)
{
  Int64 nb_loop_parallel_for = cumulative_stat.nbLoopParallelFor();
  if (nb_loop_parallel_for == 0)
    return;
  Int64 nb_chunk_parallel_for = cumulative_stat.nbChunkParallelFor();
  Int64 total_time = cumulative_stat.totalTime();
  double x = static_cast<double>(total_time);
  double x1 = 0.0;
  if (nb_loop_parallel_for > 0)
    x1 = x / static_cast<double>(nb_loop_parallel_for);
  double x2 = 0.0;
  if (nb_chunk_parallel_for > 0)
    x2 = x / static_cast<double>(nb_chunk_parallel_for);
  o << "LoopStat: global_time (ms) = " << x / 1.0e6 << "\n";
  o << "LoopStat: global_nb_loop   = " << std::setw(10) << nb_loop_parallel_for << " time=" << x1 << "\n";
  o << "LoopStat: global_nb_chunk  = " << std::setw(10) << nb_chunk_parallel_for << " time=" << x2 << "\n";
}

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

  // Informations de profiling des boucles
  {
    OStringStream ostr;
    _dumpProfiling(ostr());
    String str = ostr.str();
    if (!str.empty()){
      info() << "LoopStatistics:\n" << str;
      if (sd){
        Directory dir = sd->listingDirectory();
        String filename = String::format("loop_profiling.{0}.json",sd->parallelMng()->commRank());
        String full_filename(dir.file(filename));
        _dumpProfilingJSON(full_filename);
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
