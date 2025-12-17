// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExecutionStatsDumper.cc                                     (C) 2000-2025 */
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
  // Affiche les informations de profiling sur \a o
  _printGlobalLoopInfos(o, ProfilingRegistry::globalLoopStat());
  {
    auto f = [&](const Impl::ForLoopStatInfoList& stat_list) {
      _dumpOneLoopListStat(o, stat_list);
    };
    ProfilingRegistry::visitLoopStat(f);
  }
  // Avant d'afficher le profiling accélérateur, il faudrait être certain
  // qu'il est désactivé. Normalement c'est le cas si on utilise ArcaneMainBatch.
  {
    auto f = [&](const Impl::AcceleratorStatInfoList& stat_list) {
      stat_list.print(o);
    };
    ProfilingRegistry::visitAcceleratorStat(f);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExecutionStatsDumper::
_dumpOneLoopListStat(std::ostream& o, const Impl::ForLoopStatInfoList& stat_list)
{
  struct SortedStatInfo
  {
    bool operator<(const SortedStatInfo& rhs) const
    {
      return m_stat.execTime() > rhs.m_stat.execTime();
    }
    String m_name;
    Impl::ForLoopProfilingStat m_stat;
  };

  // Met 1 pour éviter de diviser par zéro.
  Int64 cumulative_total = 1;

  // Tri les fonctions par temps d'exécution décroissant
  std::set<SortedStatInfo> sorted_set;
  for (const auto& x : stat_list._internalImpl()->m_stat_map) {
    const auto& s = x.second;
    sorted_set.insert({ x.first, s });
    cumulative_total += s.execTime();
  }

  o << "ProfilingStat\n";
  o << std::setw(10) << "Ncall" << std::setw(10) << "Nchunk"
    << std::setw(11) << " T (ms)" << std::setw(10) << "Tck (ns)"
    << "     %  name\n";

  char old_filler = o.fill();
  for (const auto& x : sorted_set) {
    const Impl::ForLoopProfilingStat& s = x.m_stat;
    Int64 nb_loop = s.nbCall();
    Int64 nb_chunk = s.nbChunk();
    Int64 total_time_ns = s.execTime();
    Int64 total_time_us = total_time_ns / 1000;
    Int64 total_time_ms = total_time_us / 1000;
    Int64 total_time_remaining_us = total_time_us % 1000;
    Int64 time_per_chunk = (nb_chunk == 0) ? 0 : (total_time_ns / nb_chunk);
    Int64 per_mil = (total_time_ns * 1000) / cumulative_total;
    Int64 percent = per_mil / 10;
    Int64 percent_digit = per_mil % 10;

    o << std::setw(10) << nb_loop << std::setw(10) << nb_chunk
      << std::setw(7) << total_time_ms << ".";
    o << std::setfill('0') << std::setw(3) << total_time_remaining_us << std::setfill(old_filler);
    o << std::setw(10) << time_per_chunk
      << std::setw(4) << percent << "." << percent_digit << "  " << x.m_name << "\n";
  }
  o << "TOTAL=" << cumulative_total / 1000000 << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExecutionStatsDumper::
_printGlobalLoopInfos(std::ostream& o, const Impl::ForLoopCumulativeStat& cumulative_stat)
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
