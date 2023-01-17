﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Stat.cc                                                     (C) 2000-2022 */
/*                                                                           */
/* Statistiques sur le parallélisme.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/IStat.h"

#include "arcane/utils/String.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/JSONWriter.h"
#include "arcane/utils/Convert.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/IParallelMng.h"

#include "arccore/message_passing/Stat.h"

#include <cmath>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Parallel
{

using Arccore::MessagePassing::OneStat;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Statistiques sur le parallélisme.
 */
class Stat
: public Arccore::MessagePassing::Stat
, public IStat
{
 public:

  typedef Arccore::MessagePassing::Stat Base;

  Stat();
  //! Libère les ressources.
  virtual ~Stat();

  Arccore::MessagePassing::IStat* toArccoreStat() override { return this; }

  void add(const String& name, double elapsed_time, Int64 msg_size) override;
  void print(ITraceMng* msg) override;
  void enable(bool is_enabled) override { Base::enable(is_enabled); }
  void dumpJSON(JSONWriter& writer) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT IStat*
createDefaultStat()
{
  return new Stat();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT void
dumpJSON(JSONWriter& writer, const Arccore::MessagePassing::OneStat& os, bool cumulative_stat)
{
  Arcane::JSONWriter::Object o(writer, os.name());
  writer.write("Count", cumulative_stat ? os.cumulativeNbMessage() : os.nbMessage());
  writer.write("MessageSize", cumulative_stat ? os.cumulativeTotalSize() : os.totalSize());
  writer.write("TotalTime", cumulative_stat ? os.cumulativeTotalTime() : os.totalTime());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Stat::
Stat()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Stat::
~Stat()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Stat::
add(const String& name, double elapsed_time, Int64 msg_size)
{
  Arccore::MessagePassing::Stat::add(name, elapsed_time, msg_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Stat::
print(ITraceMng* msg)
{
  for (const auto& i : stats()) {
    OneStat* os = i.second;
    Real total_time = os->cumulativeTotalTime();
    Int64 div_time = static_cast<Int64>(total_time * 1000.0);
    Int64 nb_message = os->cumulativeNbMessage();
    Int64 total_size = os->cumulativeTotalSize();
    const String& name = os->name();
    if (div_time == 0)
      div_time = 1;
    if (nb_message > 0) {
      Int64 average_time = Convert::toInt64(total_time / (Real)nb_message);
      msg->info() << " MPIStat " << name << "     :" << nb_message << " messages";
      msg->info() << " MPIStat " << name << "     :" << total_size << " bytes ("
                  << total_size / div_time << " Kb/s) (average size "
                  << total_size / nb_message << " bytes)";
      msg->info() << " MPIStat " << name << " Time: " << total_time << " seconds"
                  << " (avg=" << average_time << ")";
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Stat::
dumpJSON(JSONWriter& writer)
{
  writer.writeKey("Stats");
  writer.beginArray();
  for (const auto& stat : stats())
    Parallel::dumpJSON(writer, *(stat.second)); // cumulative stats dump
  writer.endArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace
{
  std::string _formatToString(Real v)
  {
    v = v * 10000;
    Int64 v_as_int = static_cast<Int64>(std::round(v));
    Int64 a = v_as_int / 10000;
    Int64 b = v_as_int % 10000;
    std::ostringstream ostr;
    ostr << std::setfill(' ') << std::right << std::setw(6) << a << '.';
    ostr << std::setfill('0') << std::right << std::setw(4) << b;
    return ostr.str();
  }
} // namespace

extern "C++" void
printStatsCollective(IStat* s, IParallelMng* pm)
{
  // Les instances \a s de tous les rangs peuvent ne pas avoir les mêmes
  // statistiques. Pour éviter des blocages, on ne garde que les statistiques
  // communes à tout le monde.
  UniqueArray<String> input_strings;
  const auto& stat_map = s->toArccoreStat()->stats();
  for (const auto& x : stat_map)
    input_strings.add(x.first);
  UniqueArray<String> common_strings;
  MessagePassing::filterCommonStrings(pm, input_strings, common_strings);
  Int32 nb_rank = pm->commSize();
  ITraceMng* tm = pm->traceMng();
  tm->info() << "Message passing Stats (unit is second) "
             << Trace::Width(48) << "min"
             << Trace::Width(7) << "max";
  tm->info() << Trace::Width(55) << "average"
             << Trace::Width(10) << "min"
             << Trace::Width(12) << "max"
             << Trace::Width(10) << "rank"
             << Trace::Width(7) << "rank";
  for (String name : common_strings) {
    auto i = stat_map.find(name);
    if (i == stat_map.end())
      ARCANE_FATAL("Internal error: string '{0}' not in stats", name);
    Real my_time = i->second->cumulativeTotalTime();
    Real sum_time = 0.0;
    Real min_time = 0.0;
    Real max_time = 0.0;
    Int32 min_time_rank = 0;
    Int32 max_time_rank = 0;
    pm->computeMinMaxSum(my_time, min_time, max_time, sum_time, min_time_rank, max_time_rank);
    Real average_time = sum_time / static_cast<Real>(nb_rank);
    tm->info() << " MPIStatAllRanks " << Trace::Width(25) << name << " :"
               << " " << _formatToString(average_time)
               << " " << _formatToString(min_time)
               << " " << _formatToString(max_time)
               << " " << Trace::Width(6) << min_time_rank
               << " " << Trace::Width(6) << max_time_rank;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Parallel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
