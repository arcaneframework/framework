// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Stat.cc                                                     (C) 2000-2023 */
/*                                                                           */
/* Statistiques sur le parallélisme.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/parallel/IStat.h"

#include "arcane/utils/String.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/JSONWriter.h"
#include "arcane/utils/Convert.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/Properties.h"

#include "arccore/message_passing/Stat.h"

#include <cmath>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Parallel
{

namespace  MP = Arccore::MessagePassing;
using MP::OneStat;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Statistiques sur le parallélisme.
 */
class Stat
: public MP::Stat
, public IStat
{
 public:

  using Base = MP::Stat;

  class CumulativeStat
  {
   public:

    String m_name;
    Int64 m_nb_message = 0;
    Int64 m_total_size = 0;
    Real m_total_time = 0.0;
  };

  using CumulativeStatMap = std::map<String,CumulativeStat>;

  //! Infos de sérialisation
  class SerializedStats
  {
   public:

    void save(const CumulativeStatMap& stat_map)
    {
      for (auto& i : stat_map){
        const CumulativeStat& s = i.second;
        m_total_time_list.add(s.m_total_time);
        m_nb_message_list.add(s.m_nb_message);
        m_total_size_list.add(s.m_total_size);
        m_name_list.add(s.m_name);
      }
    }

    void read(CumulativeStatMap& stat_map)
    {
      Int32 n = m_name_list.size();
      for (Int32 i = 0; i < n; ++i) {
        const String& name = m_name_list[i];
        CumulativeStat& cs = stat_map[name];
        cs.m_name = name;
        cs.m_nb_message += m_nb_message_list[i];
        cs.m_total_size += m_total_size_list[i];
        cs.m_total_time += m_total_time_list[i];
      }
    }

   public:

    UniqueArray<String> m_name_list;
    UniqueArray<Int64> m_nb_message_list;
    UniqueArray<Int64> m_total_size_list;
    UniqueArray<Real> m_total_time_list;
  };

 public:

  MP::IStat* toArccoreStat() override { return this; }

  void add(const String& name, double elapsed_time, Int64 msg_size) override;
  void print(ITraceMng* msg) override;
  void enable(bool is_enabled) override { Base::enable(is_enabled); }
  void dumpJSON(JSONWriter& writer) override;
  void saveValues(ITraceMng* tm, Properties* p) override;
  void mergeValues(ITraceMng* tm, Properties* p) override;

 private:

  CumulativeStatMap m_previous_stat_map;

 private:

  // Fusionne les valeurs de l'instance avec celles contenues dans l'instance
  void _mergeStats(CumulativeStatMap& stat_map)
  {
    for (const OneStat& s : statList()) {
      CumulativeStat& cs = stat_map[s.name()];
      cs.m_name = s.name();
      cs.m_nb_message += s.cumulativeNbMessage();
      cs.m_total_size += s.cumulativeTotalSize();
      cs.m_total_time += s.cumulativeTotalTime();
    }
  }
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
  for (const auto& os : statList()) {
    Real total_time = os.cumulativeTotalTime();
    Int64 div_time = static_cast<Int64>(total_time * 1000.0);
    Int64 nb_message = os.cumulativeNbMessage();
    Int64 total_size = os.cumulativeTotalSize();
    const String& name = os.name();
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
  for (const OneStat& s : statList())
    Parallel::dumpJSON(writer, s); // cumulative stats dump
  writer.endArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Stat::
saveValues(ITraceMng* tm, Properties* p)
{
  tm->info(4) << "Saving IParallelMng Stat values";

  CumulativeStatMap current_stat_map(m_previous_stat_map);

  SerializedStats save_info;
  _mergeStats(current_stat_map);

  save_info.save(current_stat_map);

  p->set("Version", 1);
  p->set("NameList", save_info.m_name_list);
  p->set("NbMessageList", save_info.m_nb_message_list);
  p->set("TotalSizeList", save_info.m_total_size_list);
  p->set("TotalTimeList", save_info.m_total_time_list);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Stat::
mergeValues(ITraceMng* tm, Properties* p)
{
  tm->info(4) << "Merging IParallelMng Stat values";

  SerializedStats save_info;

  Int32 v = p->getInt32WithDefault("Version", 0);
  // Ne fait rien si aucune info dans la protection
  if (v == 0)
    return;
  if (v != 1) {
    tm->info() << "Warning: can not merge IParallelMng stats values because checkpoint version is not compatible";
    return;
  }

  p->get("NameList", save_info.m_name_list);
  p->get("NbMessageList", save_info.m_nb_message_list);
  p->get("TotalSizeList", save_info.m_total_size_list);
  p->get("TotalTimeList", save_info.m_total_time_list);

  save_info.read(m_previous_stat_map);
  _mergeStats(m_previous_stat_map);
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
             << Trace::Width(7) << "rank"
             << Trace::Width(7) << "nb";
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
               << " " << Trace::Width(6) << max_time_rank
               << " " << Trace::Width(6) << i->second->nbMessage();

  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Parallel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
