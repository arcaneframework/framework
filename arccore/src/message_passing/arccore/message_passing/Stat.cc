// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Stat.cc                                                     (C) 2000-2025 */
/*                                                                           */
/* Statistiques sur le parallélisme.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/Stat.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

OneStat::
OneStat(const String& name, Int64 msg_size, double elapsed_time)
: m_name(name)
, m_total_size(msg_size)
, m_total_time(elapsed_time)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneStat::
addMessage(Int64 msg_size, double elapsed_time)
{
  ++m_nb_msg;
  m_total_size += msg_size;
  m_total_time += elapsed_time;
  ++m_cumulative_nb_msg;
  m_cumulative_total_size += msg_size;
  m_cumulative_total_time += elapsed_time;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneStat::
resetCurrentStat()
{
  m_nb_msg = 0;
  m_total_size = 0;
  m_total_time = 0.0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneStat::
print(std::ostream& o)
{
  Int64 div_time = static_cast<Int64>(m_total_time * 1000.0);
  if (div_time == 0)
    div_time = 1;
  if (m_nb_msg > 0) {
    Int64 average_time = (Int64)(m_total_time / (Real)m_nb_msg);
    o << " MPIStat " << m_name << "     :" << m_nb_msg << " messages" << '\n';
    o << " MPIStat " << m_name << "     :" << m_total_size << " bytes ("
      << m_total_size / div_time << " Kb/s) (average size "
      << m_total_size / m_nb_msg << " bytes)" << '\n';
    o << " MPIStat " << m_name << " Time: " << m_total_time << " seconds"
      << " (avg=" << average_time << ")" << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Stat::
~Stat()
{
  // TODO(FL): A enlever quand on aura supprimer m_list (gestion du DEPRECATED)
  for (const auto& i : m_list) {
    OneStat* os = i.second;
    delete os;
  }
  m_list.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Stat::
add(const String& name, double elapsed_time, Int64 msg_size)
{
  if (!m_is_enabled)
    return;
  // TODO(FL): A enlever quand on aura supprimer m_list (gestion du DEPRECATED)
  OneStat* os = _find(name);
  os->addMessage(msg_size, elapsed_time);

  m_data.mergeData(OneStat(name, msg_size, elapsed_time));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Stat::
print(std::ostream& o)
{
  // TODO(FL): A enlever quand on aura supprimer m_list (gestion du DEPRECATED)
  for (const auto& i : m_list) {
    OneStat* os = i.second;
    os->print(o);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Stat::
resetCurrentStat()
{
  // TODO(FL): A enlever quand on aura supprimer m_list (gestion du DEPRECATED)
  for (auto& i : m_list)
    i.second->resetCurrentStat();

  m_data.resetCurrentStat();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

OneStat* Stat::
_find(const String& name)
{
  auto i = m_list.find(name);
  if (i != m_list.end())
    return i->second;

  OneStat* os = new OneStat(name);
  // Important: utiliser os.m_name car m_list stocke juste un
  // pointeur sur la chaîne de caractère.
  m_list.insert(OneStatMap::value_type(os->name(), os));
  return os;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StatData::
StatData(const OneStatMap& os_map)
{
  for (const auto& i : os_map)
    m_stat_col.m_stats.emplace_back(*(i.second));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StatData::
resetCurrentStat()
{
  for (auto& i : m_stat_col.m_stats)
    i.resetCurrentStat();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StatData::
mergeData(OneStat one_stat)
{
  auto pos(std::find_if(m_stat_col.m_stats.begin(), m_stat_col.m_stats.end(),
                        [&one_stat](const OneStat& os) { return (one_stat.name() == os.name()); }));
  if (pos == m_stat_col.end())
    m_stat_col.m_stats.emplace_back(one_stat);
  else
    pos->addMessage(one_stat.totalSize(), one_stat.totalTime());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StatData::
mergeAllData(const StatData& all_stat)
{
  for (const auto& stat : all_stat.m_stat_col)
    mergeData(stat);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Surcharge temporaire avant de gerer le DEPRECATED OneStatMap
void StatData::
mergeAllData(const OneStatMap& all_stat)
{
  for (const auto& stat : all_stat)
    mergeData(*(stat.second));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const StatCollection& Stat::
statList() const
{
  return m_data.stats();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
