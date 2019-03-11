// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* Stat.cc                                                     (C) 2000-2018 */
/*                                                                           */
/* Statistiques sur le parallélisme.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/Stat.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

namespace MessagePassing
{

void OneStat::
print(std::ostream& o)
{
  Int64 div_time = static_cast<Int64>(m_total_time*1000.0);
  if (div_time==0)
    div_time = 1;
  if (m_nb_msg>0){
    Int64 average_time = (Int64)(m_total_time/(Real)m_nb_msg);
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

Stat::
Stat()
: m_is_enabled(true)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Stat::
~Stat()
{
  // TODO(FL): A enlever quand on aura supprimer m_list (gestion du DEPRECATED)
  for( auto i : m_list ){
    OneStat* os = i.second;
    delete os;
  }
  m_list.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Stat::
add(const String& name,double elapsed_time,Int64 msg_size)
{
  if (!m_is_enabled)
    return;
  // TODO(FL): A enlever quand on aura supprimer m_list (gestion du DEPRECATED)
  OneStat* os = _find(name);
  os->addMessage(msg_size,elapsed_time);

  m_data.mergeData(OneStat(name, msg_size, elapsed_time));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Stat::
print(std::ostream& o)
{
  // TODO(FL): A enlever quand on aura supprimer m_list (gestion du DEPRECATED)
  for( auto i : m_list ){
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

void StatData::
mergeData(OneStat one_stat)
{
  auto pos(std::find_if(m_stat_col.begin(), m_stat_col.end(),
                        [&one_stat](const OneStat& os){return (one_stat.name() == os.name());}));
  if (pos == m_stat_col.end())
    m_stat_col.emplace_back(one_stat);
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

} // End namespace MessagePassing

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
