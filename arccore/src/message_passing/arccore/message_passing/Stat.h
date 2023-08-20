// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Stat.h                                                      (C) 2000-2023 */
/*                                                                           */
/* Statistiques sur le parallélisme.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_STAT_H
#define ARCCORE_MESSAGEPASSING_STAT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/IStat.h"

#include "arccore/base/String.h"

#include <map>
#include <set>
#include <list>
#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Ensemble de données statistiques pour le profiling.
 *
 * Cette classe est interne à Arccore.
 */
class ARCCORE_MESSAGEPASSING_EXPORT StatData
{
  //! DEPRECATED
  using OneStatMap = std::map<String, OneStat*>;

  //! Collection de statistiques
  using StatCollection = std::list<OneStat>;

 public:

  StatData() = default;
  explicit StatData(const OneStatMap& os_map);

  const StatCollection& stats() const { return m_stat_col; }

  void resetCurrentStat();

  void mergeData(OneStat one_stat);
  void mergeAllData(const StatData& all_stat);
  // Surcharge temporaire avant de gerer le DEPRECATED OneStatMap
  void mergeAllData(const OneStatMap& all_stat);

 private:

  StatCollection m_stat_col;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gestionnaire de statistiques sur le parallélisme.
 *
 * Cette classe est interne à Arccore.
 */
class ARCCORE_MESSAGEPASSING_EXPORT Stat
: public IStat
{
 public:

  //! DEPRECATED
  using OneStatMap = std::map<String, OneStat*>;

 public:

  typedef std::pair<String, OneStat*> OneStatValue;

 public:

  //! Libère les ressources.
  ~Stat() override;

 public:

  void add(const String& name, double elapsed_time, Int64 msg_size) override;
  void enable(bool is_enabled) override { m_is_enabled = is_enabled; }

  void print(std::ostream& o);

  ARCCORE_DEPRECATED_2019("Please use getData() method instead")
  const OneStatMap& stats() const override { return m_list; }

  const StatData& getData() const { return m_data; }

  void resetCurrentStat() override;

 private:

  bool m_is_enabled = true;
  OneStatMap m_list;
  StatData m_data;

 private:

  OneStat* _find(const String& name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
