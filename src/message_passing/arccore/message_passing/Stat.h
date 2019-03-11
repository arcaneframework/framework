// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* Stat.h                                                      (C) 2000-2019 */
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

namespace Arccore
{

namespace MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Un ensemble de donnees statistiques pour le profiling
class ARCCORE_MESSAGEPASSING_EXPORT StatData
{
  //! DEPRECATED
  using OneStatMap = std::map<String,OneStat*>;
  //! Collection de statisques
  using StatCollection = std::list<OneStat>;
 public:
  StatData() = default;
  StatData(const StatData&) = default;
  explicit StatData(StatData&&) = default;
  explicit StatData(const OneStatMap& os_map)
  {
    for (const auto& i : os_map)
      m_stat_col.emplace_back(*(i.second));
  }
  ~StatData() = default;

  const StatCollection& stats() const { return m_stat_col; }

  void resetCurrentStat()
  {
    for (auto& i : m_stat_col)
      i.resetCurrentStat();
  }

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
 * \brief Statistiques sur le parallélisme.
 */
class ARCCORE_MESSAGEPASSING_EXPORT Stat
: public IStat
{
 public:

  //! DEPRECATED
  using OneStatMap = std::map<String,OneStat*>;

 public:

  typedef std::pair<String,OneStat*> OneStatValue;

 public:

  Stat();
  //! Libère les ressources.
  virtual ~Stat();

 public:

 public:

  void add(const String& name,double elapsed_time,Int64 msg_size) override;
  void enable(bool is_enabled) override { m_is_enabled = is_enabled; }

  void print(std::ostream& o);

  ARCCORE_DEPRECATED_2019("Please use getData() method instead")
  const OneStatMap& stats() const { return m_list; }

  const StatData& getData() const { return m_data; }

  void resetCurrentStat() override;

 private:

  bool m_is_enabled;
  OneStatMap m_list;
  StatData m_data;

 private:

  OneStat* _find(const String& name)
  {
    OneStatMap::const_iterator i = m_list.find(name);
    if (i!=m_list.end()){
      return i->second;
    }
    OneStat* os = new OneStat(name);
    // Important: utiliser os.m_name car m_list stocke juste un
    // pointeur sur la chaîne de caractère.
    m_list.insert(OneStatMap::value_type(os->name(),os));
    return os;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace MessagePassing

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
