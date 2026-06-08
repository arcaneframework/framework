// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IStat.h                                                     (C) 2000-2025 */
/*                                                                           */
/* Statistics on parallelism.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_ISTAT_H
#define ARCCORE_MESSAGEPASSING_ISTAT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"

#include "arccore/base/String.h"

#include <iosfwd>
#include <map>
#include <list>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Statistic on a message.
 */
class ARCCORE_MESSAGEPASSING_EXPORT OneStat
{
 public:

  explicit OneStat(const String& name)
  : m_name(name)
  {}
  OneStat(const String& name, Int64 msg_size, double elapsed_time);

 public:

  //! Name of the statistic
  const String& name() const { return m_name; }

  //! Number of messages sent.
  Int64 nbMessage() const { return m_nb_msg; }
  void setNbMessage(Int64 v) { m_nb_msg = v; }

  //! Number of messages sent throughout the execution time
  Int64 cumulativeNbMessage() const { return m_cumulative_nb_msg; }
  void setCumulativeNbMessage(Int64 v) { m_cumulative_nb_msg = v; }

  //! Total size of messages sent
  Int64 totalSize() const { return m_total_size; }
  void setTotalSize(Int64 v) { m_total_size = v; }

  //! Total size of messages sent throughout the execution time
  Int64 cumulativeTotalSize() const { return m_cumulative_total_size; }
  void setCumulativeTotalSize(Int64 v) { m_cumulative_total_size = v; }

  //! Total elapsed time
  double totalTime() const { return m_total_time; }
  void setTotalTime(double v) { m_total_time = v; }

  //! Total elapsed time throughout the program execution
  double cumulativeTotalTime() const { return m_cumulative_total_time; }
  void setCumulativeTotalTime(double v) { m_cumulative_total_time = v; }

 public:

  //! Prints the instance information to \a o
  void print(std::ostream& o);

  //! Adds a message
  void addMessage(Int64 msg_size, double elapsed_time);

  //! Resets current statistics (non-cumulative)
  void resetCurrentStat();

 private:

  String m_name;
  Int64 m_nb_msg = 0;
  Int64 m_total_size = 0;
  double m_total_time = 0.0;
  Int64 m_cumulative_nb_msg = 0;
  Int64 m_cumulative_total_size = 0;
  double m_cumulative_total_time = 0.0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief List of statistics.
 *
 * It is only possible to iterate over the elements of the collection.
 *
 * The implementation used may evolve, so the explicit iterator type should
 * not be used.
 */
class ARCCORE_MESSAGEPASSING_EXPORT StatCollection
{
  friend class StatData;
  using Impl = std::list<OneStat>;

 public:

  using const_iterator = Impl::const_iterator;

 public:

  const_iterator begin() const { return m_stats.begin(); }
  const_iterator end() const { return m_stats.end(); }
  const_iterator cbegin() const { return m_stats.begin(); }
  const_iterator cend() const { return m_stats.end(); }

 private:

  Impl m_stats;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Statistics on parallelism.
 * \todo make thread-safe
 */
class ARCCORE_MESSAGEPASSING_EXPORT IStat
{
 public:

  // DEPRECATED
  using OneStatMap = std::map<String, OneStat*>;

 public:

  //! Frees resources.
  virtual ~IStat() = default;

 public:

  /*!
   * \brief Adds a statistic.
   *
   * \param name name of the statistic
   * \param elapsed_time time used for the message
   * \param msg_size size of the message sent.
   */
  virtual void add(const String& name, double elapsed_time, Int64 msg_size) = 0;

  /*!
   * \brief Enables or disables statistics.
   *
   * If statistics are disabled, the call to add() has no effect.
   */
  virtual void enable(bool is_enabled) = 0;

  //! Retrieval of statistics
  virtual const StatCollection& statList() const = 0;

  //! Resets current statistics
  virtual void resetCurrentStat() = 0;

 public:

  //! Retrieval of statistics
  ARCCORE_DEPRECATED_REASON("Y2023: Use statList() instead")
  virtual const OneStatMap& stats() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
