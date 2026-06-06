// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IStat.h                                                     (C) 2000-2025 */
/*                                                                           */
/* Statistics on messages of 'IParallelMng'.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_PARALLEL_ISTAT_H
#define ARCANE_CORE_PARALLEL_ISTAT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/JSONWriter.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Parallel.h"

#include "arccore/message_passing/IStat.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class Properties;
}

namespace Arcane::Parallel
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Statistics on parallelism.
 * \todo make thread-safe
 */
class IStat
{
 public:

  //! Frees resources.
  virtual ~IStat() {}

 public:

  /*!
   * \brief Adds a statistic.
   *
   * \param name name of the statistic.
   * \param elapsed_time time used for the message.
   * \param msg_size size of the sent message.
   */
  virtual void add(const String& name,double elapsed_time,Int64 msg_size) =0;
  
  //! Prints the statistics to \a trace.
  virtual void print(ITraceMng* trace) =0;

  /*!
   * \brief Displays the statistics collectively.
   *
   * Displays statistics common to all ranks associated with \a pm.
   *
   * This operation is collective.
   */
  virtual void printCollective(IParallelMng* pm) = 0;

  //! Enables or disables the statistics
  virtual void enable(bool is_enabled) =0;

  //! Outputs the statistics in JSON format
  virtual void dumpJSON(JSONWriter& writer) =0;

 public:

  //! Saves the current values into \a p
  virtual void saveValues(ITraceMng* tm, Properties* p) =0;

  //! Merges the current values with those saved in \a p
  virtual void mergeValues(ITraceMng* tm, Properties* p) =0;

 public:

  virtual Arccore::MessagePassing::IStat* toArccoreStat() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Creates a default instance
extern "C++" ARCANE_CORE_EXPORT IStat*
createDefaultStat();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Free function for dumping a message stat into JSON
extern "C++" ARCANE_CORE_EXPORT void
dumpJSON(JSONWriter& writer, const Arccore::MessagePassing::OneStat& os, bool cumulative_stat = true);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Displays the cumulative statistics across all ranks of \a pm.
 */
extern "C++" ARCANE_DEPRECATED_REASON("Y2023: Use IStat::printCollective() instead")
ARCANE_CORE_EXPORT void printStatsCollective(IStat* s, IParallelMng* pm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Parallel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
