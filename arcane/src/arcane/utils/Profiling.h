// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Profiling.h                                                 (C) 2000-2022 */
/*                                                                           */
/* Classes pour gérer le profilage.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_PROFILING_H
#define ARCANE_UTILS_PROFILING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include "arcane/utils/String.h"

#include <map>
#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

struct ARCANE_UTILS_EXPORT LoopStatInfo
{
 public:

  void reset()
  {
    m_nb_loop_parallel_for = 0;
    m_nb_chunk_parallel_for = 0;
  }

  void incrementNbChunkParallelFor()
  {
    ++m_nb_chunk_parallel_for;
  }

  void incrementNbLoopParallelFor()
  {
    ++m_nb_loop_parallel_for;
  }

  void merge(const LoopStatInfo* s)
  {
    if (s)
      merge(*s);
  }

  void merge(const LoopStatInfo& s)
  {
    m_nb_loop_parallel_for += s.m_nb_loop_parallel_for;
    m_nb_chunk_parallel_for += s.m_nb_chunk_parallel_for;
    m_total_time += s.m_total_time;
  }

  void printInfos(std::ostream& o);

 public:

  std::atomic<Int64> m_nb_loop_parallel_for = 0;
  std::atomic<Int64> m_nb_chunk_parallel_for = 0;
  // Temps total (en nano seconde)
  std::atomic<Int64> m_total_time = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct ARCANE_UTILS_EXPORT ScopedStatLoop
{
 public:

  explicit ScopedStatLoop(LoopStatInfo* s);
  ~ScopedStatLoop();

 public:

  double m_begin_time = 0.0;
  LoopStatInfo* m_stat_info = nullptr;
};

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO Utiliser un hash pour le map plutôt qu'une String pour accélérer les comparaisons

class ARCANE_UTILS_EXPORT StatInfoList
{
 public:

  void merge(const impl::LoopStatInfo& loop_stat_info, const ForLoopTraceInfo& loop_trace_info);
  void print(std::ostream& o);

 private:

  std::map<String, impl::LoopStatInfo> m_stat_map;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire pour le profiling.
 */
class ARCANE_UTILS_EXPORT ProfilingRegistry
{
 public:

  //! Instance locale part thread du gestionnaire des statistiques
  static StatInfoList* threadLocalInstance();

  //! Affiche les statistiques d'exécution de toutes les instances sur \a o
  static void printExecutionStats(std::ostream& o);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

