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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Statistiques d'exécution d'une boucle.
 */
struct ARCANE_UTILS_EXPORT ForLoopProfilingStat
{
 public:

  //! Ajoute les infos de l'exécution \a s
  void add(const ForLoopOneExecInfo& s);

  Int64 nbCall() const { return m_nb_call; }
  Int64 nbChunk() const { return m_nb_chunk; }
  Int64 execTime() const { return m_exec_time; }

 private:

  Int64 m_nb_call = 0;
  Int64 m_nb_chunk = 0;
  Int64 m_exec_time = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct ARCANE_UTILS_EXPORT ScopedStatLoop
{
 public:

  explicit ScopedStatLoop(ForLoopOneExecInfo* s);
  ~ScopedStatLoop();

 public:

  double m_begin_time = 0.0;
  ForLoopOneExecInfo* m_stat_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO Utiliser un hash pour le map plutôt qu'une String pour accélérer les comparaisons

class ARCANE_UTILS_EXPORT StatInfoList
{
 public:

  void merge(const ForLoopOneExecInfo& loop_stat_info, const ForLoopTraceInfo& loop_trace_info);
  void print(std::ostream& o);

 private:

  std::map<String, impl::ForLoopProfilingStat> m_stat_map;
};

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour gérer le profiling d'une seule exécution d'une boucle.
 */
struct ARCANE_UTILS_EXPORT ForLoopOneExecInfo
{
 public:

  void incrementNbChunk() { ++m_nb_chunk; }
  void setExecTime(Int64 v) { m_exec_time = v; }

  Int64 nbChunk() const { return m_nb_chunk; }
  Int64 execTime() const { return m_exec_time; }

  void reset()
  {
    m_nb_chunk = 0;
    m_exec_time = 0;
  }

 private:

  //! Nombre de chunk de décomposition de la boucle (en multi-thread)
  std::atomic<Int64> m_nb_chunk = 0;
  // Temps total (en nano seconde)
  std::atomic<Int64> m_exec_time = 0;
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
  static impl::StatInfoList* threadLocalInstance();

  //! Affiche les statistiques d'exécution de toutes les instances sur \a o
  static void printExecutionStats(std::ostream& o);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
