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
  void add(const ForLoopOneExecStat& s);

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

  explicit ScopedStatLoop(ForLoopOneExecStat* s);
  ~ScopedStatLoop();

 public:

  double m_begin_time = 0.0;
  ForLoopOneExecStat* m_stat_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO Utiliser un hash pour le map plutôt qu'une String pour accélérer les comparaisons

class ARCANE_UTILS_EXPORT StatInfoList
{
 public:

  void merge(const ForLoopOneExecStat& loop_stat_info, const ForLoopTraceInfo& loop_trace_info);
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
class ARCANE_UTILS_EXPORT ForLoopOneExecStat
{
 public:

  /*!
   * \brief Incrémente le nombre de chunk utilisé.
   *
   * Cette méthode peut être appelée simultanément par plusieurs threads.
   */
  void incrementNbChunk() { ++m_nb_chunk; }

  //! Positionne le temps d'exécution de la boucle en nanoseconde
  void setExecTime(Int64 v) { m_exec_time = v; }

  //! Nombre de chunks
  Int64 nbChunk() const { return m_nb_chunk; }

  //! Temps d'exécution
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
  Int64 m_exec_time = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire pour le profiling.
 *
 * Il est possible d'activer le profilage en appelant setProfilingLevel() avec
 * une valeur supérieur ou égale à 1.
 *
 * L'ajout de statistiques se fait en récupérant une instance de
 * impl::StatInfoList spécifique au thread en cours d'exécution.
 */
class ARCANE_UTILS_EXPORT ProfilingRegistry
{
 public:

  /*!
   * \internal.
   * Instance locale par thread du gestionnaire des statistiques
   */
  static impl::StatInfoList* threadLocalInstance();

  //! Affiche les statistiques d'exécution de toutes les instances sur \a o
  static void printExecutionStats(std::ostream& o);

  /*!
   * \brief Positionne le niveau de profilage.
   *
   * Si 0, alors il n'y a pas de profilage. Le profilage est actif à partir
   * du niveau 1.
   */
  static void setProfilingLevel(Int32 level);

  //! Niveau de profilage
  static Int32 profilingLevel() { return m_profiling_level; }

  //! Indique si le profilage est actif.
  static bool hasProfiling() { return m_profiling_level > 0; }

 private:

  static Int32 m_profiling_level;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
