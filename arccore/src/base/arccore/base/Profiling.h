// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Profiling.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Classes pour gérer le profilage.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_PROFILING_H
#define ARCCORE_BASE_PROFILING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

#include <atomic>
#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Impl
{
class AcceleratorStatInfoList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe permettant de récupérer le temps passé entre l'appel au
 * constructeur et au destructeur.
 */
class ARCCORE_BASE_EXPORT ScopedStatLoop
{
 public:

  explicit ScopedStatLoop(ForLoopOneExecStat* s);
  ~ScopedStatLoop();

 public:

  Int64 m_begin_time = 0.0;
  ForLoopOneExecStat* m_stat_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Statistiques d'exécution des boucles.
 */
class ARCCORE_BASE_EXPORT ForLoopStatInfoList
{
 public:

  ForLoopStatInfoList();
  ~ForLoopStatInfoList();

 public:

  void merge(const ForLoopOneExecStat& loop_stat_info, const ForLoopTraceInfo& loop_trace_info);

 public:

  /*!
   * \internal
   * \brief Type opaque pour l'implémentation interne.
   */
  ForLoopStatInfoListImpl* _internalImpl() const { return m_p; }

 private:

  ForLoopStatInfoListImpl* m_p = nullptr;
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
class ARCCORE_BASE_EXPORT ForLoopOneExecStat
{
 public:

  /*!
   * \brief Incrémente le nombre de chunk utilisé.
   *
   * Cette méthode peut être appelée simultanément par plusieurs threads.
   */
  void incrementNbChunk() { ++m_nb_chunk; }

  //! Positionne le temps de début de la boucle (en nanoseconde)
  void setBeginTime(Int64 v) { m_begin_time = v; }

  //! Positionne le temps de fin de la boucle en nanoseconde
  void setEndTime(Int64 v) { m_end_time = v; }

  //! Nombre de chunks
  Int64 nbChunk() const { return m_nb_chunk; }

  /*!
   * \brief Temps d'exécution (en nanoseconde).
   *
   * La valeur retournée n'est valide que si setBeginTime() et setEndTime()
   * ont été appelés avant.
   */
  Int64 execTime() const { return m_end_time - m_begin_time; }

  void reset()
  {
    m_nb_chunk = 0;
    m_begin_time = 0;
    m_end_time = 0;
  }

 private:

  //! Nombre de chunk de décomposition de la boucle (en multi-thread)
  std::atomic<Int64> m_nb_chunk = 0;

  // Temps de début d'exécution
  Int64 m_begin_time = 0;

  // Temps de fin d'exécution
  Int64 m_end_time = 0;
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
 * impl::ForLoopStatInfoList spécifique au thread en cours d'exécution.
 */
class ARCCORE_BASE_EXPORT ProfilingRegistry
{
 public:

  /*!
   * TODO: rendre obsolète. Utiliser à la place:
   * static impl::ForLoopStatInfoList* _threadLocalForLoopInstance();
   */
  ARCCORE_DEPRECATED_REASON("Y2023: Use _threadLocalForLoopInstance() instead")
  static Impl::ForLoopStatInfoList* threadLocalInstance();

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

  /*!
   * \brief Visite la liste des statistiques des boucles
   *
   * Il y a une instance de impl::ForLoopStatInfoList par thread qui a
   * exécuté une boucle.
   *
   * Cette méthode ne doit pas être appelée s'il y a des boucles en cours d'exécution.
   */
  static void visitLoopStat(const std::function<void(const Impl::ForLoopStatInfoList&)>& f);

  /*!
   * \brief Visite la liste des statistiques sur accélérateur
   *
   * Il y a une instance de impl::AcceleratorStatInfoList par thread qui a
   * exécuté une boucle.
   *
   * Cette méthode ne doit pas être appelée lorsque le profiling est actif.
   */
  static void visitAcceleratorStat(const std::function<void(const Impl::AcceleratorStatInfoList&)>& f);

  static const Impl::ForLoopCumulativeStat& globalLoopStat();

 public:

  // API publique mais réservée à Arcane.

  /*!
   * \internal.
   * Instance locale par thread du gestionnaire des statistiques de boucle
   */
  static Impl::ForLoopStatInfoList* _threadLocalForLoopInstance();

  /*!
   * \internal.
   * Instance locale par thread du gestionnaire des statistiques pour accélérateur
   */
  static Impl::AcceleratorStatInfoList* _threadLocalAcceleratorInstance();

 private:

  static Int32 m_profiling_level;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
