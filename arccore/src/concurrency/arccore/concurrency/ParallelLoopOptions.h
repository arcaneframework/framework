// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelLoopOptions.h                                       (C) 2000-2025 */
/*                                                                           */
/* Options de configuration pour les boucles parallèles en multi-thread.     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_PARALLELLOOPOPTIONS_H
#define ARCCORE_CONCURRENCY_PARALLELLOOPOPTIONS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ConcurrencyGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Concurrency
 * \brief Options d'exécution d'une boucle parallèle en multi-thread.
 *
 * Cette classe permet de spécifier des paramètres d'exécution d'une
 * boucle parallèle.
 */
class ARCCORE_CONCURRENCY_EXPORT ParallelLoopOptions
{
 private:

  //! Drapeau pour indiquer quels champs ont été positionnés.
  enum SetFlags
  {
    SF_MaxThread = 1,
    SF_GrainSize = 2,
    SF_Partitioner = 4
  };

 public:

  //! Type du partitionneur
  enum class Partitioner
  {
    //! Laisse le partitionneur géré le partitionnement et l'ordonnancement (défaut)
    Auto = 0,
    /*!
     * \brief Utilise un partitionnement statique.
     *
     * Dans ce mode, grainSize() n'est pas utilisé et le partitionnement ne
     * dépend que du nombre de threads et de l'intervalle d'itération.
     *
     * A noter que l'ordonnencement reste dynamique et donc du exécution à
     * l'autre ce n'est pas forcément le même thread qui va exécuter
     * le même bloc d'itération.
     */
    Static = 1,
    /*!
     * \brief Utilise un partitionnement et un ordonnancement statique.
     *
     * Ce mode est similaire à Partitioner::Static mais l'ordonnancement
     * est déterministe pour l'attribution des tâches: la valeur
     * renvoyée par TaskFactory::currentTaskIndex() est déterministe.
     *
     * \note Actuellement ce mode de partitionnement n'est disponible que
     * pour la parallélisation des boucles 1D.
     */
    Deterministic = 2
  };

 public:

  ParallelLoopOptions()
  : m_grain_size(0)
  , m_max_thread(-1)
  , m_partitioner(Partitioner::Auto)
  , m_flags(0)
  {}

 public:

  //! Nombre maximal de threads autorisés.
  Int32 maxThread() const { return m_max_thread; }
  /*!
   * \brief Positionne le nombre maximal de threads autorisé.
   *
   * Si \a v vaut 0 ou 1, l'exécution sera séquentielle.
   * Si \a v est supérieur à TaskFactory::nbAllowedThread(), c'est
   * cette dernière valeur qui sera utilisée.
   */
  void setMaxThread(Integer v)
  {
    m_max_thread = v;
    m_flags |= SF_MaxThread;
  }
  //! Indique si maxThread() est positionné
  bool hasMaxThread() const { return m_flags & SF_MaxThread; }

  //! Taille d'un intervalle d'itération.
  Integer grainSize() const { return m_grain_size; }
  //! Positionne la taille (approximative) d'un intervalle d'itération
  void setGrainSize(Integer v)
  {
    m_grain_size = v;
    m_flags |= SF_GrainSize;
  }
  //! Indique si grainSize() est positionné
  bool hasGrainSize() const { return m_flags & SF_GrainSize; }

  //! Type du partitionneur
  Partitioner partitioner() const { return m_partitioner; }
  //! Positionne le type du partitionneur
  void setPartitioner(Partitioner v)
  {
    m_partitioner = v;
    m_flags |= SF_Partitioner;
  }
  //! Indique si grainSize() est positionné
  bool hasPartitioner() const { return m_flags & SF_Partitioner; }

 public:

  //! Fusionne les valeurs non modifiées de l'instance par celles de \a po.
  void mergeUnsetValues(const ParallelLoopOptions& po)
  {
    if (!hasMaxThread())
      setMaxThread(po.maxThread());
    if (!hasGrainSize())
      setGrainSize(po.grainSize());
    if (!hasPartitioner())
      setPartitioner(po.partitioner());
  }

 private:

  //! Taille d'un bloc de la boucle
  Int32 m_grain_size = 0;
  //!< Nombre maximum de threads pour la boucle
  Int32 m_max_thread = -1;
  //!< Type de partitionneur.
  Partitioner m_partitioner = Partitioner::Auto;

  unsigned int m_flags = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
