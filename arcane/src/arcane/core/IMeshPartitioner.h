// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshPartitioner.h                                          (C) 2000-2025 */
/*                                                                           */
/* Interface d'un partitionneur de maillage.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMESHPARTITIONER_H
#define ARCANE_IMESHPARTITIONER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IMeshPartitionerBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un partitionneur de maillage.
 *
 * Le partitionneur réaffecte les propriétaires des entités.
 * Il n'effectue pas directement l'échange d'entité.
 * Le partitionneur peut utiliser certaines informations comme
 * le timeRatio() ou imbalance() pour calculer un partionnement efficace.
 */
class ARCANE_CORE_EXPORT IMeshPartitioner
: public IMeshPartitionerBase
{
 public:

  virtual void build() = 0;

 public:

  using IMeshPartitionerBase::partitionMesh;

  virtual void partitionMesh(bool initial_partition, Int32 nb_part) = 0;

  //! Maillage associé au partitionneur
  ARCCORE_DEPRECATED_2021("Use primaryMesh() instead")
  virtual IMesh* mesh() const = 0;

  //! Maillage associé
  virtual IPrimaryMesh* primaryMesh() override;

 public:

  /*!{ \name compact
   *
   * Proportion du temps de calcul de ce sous-domaine par rapport à celui
   * du sous-domaine qui à le temps de calcul de plus élevé.
   */
  //! Positionne la proportion du temps de calcul
  //virtual void setTimeRatio(Real v) =0;
  //! Proportion du temps de calcul
  //virtual Real timeRatio() const =0;
  //@}

  //! Temps de calcul du sous-domaine le plus chargé
  virtual ARCANE_DEPRECATED_116 void setMaximumComputationTime(Real v) = 0;
  virtual ARCANE_DEPRECATED_116 Real maximumComputationTime() const = 0;

  /*! \brief Temps de calcul de se sous-domaine.
   * Le premier élément indique le temps de calcul du sous-domaine
   * correspondante aux calcul dont le cout est proportionnel aux mailles.
   * Les suivants doivent être associées à une variable (à faire).
   */
  virtual ARCANE_DEPRECATED_116 void setComputationTimes(RealConstArrayView v) = 0;
  virtual ARCANE_DEPRECATED_116 RealConstArrayView computationTimes() const = 0;

  /*!@{ \name imbalance
   *
   * Déséquilibre de temps calcul. Il est calculé comme suit
   * imbalance = (max_computation_time - min_computation_time) / min_computation_time;
   */
  //! Positionne le déséquilibre de temps de calcul
  virtual void setImbalance(Real v) = 0;
  //! Déséquilibre de temps de calcul
  virtual Real imbalance() const = 0;
  //@}

  //! Positionne le déséquilibre maximal autorisé
  virtual void setMaxImbalance(Real v) = 0;
  //! Déséquilibre maximal autorisé
  virtual Real maxImbalance() const = 0;

  //! Permet de définir les poids des objets à partitionner : on doit utiliser le ILoadBalanceMng maintenant.
  virtual ARCANE_DEPRECATED_116 void setCellsWeight(ArrayView<float> weights, Integer nb_weight) = 0;
  virtual ARCANE_DEPRECATED_116 ArrayView<float> cellsWeight() const = 0;

  //! Change le ILoadBalanceMng à utiliser.
  virtual void setILoadBalanceMng(ILoadBalanceMng* mng) = 0;
  virtual ILoadBalanceMng* loadBalanceMng() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
