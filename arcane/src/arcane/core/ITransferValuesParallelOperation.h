// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITransferValuesParallelOperation.h                          (C) 2000-2025 */
/*                                                                           */
/* Transfert de valeurs sur différents processeurs.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITRANSFERVALUESPARALLELOPERATION_H
#define ARCANE_CORE_ITRANSFERVALUESPARALLELOPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Envoie de valeurs sur différents processeurs.
 *
 * Cette opération permet de communiquer des valeurs avec les autres
 * processeurs. Le tableau \a ranks indique pour chaque élément le rang
 * du processeur auquel il est destiné. Il est ensuite possible de spécifier
 * des tableaux contenant les valeurs à envoyer et à recevoir. Les
 * tableaux d'envois doivent avoir le même nombre d'élément que \a ranks
 *
 * Une instance ne sert qu'une fois. Une fois le transfert terminé,
 * elle peut être détruite.
 *
 * Par exemple, pour un cas à 3 processeurs:
 * \code
 * // Processeur de rang 0:
 * Int32UniqueArray ranks;
 * ranks.add(2); // Envoie au rang 2
 * ranks.add(1); // Envoie au rang 1
 * ranks.add(1); // Envoie au rang 1
 * Int32UniqueArray values_1;
 * values_1.add(5); // Envoie 5 au rang 2 (ranks[0])
 * values_1.add(7); // Envoie 7 au rang 1 (ranks[1])
 * values_1.add(6); // Envoie 6 au rang 1 (ranks[2])
 * Int64UniqueArray values_2;
 * values_2.add(-5); // Envoie -5 au rang 2 (ranks[0])
 * values_2.add(-7); // Envoie -7 au rang 1 (ranks[1])
 * values_2.add(-6); // Envoie -6 au rang 1 (ranks[2])

 * // Processeur de rang 1:
 * Int32UniqueArray ranks;
 * ranks.add(0); // Envoie au rang 0
 * ranks.add(2); // Envoie au rang 2
 * Int32UniqueArray values_1;
 * values_1.add(1); // Envoie 1 au rang 0 (ranks[0])
 * values_1.add(3); // Envoie 3 au rang 2 (ranks[1])
 * Int64UniqueArray values_2;
 * values_2.add(23); // Envoie 23 au rang 0 (ranks[0])
 * values_2.add(24); // Envoie 24 au rang 2 (ranks[1])

 * // Processeur de rang 2:
 * Int32UniqueArray ranks;
 * ranks.add(0); // Envoie au rang 0
 * ranks.add(0); // Envoie au rang 0
 * Int32UniqueArray values_1;
 * values_1.add(0); // Envoie 1 au rang 0 (ranks[0])
 * values_1.add(4); // Envoie 3 au rang 0 (ranks[1])
 * Int64UniqueArray values_2;
 * values_2.add(-1); // Envoie -1 au rang 0 (ranks[0])
 * values_2.add(4); // Envoie 4 au rang 0 (ranks[1])

 * \endcode
 *
 * Pour effectuer le transfert
 *
 * \code
 * Int32UniqueArray recv_values_1;
 * Int64UniqueArray recv_values_2;
 * op->setTransferRanks(ranks);
 * op->addArray(values_1,recv_values_1);
 * op->addArray(values_2,recv_values_2);
 * op->transferValues();
 * \endcode
 *
 * Après envoie la processeur de rang 0 aura les valeurs suivantes:
 * \code
 * recv_values_1[0] == 1; // envoyé par le rang 1
 * recv_values_1[1] == 0; // envoyé par le rang 2
 * recv_values_1[2] == 4; // envoyé par le rang 2
 * recv_values_2[0] == 23; // envoyé par le rang 1
 * recv_values_2[1] == -1; // envoyé par le rang 2
 * recv_values_2[2] == 4; // envoyé par le rang 2
 * \endcode
 *
 * A noter que l'ordre des éléments est indéterminé
 */
class ARCANE_CORE_EXPORT ITransferValuesParallelOperation
{
 public:

  //! Destructeur
  virtual ~ITransferValuesParallelOperation() = default;

 public:

  //! Gestionnaire de parallélisme associé
  virtual IParallelMng* parallelMng() = 0;

 public:

  //! Positionne le tableau indiquant à qui envoyer les valeurs.
  virtual void setTransferRanks(Int32ConstArrayView ranks) = 0;
  //! Ajoute un tableau de \c Int32
  virtual void addArray(Int32ConstArrayView send_values, SharedArray<Int32> recv_value) = 0;
  //! Ajoute un tableau de \c Int64
  virtual void addArray(Int64ConstArrayView send_values, SharedArray<Int64> recv_values) = 0;
  //! Ajoute un tableau de \c Int64
  virtual void addArray(RealConstArrayView send_values, SharedArray<Real> recv_values) = 0;
  /*!
   * \brief Envoie et réceptionne les valeurs.
   *
   * Cet appel est collectif et bloquant.
   */
  virtual void transferValues() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
