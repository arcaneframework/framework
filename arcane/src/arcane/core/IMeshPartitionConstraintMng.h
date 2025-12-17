// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshPartitionConstraintMng.h                               (C) 2000-2025 */
/*                                                                           */
/* Interface d'un gestionnaire de contraintes de partitionnement de maillage.*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHPARTITIONCONSTRAINTMNG_H
#define ARCANE_CORE_IMESHPARTITIONCONSTRAINTMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un gestionnaire de contraintes
 * de partitionnement de maillage.
 */
class IMeshPartitionConstraintMng
{
 public:

  virtual ~IMeshPartitionConstraintMng() = default; //!< Libère les ressources

 public:

  //! Ajoute une contrainte
  virtual void addConstraint(IMeshPartitionConstraint* constraint) = 0;

  /*!
   * \brief Supprime une contrainte.
   *
   * L'appelant devient propriétaire de \a constraint et doit
   * gérer sa destruction.
   */
  virtual void removeConstraint(IMeshPartitionConstraint* constraint) = 0;

  /*!
   * \brief Calcule et applique des contraintes.
   *
   * Calcule les contraintes sur le maillage en appliquant
   * pour chaque contrainte la méthode IMeshPartitionConstraint::addLinkedCell()
   * et les applique en modifiant la variable IItemFamily::itemsNewOwner()
   * des mailles. Toutes les mailles qui doivent être liées en elle sont
   * alors mises dans une même partition.
   *
   * Cette opération ne garantit pas que les partitions résultantes soient
   * équilibrées au niveau de la charge. Pour cela,
   * il faut utiliser un service de re-partionnement (IMeshPartitioner)
   * qui prennent en compte ces contraintes.
   *
   * Cette opération est collective.
   */
  virtual void computeAndApplyConstraints() = 0;

  /*!
   * \brief Calcule les contraintes et retourne une liste d'entités liées.
   *
   * Calcule les contraintes comme pour computeAndApplyConstraints()
   * mais ne modifie pas le propriétaire. A la place, retourne une
   * liste contenant les listes des uniqueId() des entités qui doivent
   * être liées.
   *
   * Cette opération est collective.
   */
  virtual void computeConstraintList(Int64MultiArray2& tied_uids) = 0;

  //! Ajoute une contrainte
  virtual void addWeakConstraint(IMeshPartitionConstraint* constraint) = 0;

  /*!
   * \brief Supprime une contrainte.
   *
   * L'appelant devient propriétaire de \a constraint et doit
   * gérer sa destruction.
   */
  virtual void removeWeakConstraint(IMeshPartitionConstraint* constraint) = 0;

  /*!
   * \brief Calcule et applique des contraintes.
   *
   * Calcule les contraintes sur le maillage en appliquant
   * pour chaque contrainte la méthode IMeshPartitionConstraint::addLinkedCell()
   * et les applique en modifiant la variable IItemFamily::itemsNewOwner()
   * des mailles. Toutes les mailles qui doivent être liées en elle sont
   * alors mises dans une même partition.
   *
   * Cette opération ne garantit pas que les partitions résultantes soient
   * équilibrées au niveau de la charge. Pour cela,
   * il faut utiliser un service de re-partionnement (IMeshPartitioner)
   * qui prennent en compte ces contraintes.
   *
   * Cette opération est collective.
   */
  virtual void computeAndApplyWeakConstraints() = 0;

  /*!
   * \brief Calcule les contraintes et retourne une liste d'entités liées.
   *
   * Calcule les contraintes comme pour computeAndApplyConstraints()
   * mais ne modifie pas le propriétaire. A la place, retourne une
   * liste contenant les listes des uniqueId() des entités qui doivent
   * être liées.
   *
   * Cette opération est collective.
   */
  virtual void computeWeakConstraintList(Int64MultiArray2& tied_uids) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
