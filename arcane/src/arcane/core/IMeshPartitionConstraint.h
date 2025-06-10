// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshPartitionConstraint.h                                  (C) 2000-2025 */
/*                                                                           */
/* Interface d'une contrainte de partitionnement d'un maillage.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHPARTITIONCONSTRAINT_H
#define ARCANE_CORE_IMESHPARTITIONCONSTRAINT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une contrainte de partitionnement d'un maillage.
 *
 * Les instances de cette interface sont gérées par un
 * IMeshPartitionConstraintMng.
 */
class IMeshPartitionConstraint
{
 public:

  virtual ~IMeshPartitionConstraint() = default; //!< Libère les ressources

 public:

  /*!
   * \brief Ajoute un ensemble de contraintes sur les mailles.
   *
   * Ajoute dans le tableau \a linked_cells un ensemble de couples de
   * uniqueId() de mailles qui doivent rester dans la même partition.
   * Par exemple, si les mailles 25 et 23 doivent rester connectées, il
   * suffit d'appeler:
   * \code
   * linked_cells.add(23);
   * linked_cells.add(25);
   * \endcode
   *
   * Il faut toujours ajouter des couples de uniqueId(), en répétant
   * éventuellement les mailles. Par exemple, si on souhaite
   * que les mailles 35, 37 et 39,il faut faire comme suit:
   * \code
   * linked_cells.add(35);
   * linked_cells.add(37);
   * linked_cells.add(35);
   * linked_cells.add(39);
   * \endcode
   * Le tableau \a linked_cells doit avoir une taille multiple de 2.
   * Le tableau \a linked owners indique pour chaque couple à quel sous-domaine
   * il doit appartenir.
   *
   * TODO: Supprimer \a linked_owners
   *
   * \warning : chaque paire doit commencer par la cellule d'uid le plus petit.
   * Le "owner" indicate for every couple that correspond to the first cell.
   */
  virtual void addLinkedCells(Int64Array& linked_cells, Int32Array& linked_owners) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
