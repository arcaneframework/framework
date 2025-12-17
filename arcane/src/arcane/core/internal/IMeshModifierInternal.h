// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshModifierInternal.h                                     (C) 2000-2025 */
/*                                                                           */
/* Partie interne à Arcane de IMeshModifier.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IMESHMODIFIERINTERNAL_H
#define ARCANE_CORE_INTERNAL_IMESHMODIFIERINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Partie interne de IMeshModifier.
 */
class ARCANE_CORE_EXPORT IMeshModifierInternal
{
 public:

  virtual ~IMeshModifierInternal() = default;

  /*!
   * \brief Suppime les entités marquées avec ItemFlags::II_NeedRemove.
   *
   * Cette méthode est appelée dans MeshExchanger
   */
  virtual void removeNeedRemoveMarkedItems() = 0;

  /*!
   * \brief Ajoute un nœud.
   *
   * Ajoute un nœud de numéro unique \a uid. Si le nœud existe déjà,
   * il est retournée. Il n'est en général pas utile de créer directement les
   * nœuds, car ces derniers le sont automatiquement quand on ajoute une arête,
   * une face ou une maille.
   *
   * \return Le nœud créé ou le nœud existant avec le numéro unique \a unique_id
   * s'il existe déjà.
   *
   * \note Pour des raisons de performance, il est préférable d'appeler addNodes() si
   * on doit ajouter beaucoup de nœuds.
   */
  virtual NodeLocalId addNode(ItemUniqueId unique_id) = 0;

  /*!
   * \brief Ajoute une face.
   *
   * Ajoute une face de numéro unique \a uid, de type \a type_id et contenant
   * les nœuds dont les numéros uniques sont \a nodes_uids. Si la face
   * existe déjà, elle est retournée.
   *
   * \return La face créée ou la face existante avec le numéro unique \a unique_id
   * si elle existe déjà.
   *
   * \note Pour des raisons de performance, il est préférable d'appeler addFaces() si
   * on doit ajouter beaucoup de faces.
   */
  virtual FaceLocalId addFace(ItemUniqueId unique_id, ItemTypeId type_id, ConstArrayView<Int64> nodes_uid) = 0;

  /*!
   * \brief Ajoute une maille.
   *
   * Ajoute une maille de numéro unique \a uid, de type \a type_id et contenant
   * les nœuds dont les numéros uniques sont \a nodes_uids. Si la maille
   * existe déjà, elle est retournée.
   *
   * \return La maille créée ou la maille existante avec le numéro unique \a unique_id
   * si elle existe déjà.
   *
   * \note Pour des raisons de performance, il est préférable d'appeler addCells() si
   * on doit ajouter beaucoup de mailles.
   */
  virtual CellLocalId addCell(ItemUniqueId unique_id, ItemTypeId type_id, ConstArrayView<Int64> nodes_uid) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
