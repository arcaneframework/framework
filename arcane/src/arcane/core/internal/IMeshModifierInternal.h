// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshModifierInternal.h                                     (C) 2000-2025 */
/*                                                                           */
/* Internal Arcane component of IMeshModifier.                               */
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
 * \brief Internal part of IMeshModifier.
 */
class ARCANE_CORE_EXPORT IMeshModifierInternal
{
 public:

  virtual ~IMeshModifierInternal() = default;

  /*!
   * \brief Deletes entities marked with ItemFlags::II_NeedRemove.
   *
   * This method is called in MeshExchanger
   */
  virtual void removeNeedRemoveMarkedItems() = 0;

  /*!
   * \brief Adds a node.
   *
   * Adds a node with a unique ID \a uid. If the node already exists,
   * it is returned. It is generally not useful to create nodes directly,
   * as they are automatically created when an edge,
   * a face, or a cell is added.
   *
   * \return The created node or the existing node with the unique ID \a unique_id
   * if it already exists.
   *
   * \note For performance reasons, it is preferable to call addNodes() if
   * many nodes need to be added.
   */
  virtual NodeLocalId addNode(ItemUniqueId unique_id) = 0;

  /*!
   * \brief Adds a face.
   *
   * Adds a face with a unique ID \a uid, of type \a type_id, and containing
   * the nodes whose unique IDs are \a nodes_uids. If the face
   * already exists, it is returned.
   *
   * \return The created face or the existing face with the unique ID \a unique_id
   * if it already exists.
   *
   * \note For performance reasons, it is preferable to call addFaces() if
   * many faces need to be added.
   */
  virtual FaceLocalId addFace(ItemUniqueId unique_id, ItemTypeId type_id, ConstArrayView<Int64> nodes_uid) = 0;

  /*!
   * \brief Adds a cell.
   *
   * Adds a cell with a unique ID \a uid, of type \a type_id, and containing
   * the nodes whose unique IDs are \a nodes_uids. If the cell
   * already exists, it is returned.
   *
   * \return The created cell or the existing cell with the unique ID \a unique_id
   * if it already exists.
   *
   * \note For performance reasons, it is preferable to call addCells() if
   * many cells need to be added.
   */
  virtual CellLocalId addCell(ItemUniqueId unique_id, ItemTypeId type_id, ConstArrayView<Int64> nodes_uid) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
