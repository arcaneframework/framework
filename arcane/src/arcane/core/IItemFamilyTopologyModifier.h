// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilyTopologyModifier.h                               (C) 2000-2025 */
/*                                                                           */
/* Topology modification interface for entities within a family.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IITEMFAMILYTOPOLOGYMODIFIER_H
#define ARCANE_IITEMFAMILYTOPOLOGYMODIFIER_H
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
 * \ingroup Mesh
 * \brief Interface for modifying the topology of entities within a family.
 *
 * This class is intended to be temporary and serves to replace
 * direct calls to ItemInternal by managing old or new
 * connectivities.
 *
 * \warning This class allows direct modification of the connectivities
 * of unstructured meshes, which can introduce inconsistencies in
 * the topology and connectivities. Therefore, extreme caution must be
 * used when employing methods from this class. It is preferable
 * to use the methods of IMeshModifier if you wish to add/remove
 * mesh entities while guaranteeing consistency.
 */
class ARCANE_CORE_EXPORT IItemFamilyTopologyModifier
{
 public:

  //! Frees resources
  virtual ~IItemFamilyTopologyModifier() = default;

 public:

  //! Associated family
  virtual IItemFamily* family() const = 0;

 public:

  /*!
   * \brief Replaces a node of an entity.
   *
   * Replaces the \a index-th node of the entity in the family \a family()
   * with local ID \a item_lid by the node with local ID \a new_node_lid.
   */
  virtual void replaceNode(ItemLocalId item_lid, Integer index,
                           ItemLocalId new_node_lid) = 0;

  /*!
   * \brief Replaces an edge of an entity.
   *
   * Replaces the \a index-th edge of the entity in the family \a family()
   * with local ID \a item_lid by the edge with local ID \a new_edge_lid.
   */
  virtual void replaceEdge(ItemLocalId item_lid, Integer index,
                           ItemLocalId new_edge_lid) = 0;

  /*!
   * \brief Replaces a face of an entity.
   *
   * Replaces the \a index-th face of the entity in the family \a family()
   * with local ID \a item_lid by the face with local ID \a new_face_lid.
   */
  virtual void replaceFace(ItemLocalId item_lid, Integer index,
                           ItemLocalId new_face_lid) = 0;

  /*!
   * \brief Replaces a cell of an entity.
   *
   * Replaces the \a index-th cell of the entity in the family \a family()
   * with local ID \a item_lid by the face with local ID \a new_cell_lid.
   */
  virtual void replaceCell(ItemLocalId item_lid, Integer index,
                           ItemLocalId new_cell_lid) = 0;

  /*!
   * \brief Replaces a parent entity of an entity.
   *
   * Replaces the \a index-th parent entity of the entity in the family \a family()
   * with local ID \a item_lid by the parent entity with local ID \a new_hparent_lid.
   */
  virtual void replaceHParent(ItemLocalId item_lid, Integer index,
                              ItemLocalId new_hparent_lid) = 0;

  /*!
   * \brief Replaces a child entity of an entity.
   *
   * Replaces the \a index-th child entity of the entity in the family \a family()
   * with local ID \a item_lid by the child entity with local ID \a new_hchild_lid.
   */
  virtual void replaceHChild(ItemLocalId item_lid, Integer index,
                             ItemLocalId new_hchild_lid) = 0;

  /*!
   * \brief Finds and replaces a node of an entity.
   *
   * Replaces the node with local ID \a old_node_lid of the entity in the family \a family()
   * with local ID \a item_lid by the node with local ID \a new_node_lid.
   *
   * Throws an exception if the node \a old_node_id is not found.
   */
  virtual void findAndReplaceNode(ItemLocalId item_lid, ItemLocalId old_node_lid,
                                  ItemLocalId new_node_lid) = 0;

  /*!
   * \brief Finds and replaces an edge of an entity.
   *
   * Replaces the edge with local ID \a old_edge_lid of the entity in the family \a family()
   * with local ID \a item_lid by the edge with local ID \a new_edge_lid.
   *
   * Throws an exception if the edge \a old_edge_lid is not found.
   */
  virtual void findAndReplaceEdge(ItemLocalId item_lid, ItemLocalId old_edge_lid,
                                  ItemLocalId new_edge_lid) = 0;

  /*!
   * \brief Finds and replaces a face of an entity.
   *
   * Replaces the face with local ID \a old_face_lid of the entity in the family \a family()
   * with local ID \a item_lid by the face with local ID \a new_face_lid.
   *
   * Throws an exception if the face \a old_face_lid is not found.
   */
  virtual void findAndReplaceFace(ItemLocalId item_lid, ItemLocalId old_face_lid,
                                  ItemLocalId new_face_lid) = 0;

  /*!
   * \brief Finds and replaces a cell of an entity.
   *
   * Replaces the cell with local ID \a old_cell_lid of the entity in the family \a family()
   * with local ID \a item_lid by the face with local ID \a new_cell_lid.
   *
   * Throws an exception if the cell \a old_cell_lid is not found.
   */
  virtual void findAndReplaceCell(ItemLocalId item_lid, ItemLocalId old_cell_lid,
                                  ItemLocalId new_cell_lid) = 0;

  /*!
   * \brief Positions a cell in front and behind a face.
   *
   * This method is only implemented for face families. For other
   * families, it raises a NotSupportedException.
   *
   * * \param face_lid local ID of the face
   * \param back_cell_lid local ID of the cell behind (or NULL_ITEM_LOCAL_ID)
   * \param front_cell_lid local ID of the cell in front (or NULL_ITEM_LOCAL_ID)
   */
  virtual void setBackAndFrontCells(FaceLocalId face_lid, CellLocalId back_cell_lid,
                                    CellLocalId front_cell_lid);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
