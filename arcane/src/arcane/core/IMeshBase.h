// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshBase.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Interface for base mesh operations                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHBASE_H
#define ARCANE_CORE_IMESHBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for base mesh operations.
 *
 * This interface is created to gradually implement IMesh operations in a
 * new implementation.
 *
 * This interface should be temporary.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IItemFamilyModifier;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshBase
{
 public:

  virtual ~IMeshBase() = default;

 public:

  //! Handle on this mesh
  virtual MeshHandle handle() const = 0;

 public:

  //! Mesh name
  virtual String name() const = 0;

  //! Number of mesh nodes
  virtual Integer nbNode() = 0;

  //! Number of mesh edges
  virtual Integer nbEdge() = 0;

  //! Number of mesh faces
  virtual Integer nbFace() = 0;

  //! Number of mesh cells
  virtual Integer nbCell() = 0;

  /*!
   * \brief Number of elements of type \a ik.
   * \pre ik==IK_Node || ik==IK_Edge || ik==IK_Face || ik==IK_Cell
   */
  virtual Integer nbItem(eItemKind ik) = 0;

  //! Associated message manager
  virtual ITraceMng* traceMng() = 0;

  /*!
   * \brief Mesh dimension (1D, 2D, or 3D).
   *
   * The dimension corresponds to the dimension of the cells.
   * If cells of multiple dimensions are present, the highest dimension
   * is returned.
   * If the dimension has not yet been set, returns -1;
   */
  virtual Integer dimension() = 0;

  //! Group of all nodes
  virtual NodeGroup allNodes() = 0;

  //! Group of all edges
  virtual EdgeGroup allEdges() = 0;

  //! Group of all faces
  virtual FaceGroup allFaces() = 0;

  //! Group of all cells
  virtual CellGroup allCells() = 0;

  //! Group of all domain-specific nodes
  virtual NodeGroup ownNodes() = 0;

  //! Group of all domain-specific edges
  virtual EdgeGroup ownEdges() = 0;

  //! Group of all domain-specific faces
  virtual FaceGroup ownFaces() = 0;

  //! Group of all domain-specific cells
  virtual CellGroup ownCells() = 0;

  //! Group of all faces on the boundary.
  virtual FaceGroup outerFaces() = 0;

 public:

  //! Create a particle family named \a name
  virtual IItemFamily* createItemFamily(eItemKind ik, const String& name) = 0;

  /*!
   * \brief Returns the family named \a name.
   *
   * If \a create_if_needed is true, the family is created if it did not exist.
   * If \a register_modifier_if_created is true, the family modifier is registered
   */
  virtual IItemFamily* findItemFamily(eItemKind ik, const String& name, bool create_if_needed = false,
                                      bool register_modifier_if_created = false) = 0;

  /*!
   * \brief Returns the family named \a name.
   *
   * If the requested family does not exist, if \a throw_exception is true, an
   * exception is thrown, otherwise a null pointer is returned.
   */
  virtual IItemFamily* findItemFamily(const String& name, bool throw_exception = false) = 0;

  /*!
   * \brief Returns the IItemFamilyModifier interface for the family
   * named \a name and of type \a ik
   *
   * If this modifier is not found, returns nullptr
   */
  virtual IItemFamilyModifier* findItemFamilyModifier(eItemKind ik, const String& name) = 0;

  /*!
   * \brief Returns the entity family of type \a ik.
   *
   * \pre ik==IK_Node || ik==IK_Edge || ik==IK_Face || ik==IK_Cell
   */
  virtual IItemFamily* itemFamily(eItemKind ik) = 0;

  //! Returns the node family.
  virtual IItemFamily* nodeFamily() = 0;
  //! Returns the edge family.
  virtual IItemFamily* edgeFamily() = 0;
  //! Returns the face family.
  virtual IItemFamily* faceFamily() = 0;
  //! Returns the cell family.
  virtual IItemFamily* cellFamily() = 0;

  virtual IItemFamilyCollection itemFamilies() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
