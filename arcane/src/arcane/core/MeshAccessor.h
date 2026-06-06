// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshAccessor.h                                              (C) 2000-2025 */
/*                                                                           */
/* Access to mesh information.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHACCESSOR_H
#define ARCANE_CORE_MESHACCESSOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arcane/core/VariableTypedef.h"
#include "arcane/core/MeshHandle.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISubDomain;
class IMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Access to mesh information.
 * \ingroup Mesh
 */
class ARCANE_CORE_EXPORT MeshAccessor
{
 public:

  ARCCORE_DEPRECATED_2020("Use constructor with MeshHande")
  MeshAccessor(ISubDomain* sd);
  MeshAccessor(IMesh* mesh);
  MeshAccessor(const MeshHandle& mesh_handle);

 public:

  //! Returns the number of cells in the mesh
  Integer nbCell() const;
  //! Returns the number of faces in the mesh
  Integer nbFace() const;
  //! Returns the number of edges in the mesh
  Integer nbEdge() const;
  //! Returns the number of nodes in the mesh
  Integer nbNode() const;

  //! Returns the coordinates of the mesh nodes
  VariableNodeReal3& nodesCoordinates() const;

  //! Returns the group containing all nodes
  NodeGroup allNodes() const;
  //! Returns the group containing all edges
  EdgeGroup allEdges() const;
  //! Returns the group containing all faces
  FaceGroup allFaces() const;
  //! Returns the group containing all cells
  CellGroup allCells() const;
  //! Returns the group containing all boundary faces.
  FaceGroup outerFaces() const;
  /*! \brief Returns the group containing all nodes specific to this domain.
   *
   * In sequential mode, this is allNodes(). In parallel mode, it is
   * all nodes that are not ghost nodes. The set of ownNodes() groups from all
   * sub-domains forms a partition of the global mesh.
   */
  NodeGroup ownNodes() const;
  /*! \brief Returns the group containing all cells specific to this domain.
   *
   * In sequential mode, this is allCells(). In parallel mode, it is
   * all cells that are not ghost cells. The set of ownCells() groups from all
   * sub-domains forms a partition of the global mesh.
   */
  CellGroup ownCells() const;
  /*! \brief Group containing all faces specific to this domain.
   *
   * In sequential mode, this is allFaces(). In parallel mode, it is
   * all faces that are not ghost faces. The set of ownFaces() groups from all
   * sub-domains forms a partition of the global mesh.
   */
  FaceGroup ownFaces() const;
  /*! \brief Group containing all edges specific to this domain.
   *
   * In sequential mode, this is allEdges(). In parallel mode, it is
   * all edges that are not ghost edges. The set of ownEdges() groups from all
   * sub-domains forms a partition of the global mesh.
   */
  EdgeGroup ownEdges() const;

 public:

  inline IMesh* mesh() const { return m_mesh_handle.mesh(); }
  inline const MeshHandle& meshHandle() const { return m_mesh_handle; }

 private:

  MeshHandle m_mesh_handle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
