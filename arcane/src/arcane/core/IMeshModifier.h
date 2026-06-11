// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshModifier.h                                             (C) 2000-2025 */
/*                                                                           */
/* Mesh modification interface.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHMODIFIER_H
#define ARCANE_CORE_IMESHMODIFIER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMesh;
class IExtraGhostCellsBuilder;
class IExtraGhostParticlesBuilder;
class IAMRTransportFunctor;
class IMeshModifierInternal;
class IMeshModifier;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Arguments for IMeshModifier::addCells().
 *
 * The format of cellsInfos() is identical to that of the
 * IMesh::allocateCells() method. If \a cellsLocalIds() is not empty, it
 * will contain the local IDs of the created cells.
 *
 * If an added cell has the same uniqueId() as an existing cell, the
 * existing cell is kept as is and nothing happens.
 *
 * The created cells are considered to belong to this subdomain.
 * If this is not the case, their ownership must be modified afterwards.
 *
 * By default, when adding cells, if the associated faces do not exist,
 * they are created automatically. This is only possible in sequential mode.
 * It is possible to disable this by calling setAllowBuildFaces().
 * In parallel, the value of isAllowBuildFaces() is ignored.
 */
class MeshModifierAddCellsArgs
{
 public:

  MeshModifierAddCellsArgs(Integer nb_cell, Int64ConstArrayView cell_infos)
  : m_nb_cell(nb_cell)
  , m_cell_infos(cell_infos)
  {}

  MeshModifierAddCellsArgs(Integer nb_cell, Int64ConstArrayView cell_infos,
                           Int32ArrayView cell_lids)
  : MeshModifierAddCellsArgs(nb_cell, cell_infos)
  {
    m_cell_lids = cell_lids;
  }

 public:

  Int32 nbCell() const { return m_nb_cell; }
  Int64ConstArrayView cellInfos() const { return m_cell_infos; }
  Int32ArrayView cellLocalIds() const { return m_cell_lids; }

  //! Indicates whether associated faces are allowed to be built
  void setAllowBuildFaces(bool v) { m_is_allow_build_faces = v; }
  bool isAllowBuildFaces() const { return m_is_allow_build_faces; }

 private:

  Int32 m_nb_cell = 0;
  Int64ConstArrayView m_cell_infos;
  //! Returns, list of localIds() of the created cells
  Int32ArrayView m_cell_lids;
  bool m_is_allow_build_faces = true;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Arguments for IMeshModifier::addFaces().
 */
class MeshModifierAddFacesArgs
{
 public:

  MeshModifierAddFacesArgs(Int32 nb_face, Int64ConstArrayView face_infos)
  : m_nb_face(nb_face)
  , m_face_infos(face_infos)
  {}

  MeshModifierAddFacesArgs(Int32 nb_face, Int64ConstArrayView face_infos,
                           Int32ArrayView face_lids)
  : MeshModifierAddFacesArgs(nb_face, face_infos)
  {
    m_face_lids = face_lids;
  }

 public:

  Int32 nbFace() const { return m_nb_face; }
  Int64ConstArrayView faceInfos() const { return m_face_infos; }
  Int32ArrayView faceLocalIds() const { return m_face_lids; }

 private:

  Int32 m_nb_face = 0;
  Int64ConstArrayView m_face_infos;
  //! Returns, list of localIds() of the created faces
  Int32ArrayView m_face_lids;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Mesh modification interface.
 *
 * This interface provides the services for modifying a mesh.
 * Mesh manipulation is a complex mechanism and is reserved for experienced
 * users. Some manipulations may leave the mesh in an inconsistent state.
 *
 * The supported operations depend on the mesh type. For performance reasons,
 * adding and deleting entities do not directly update the entity variables
 * or groups. For this to be taken into account, the endUpdate() method must
 * be called. In parallel, this also triggers the update of ghost entities.
 */
class ARCANE_CORE_EXPORT IMeshModifier
{
 public:

  virtual ~IMeshModifier() {} //<! Releases resources

 public:

  virtual void build() = 0;

 public:

  //! Associated mesh
  virtual IMesh* mesh() = 0;

 public:

  /*!
   * \brief Sets the property indicating whether the mesh can evolve.
   *
   * This property must be set to true if you wish to modify the mesh,
   * for example by exchanging entities via the exchangeItems() call.
   * This only concerns nodes, edges, faces, and cells, but not particles,
   * which can still be created and destroyed.
   *
   * By default, isDynamic() is false.
   *
   * The property setting can only be done during initialization.
   */
  virtual void setDynamic(bool v) = 0;

  /*!
   * \brief Adds cells.
   *
   * Adds cells. The format of \a cells_infos is identical to that of
   * the IMesh::allocateCells() method. If \a cells_lid is not empty,
   * it will contain the local IDs of the created cells. It is possible
   * to perform multiple successive additions. Once the additions are
   * complete, the endUpdate() method must be called. If an added cell
   * has the same uniqueId() as an existing cell, the existing cell is
   * kept as is and nothing happens.
   *
   * The created cells are considered to belong to this subdomain.
   * If this is not the case, their ownership must be modified afterwards.
   *
   * This method is collective. If a subdomain does not wish to add cells,
   * it is possible to pass an empty array.
   */
  virtual void addCells(Integer nb_cell, Int64ConstArrayView cell_infos,
                        Int32ArrayView cells_lid = Int32ArrayView()) = 0;

  //! Adds cells
  virtual void addCells(const MeshModifierAddCellsArgs& args);

  /*!
   * \brief Adds faces.
   *
   * \sa addFaces(const MeshModifierAddFacesArgs&)
   */
  virtual void addFaces(Integer nb_face, Int64ConstArrayView face_infos,
                        Int32ArrayView face_lids = Int32ArrayView()) = 0;

  /*!
   * \brief Adds faces.
   *
   * Adds faces. The format of \a face_infos is identical to that of the
   * IMesh::allocateCells() method. If \a face_lids is not empty, it will
   * contain the local IDs of the created faces. It is possible to perform
   * multiple successive additions. Once the additions are complete, the
   * endUpdate() method must be called. If an added face has the same uniqueId()
   * as an existing face, the existing face is kept as is and nothing happens.
   *
   * The created faces are considered to belong to this subdomain.
   * If this is not the case, their ownership must be modified afterwards.
   */
  virtual void addFaces(const MeshModifierAddFacesArgs& args);

  /*!
   * \brief Adds edges.
   *
   * Adds edges. The format of \a edge_infos is identical to that of the
   * IMesh::allocateCells() method. If \a edge_lids is not empty, it will
   * contain the local IDs of the created edges. It is possible to perform
   * multiple successive additions. Once the additions are complete, the
   * endUpdate() method must be called. If an added edge has the same uniqueId()
   * as an existing edge, the existing edge is kept as is and nothing happens.
   *
   * The created edges are considered to belong to this subdomain.
   * If this is not the case, their ownership must be modified afterwards.
   *
   * This method is collective. If a subdomain does not wish to add edges,
   * it is possible to pass an empty array.
   */
  virtual void addEdges(Integer nb_edge, Int64ConstArrayView edge_infos,
                        Int32ArrayView edge_lids = Int32ArrayView()) = 0;

  /*!
   * \brief Adds nodes.
   *
   * Adds nodes with unique identifiers being the values of the
   * \a nodes_uid array. If \a nodes_lid is not empty, it will contain the
   * local IDs of the created nodes. It is possible to perform multiple
   * successive additions. Once the additions are complete, the endUpdate()
   * method must be called. It is possible to specify an already existing
   * uniqueId(). In this case, the node is simply ignored.
   *
   * The created nodes are considered to belong to this subdomain.
   * If this is not the case, their ownership must be modified afterwards.
   *
   * This method is collective. If a subdomain does not wish to add nodes,
   * it is possible to pass an empty array.
   */
  virtual void addNodes(Int64ConstArrayView nodes_uid,
                        Int32ArrayView nodes_lid = Int32ArrayView()) = 0;

  /*!
   * \brief Removes cells.
   *
   * Removes the cells whose local IDs are provided in \a cells_local_id.
   * It is possible to perform multiple successive removals. Once the removals
   * are complete, the endUpdate() method must be called.
   */
  virtual void removeCells(Int32ConstArrayView cells_local_id) = 0;

  virtual void removeCells(Int32ConstArrayView cells_local_id, bool update_ghost) = 0;

  /*!
   * \brief Detaches cells from the mesh.
   *
   * The detached cells are disconnected from the mesh. The nodes, edges,
   * and faces of these cells no longer reference them, and the uniqueId()
   * of these cells can be reused. To permanently destroy these cells, the
   * removeDetachedCells() method must be called.
   */
  virtual void detachCells(Int32ConstArrayView cells_local_id) = 0;

  /*!
   * \brief Removes detached cells
   *
   * Removes detached cells via detachCells().
   * It is possible to perform multiple successive removals. Once the removals
   * are complete, the endUpdate() method must be called.
   */
  virtual void removeDetachedCells(Int32ConstArrayView cells_local_id) = 0;

  //! AMR
  virtual void flagCellToRefine(Int32ConstArrayView cells_lids) = 0;
  virtual void flagCellToCoarsen(Int32ConstArrayView cells_lids) = 0;
  virtual void refineItems() = 0;
  virtual void coarsenItems() = 0;
  virtual void coarsenItemsV2(bool update_parent_flag) = 0;
  virtual bool adapt() = 0;
  virtual void registerCallBack(IAMRTransportFunctor* f) = 0;
  virtual void unRegisterCallBack(IAMRTransportFunctor* f) = 0;
  virtual void addHChildrenCells(Cell parent_cell, Integer nb_cell,
                                 Int64ConstArrayView cells_infos, Int32ArrayView cells_lid = Int32ArrayView()) = 0;

  virtual void addParentCellToCell(Cell child, Cell parent) = 0;
  virtual void addChildCellToCell(Cell parent, Cell child) = 0;

  virtual void addParentFaceToFace(Face child, Face parent) = 0;
  virtual void addChildFaceToFace(Face parent, Face child) = 0;

  virtual void addParentNodeToNode(Node child, Node parent) = 0;
  virtual void addChildNodeToNode(Node parent, Node child) = 0;

  //! Deletes all entities of all families in this mesh.
  virtual void clearItems() = 0;

  /*!
   * \brief Adds cells from the data contained in \a buffer.
   *
   * \a buffer must contain serialized cells, for example by
   * calling IMesh::serializeCells().
   *
   * \deprecated Use IMesh::cellFamily()->policyMng()->createSerializer() instead.
   */
  ARCANE_DEPRECATED_240 virtual void addCells(ISerializer* buffer) = 0;

  /*!
   * \brief Adds cells from the data contained in \a buffer.
   *
   * \a buffer must contain serialized cells, for example by
   * calling IMesh::serializeCells(). In return, \a cells_local_id
   * contains the list of localIds() of the deserialized cells. A cell may
   * appear multiple times in this list if it appears multiple times in \a buffer.
   *
   * \deprecated Use IMesh::cellFamily()->policyMng()->createSerializer() instead.
   */
  ARCANE_DEPRECATED_240 virtual void addCells(ISerializer* buffer, Int32Array& cells_local_id) = 0;

  /*!
   * \brief Notifies the instance that mesh modification is finished.
   *
   * This method is collective.
   */
  virtual void endUpdate() = 0;

  virtual void endUpdate(bool update_ghost_layer, bool remove_old_ghost) = 0; // SDC: this signature is needed @IFPEN.

 public:

  /*!
   * \brief Updates the ghost layer.
   *
   * This operation is collective.
   */
  virtual void updateGhostLayers() = 0;

  //! AMR
  virtual void updateGhostLayerFromParent(Array<Int64>& ghost_cell_to_refine,
                                          Array<Int64>& ghost_cell_to_coarsen,
                                          bool remove_old_ghost) = 0;

  //! addition of the "extraordinary" ghost cells addition algorithm.
  virtual void addExtraGhostCellsBuilder(IExtraGhostCellsBuilder* builder) = 0;

  //! Removes the association with the \a builder instance.
  virtual void removeExtraGhostCellsBuilder(IExtraGhostCellsBuilder* builder) = 0;

  //! Addition of the "extraordinary" ghost particle addition algorithm
  virtual void addExtraGhostParticlesBuilder(IExtraGhostParticlesBuilder* builder) = 0;

  //! Removes the association with the \a builder instance.
  virtual void removeExtraGhostParticlesBuilder(IExtraGhostParticlesBuilder* builder) = 0;

 public:

  //! Merges the meshes of \a meshes with the current mesh.
  virtual void mergeMeshes(ConstArrayView<IMesh*> meshes) = 0;

 public:

  //! Internal API for Arcane
  virtual IMeshModifierInternal* _modifierInternalApi() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
