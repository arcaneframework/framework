// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellDirectionMng.cc                                         (C) 2000-2026 */
/*                                                                           */
/* Info on the meshes in an X, Y, or Z direction of a structured mesh.       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CELLDIRECTIONMNG_H
#define ARCANE_CARTESIANMESH_CELLDIRECTIONMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

#include "arcane/core/Item.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IndexedItemConnectivityView.h"

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"
#include "arcane/cartesianmesh/CartesianItemDirectionInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Mesh before and after a mesh along a direction.
 *
 * Instances of this class are temporary and constructed via
 * CellDirectionMng::cell().
 */
class ARCANE_CARTESIANMESH_EXPORT DirCell
{
 public:

  DirCell(Cell n, Cell p)
  : m_previous(p)
  , m_next(n)
  {}

 public:

  //! Previous mesh
  Cell previous() const { return m_previous; }
  //! Previous mesh
  CellLocalId previousId() const { return CellLocalId(m_previous.localId()); }
  //! Next mesh
  Cell next() const { return m_next; }
  //! Next mesh
  CellLocalId nextId() const { return CellLocalId(m_next.localId()); }

 private:

  Cell m_previous;
  Cell m_next;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Mesh before and after a mesh along a direction.
 *
 * Instances of this class are temporary and constructed via
 * CellDirectionMng::cellLocalId().
 */
class ARCANE_CARTESIANMESH_EXPORT DirCellLocalId
{
 public:

  constexpr ARCCORE_HOST_DEVICE DirCellLocalId(CellLocalId n, CellLocalId p)
  : m_previous(p)
  , m_next(n)
  {}

 public:

  //! Previous mesh
  constexpr ARCCORE_HOST_DEVICE CellLocalId previous() const { return m_previous; }
  //! Previous mesh
  constexpr ARCCORE_HOST_DEVICE CellLocalId previousId() const { return m_previous; }
  //! Next mesh
  constexpr ARCCORE_HOST_DEVICE CellLocalId next() const { return m_next; }
  //! Next mesh
  constexpr ARCCORE_HOST_DEVICE CellLocalId nextId() const { return m_next; }

 private:

  CellLocalId m_previous;
  CellLocalId m_next;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Mesh with directional node information.
 *
 * Instances of this class are temporary and constructed via
 * CellDirectionMng::cellNode().
 */
class ARCANE_CARTESIANMESH_EXPORT DirCellNode
{
  friend CellDirectionMng;
  using Int8 = std::int8_t;

 private:

  DirCellNode(Cell c, const Int8* nodes_indirection)
  : m_cell(c)
  , m_nodes_indirection(nodes_indirection)
  {}

 public:

  //! Associated mesh
  Cell cell() const { return m_cell; }
  //! Associated mesh
  CellLocalId cellId() const { return CellLocalId(m_cell.localId()); }

  //! Node forward left in the direction
  Node nextLeft() const { return m_cell.node(m_nodes_indirection[CNP_NextLeft]); }
  //! Node forward right in the direction
  Node nextRight() const { return m_cell.node(m_nodes_indirection[CNP_NextRight]); }
  //! Node backward right in the direction
  Node previousRight() const { return m_cell.node(m_nodes_indirection[CNP_PreviousRight]); }
  //! Node backward left in the direction
  Node previousLeft() const { return m_cell.node(m_nodes_indirection[CNP_PreviousLeft]); }

  //! Node forward left in the direction
  NodeLocalId nextLeftId() const { return NodeLocalId(m_cell.nodeId(m_nodes_indirection[CNP_NextLeft])); }
  //! Node forward right in the direction
  NodeLocalId nextRightId() const { return NodeLocalId(m_cell.nodeId(m_nodes_indirection[CNP_NextRight])); }
  //! Node backward right in the direction
  NodeLocalId previousRightId() const { return NodeLocalId(m_cell.nodeId(m_nodes_indirection[CNP_PreviousRight])); }
  //! Node backward left in the direction
  NodeLocalId previousLeftId() const { return NodeLocalId(m_cell.nodeId(m_nodes_indirection[CNP_PreviousLeft])); }

  //! Node forward left in the direction
  Node topNextLeft() const { return m_cell.node(m_nodes_indirection[CNP_TopNextLeft]); }
  //! Node forward right in the direction
  Node topNextRight() const { return m_cell.node(m_nodes_indirection[CNP_TopNextRight]); }
  //! Node backward right in the direction
  Node topPreviousRight() const { return m_cell.node(m_nodes_indirection[CNP_TopPreviousRight]); }
  //! Node backward left in the direction
  Node topPreviousLeft() const { return m_cell.node(m_nodes_indirection[CNP_TopPreviousLeft]); }

  //! Node forward left in the direction
  NodeLocalId topNextLeftId() const { return NodeLocalId(m_cell.nodeId(m_nodes_indirection[CNP_TopNextLeft])); }
  //! Node forward right in the direction
  NodeLocalId topNextRightId() const { return NodeLocalId(m_cell.nodeId(m_nodes_indirection[CNP_TopNextRight])); }
  //! Node backward right in the direction
  NodeLocalId topPreviousRightId() const { return NodeLocalId(m_cell.nodeId(m_nodes_indirection[CNP_TopPreviousRight])); }
  //! Node backward left in the direction
  NodeLocalId topPreviousLeftId() const { return NodeLocalId(m_cell.nodeId(m_nodes_indirection[CNP_TopPreviousLeft])); }

 private:

  Cell m_cell;
  const Int8* m_nodes_indirection;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Mesh with directional node information.
 *
 * Instances of this class are temporary and constructed via
 * CellDirectionMng::cellNode().
 */
class ARCANE_CARTESIANMESH_EXPORT DirCellNodeLocalId
{
  friend CellDirectionMng;
  using Int8 = std::int8_t;

 private:

  ARCCORE_HOST_DEVICE DirCellNodeLocalId(CellLocalId c, const Int8* nodes_indirection, IndexedCellNodeConnectivityView view)
  : m_cell(c)
  , m_nodes_indirection(nodes_indirection)
  , m_view(view)
  {}

 public:

  //! Associated mesh
  ARCCORE_HOST_DEVICE CellLocalId cellId() const { return m_cell; }

  //! Node forward left in the direction
  ARCCORE_HOST_DEVICE NodeLocalId nextLeftId() const { return m_view.nodeId(m_cell, m_nodes_indirection[CNP_NextLeft]); }
  //! Node forward right in the direction
  ARCCORE_HOST_DEVICE NodeLocalId nextRightId() const { return m_view.nodeId(m_cell, m_nodes_indirection[CNP_NextRight]); }
  //! Node backward right in the direction
  ARCCORE_HOST_DEVICE NodeLocalId previousRightId() const { return m_view.nodeId(m_cell, m_nodes_indirection[CNP_PreviousRight]); }
  //! Node backward left in the direction
  ARCCORE_HOST_DEVICE NodeLocalId previousLeftId() const { return m_view.nodeId(m_cell, m_nodes_indirection[CNP_PreviousLeft]); }

  //! Node forward left in the direction
  ARCCORE_HOST_DEVICE NodeLocalId topNextLeftId() const { return m_view.nodeId(m_cell, m_nodes_indirection[CNP_TopNextLeft]); }
  //! Node forward right in the direction
  ARCCORE_HOST_DEVICE NodeLocalId topNextRightId() const { return m_view.nodeId(m_cell, m_nodes_indirection[CNP_TopNextRight]); }
  //! Node backward right in the direction
  ARCCORE_HOST_DEVICE NodeLocalId topPreviousRightId() const { return m_view.nodeId(m_cell, m_nodes_indirection[CNP_TopPreviousRight]); }
  //! Node backward left in the direction
  ARCCORE_HOST_DEVICE NodeLocalId topPreviousLeftId() const { return m_view.nodeId(m_cell, m_nodes_indirection[CNP_TopPreviousLeft]); }

 private:

  CellLocalId m_cell;
  const Int8* m_nodes_indirection;
  IndexedCellNodeConnectivityView m_view;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Mesh with directional face information.
 *
 * Instances of this class are temporary and constructed via
 * CellDirectionMng::cellFace().
 */
class ARCANE_CARTESIANMESH_EXPORT DirCellFace
{
  friend CellDirectionMng;

 private:

  DirCellFace(Cell c, Int32 next_face_index, Int32 previous_face_index)
  : m_cell(c)
  , m_next_face_index(next_face_index)
  , m_previous_face_index(previous_face_index)
  {
  }

 public:

  //! Associated mesh
  Cell cell() const { return m_cell; }
  //! Associated mesh
  CellLocalId cellId() const { return m_cell.itemLocalId(); }

  //! Face connected to the mesh after the current mesh in the direction
  Face next() const { return m_cell.face(m_next_face_index); }
  //! Face connected to the mesh after the current mesh in the direction
  FaceLocalId nextId() const { return m_cell.faceId(m_next_face_index); }

  //! Face connected to the mesh before the current mesh in the direction
  Face previous() const { return m_cell.face(m_previous_face_index); }
  //! Face connected to the mesh before the current mesh in the direction
  FaceLocalId previousId() const { return m_cell.faceId(m_previous_face_index); }

  //! Local index in the face of the next() mesh
  Int32 nextLocalIndex() const { return m_next_face_index; }

  //! Local index in the face of the previous() mesh
  Int32 previousLocalIndex() const { return m_previous_face_index; }

 private:

  Cell m_cell;
  Int32 m_next_face_index;
  Int32 m_previous_face_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Mesh with directional face information.
 *
 * Instances of this class are temporary and constructed via
 * CellDirectionMng::cellFace().
 */
class ARCANE_CARTESIANMESH_EXPORT DirCellFaceLocalId
{
  friend CellDirectionMng;

 private:

  ARCCORE_HOST_DEVICE DirCellFaceLocalId(CellLocalId c, Int32 next_face_index,
                                         Int32 previous_face_index,
                                         IndexedCellFaceConnectivityView view)
  : m_cell(c)
  , m_next_face_index(next_face_index)
  , m_previous_face_index(previous_face_index)
  , m_view(view)
  {
  }

 public:

  //! Associated mesh
  ARCCORE_HOST_DEVICE CellLocalId cell() const { return m_cell; }
  //! Associated mesh
  ARCCORE_HOST_DEVICE CellLocalId cellId() const { return m_cell; }

  //! Face connected to the mesh after the current mesh in the direction
  ARCCORE_HOST_DEVICE FaceLocalId next() const { return m_view.faceId(m_cell, m_next_face_index); }
  //! Face connected to the mesh after the current mesh in the direction
  ARCCORE_HOST_DEVICE FaceLocalId nextId() const { return m_view.faceId(m_cell, m_next_face_index); }

  //! Face connected to the mesh before the current mesh in the direction
  ARCCORE_HOST_DEVICE FaceLocalId previous() const { return m_view.faceId(m_cell, m_previous_face_index); }
  //! Face connected to the mesh before the current mesh in the direction
  ARCCORE_HOST_DEVICE FaceLocalId previousId() const { return m_view.faceId(m_cell, m_previous_face_index); }

  //! Local index in the face of the next() mesh
  ARCCORE_HOST_DEVICE Int32 nextLocalIndex() const { return m_next_face_index; }

  //! Local index in the face of the previous() mesh
  ARCCORE_HOST_DEVICE Int32 previousLocalIndex() const { return m_previous_face_index; }

 private:

  CellLocalId m_cell;
  Int32 m_next_face_index;
  Int32 m_previous_face_index;
  IndexedCellFaceConnectivityView m_view;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Info about the meshes in a specific X, Y, or Z direction of a structured mesh.
 *
 * This class contains the information needed to return the list
 * of meshes in a given direction and to know the mesh before and after
 * within that direction.
 *
 * Instances of this class are managed by an ICartesianMesh and are
 * temporary. They should not be kept from one iteration to the next
 * because they are invalidated if the mesh changes.
 *
 * This class has a reference semantics.
 *
 * For example, to iterate over the meshes in the X direction:
 
 \code
 * ICartesianMesh* cartesian_mesh = ...;
 * CellDirectionMng cdm(cartesian_mesh->cellDirection(MD_DirX));
 * ENUMERATE_CELL(icell,cdm.allCells()){
 *   DirCell dir_cell(cdm[icell]);
 *   Cell next = dir_cell.next();
 *   Cell prev = dir_cell.previous();
 * }
 \endcode
 *
 */
class ARCANE_CARTESIANMESH_EXPORT CellDirectionMng
{
  friend CartesianMeshImpl;
  friend CartesianMeshPatch;
  class Impl;
  static const int MAX_NB_NODE = 8;
  using Int8 = std::int8_t;

 private:

  using ItemDirectionInfo = impl::CartesianItemDirectionInfo;

 public:

  /*!
   * \brief Creates an empty instance.
   *
   * The instance is not valid until _internalInit() has been called.
   */
  CellDirectionMng();

 public:

  //! Directional mesh corresponding to mesh \a c.
  DirCell cell(Cell c) const
  {
    return _cell(c.localId());
  }
  //! Directional mesh corresponding to mesh \a c.
  DirCell cell(CellLocalId c) const
  {
    return _cell(c.localId());
  }
  //! Directional mesh corresponding to mesh \a c.
  DirCell dirCell(CellLocalId c) const
  {
    return _cell(c.localId());
  }
  //! Directional mesh corresponding to mesh \a c.
  ARCCORE_HOST_DEVICE DirCellLocalId dirCellId(CellLocalId c) const
  {
    return _dirCellId(c);
  }

  //! Mesh with directional info at nodes corresponding to mesh \a c.
  DirCellNode cellNode(Cell c) const
  {
    return DirCellNode(c, m_nodes_indirection);
  }

  //! Mesh with directional info at nodes corresponding to mesh \a c.
  DirCellNode cellNode(CellLocalId c) const
  {
    return DirCellNode(m_cells[c.localId()], m_nodes_indirection);
  }

  //! Mesh with directional info at nodes corresponding to mesh \a c.
  DirCellNode dirCellNode(CellLocalId c) const
  {
    return DirCellNode(m_cells[c.localId()], m_nodes_indirection);
  }

  //! Mesh with directional info at nodes corresponding to mesh \a c.
  ARCCORE_HOST_DEVICE DirCellNodeLocalId dirCellNodeId(CellLocalId c) const
  {
    return DirCellNodeLocalId(c, m_nodes_indirection, m_cell_node_view);
  }

  //! Mesh with directional info at faces corresponding to mesh \a c.
  DirCellFace cellFace(Cell c) const
  {
    return DirCellFace(c, m_next_face_index, m_previous_face_index);
  }
  //! Mesh with directional info at faces corresponding to mesh \a c.
  DirCellFace cellFace(CellLocalId c) const
  {
    return DirCellFace(m_cells[c.localId()], m_next_face_index, m_previous_face_index);
  }

  //! Mesh with directional info at faces corresponding to mesh \a c.
  ARCCORE_HOST_DEVICE DirCellFaceLocalId dirCellFaceId(CellLocalId c) const
  {
    return DirCellFaceLocalId(c, m_next_face_index, m_previous_face_index, m_cell_face_view);
  }

  //! Group of all meshes in the direction.
  CellGroup allCells() const;

  /*!
   * \brief Group of all overlap meshes in the direction.
   *
   *   0   1  2  3  4
   * ┌───┬──┬──┬──┬──┐
   * │   │  │  │  │  │
   * │   ├──┼──┼──┼──┤
   * │   │  │  │  │  │
   * └───┴──┴──┴──┴──┘
   *
   * 0 : level -1
   * 1 and 2 : Overlap meshes (overlapCells)
   * 3 : Outer meshes (outerCells)
   * 4 : Inner meshes (innerCells)
   *
   * The overlap mesh layer designates the layer of meshes of the same
   * level around the patch. These meshes may belong to one or more
   * patches.
   */
  CellGroup overlapCells() const;

  /*!
   * \brief Group of all patch meshes in the direction.
   *
   * Groups all meshes that are neither overlap nor ghost.
   * (`innerCells() + outerCells()` or simply `!overlapCells()`)
   */
  CellGroup inPatchCells() const;

  /*!
   * \brief Group of all inner meshes in the direction.
   *
   * A mesh is considered inner if its mesh
   * before or after is not null and is not an overlap mesh.
   */
  CellGroup innerCells() const;

  /*!
   * \brief Group of all outer meshes in the direction.
   *
   * A mesh is considered outer if its mesh
   * before or after is an overlap mesh or is null (if it is at the edge of the
   * domain or if there are no overlap mesh layers).
   */
  CellGroup outerCells() const;

  //! Directional mesh corresponding to mesh \a c.
  DirCell operator[](Cell c) const
  {
    return _cell(c.localId());
  }

  //! Directional mesh corresponding to mesh \a c.
  DirCell operator[](CellLocalId c) const
  {
    return _cell(c.localId());
  }

  //! Directional mesh corresponding to the iterator of mesh \a icell.
  DirCell operator[](CellEnumerator icell) const
  {
    return _cell(icell.itemLocalId());
  }

  /*!
   * \brief Global number of meshes in this direction.
   *
   * \note The returned value is only valid if the
   * mesh was created with a specific generator, such as
   * the SodMeshGenerator or the CartesianMeshGenerator. Otherwise, the returned value is (-1)
   */
  Int64 globalNbCell() const;

  /*!
   * \brief Number of own meshes in this direction.
   *
   * \note The returned value is only valid if the
   * mesh was created with a specific generator, such as
   * the SodMeshGenerator or the CartesianMeshGenerator. Otherwise, the returned value is (-1)
   */
  Int32 ownNbCell() const;

  /*!
   * \brief Offset of the subdomain in this direction.
   *
   * Assuming the global Cartesian mesh is divided into
   * several rectangular subdomains that form a grid,
   * this method returns the position of this subdomain in this grid
   * for this direction.
   *
   * \warning Using this method assumes that each
   * subdomain is parallelepiped (in 3D) or rectangular (in 2D)
   * which is not necessarily the case, especially with
   * mesh migration load balancing mechanisms.
   *
   * \note The returned value is only valid if the
   * mesh was created with a specific generator, such as
   * the CartesianMeshGenerator. Otherwise, the returned value is (-1)
   */
  Int32 subDomainOffset() const;

  /*!
   * \brief Offset of the first own mesh of this subdomain in this direction.
   *
   * Assuming the global Cartesian mesh is divided into
   * several rectangular subdomains that form a grid,
   * this method returns the position of the first
   * own mesh of this subdomain in this grid for this direction.
   *
   * \warning Using this method assumes that each
   * subdomain is parallelepiped (in 3D) or rectangular (in 2D)
   * which is not necessarily the case, especially with
   * mesh migration load balancing mechanisms.
   *
   * \note The returned value is only valid if the
   * mesh was created with a specific generator, such as
   * the CartesianMeshGenerator. Otherwise, the returned value is (-1)
   */
  Int64 ownCellOffset() const;

 private:

  //! Directional mesh corresponding to local mesh number \a local_id
  DirCell _cell(Int32 local_id) const
  {
    ItemDirectionInfo d = m_infos_view[local_id];
    return DirCell(m_cells[d.m_next_lid], m_cells[d.m_previous_lid]);
  }

  //! Directional mesh corresponding to local mesh number \a local_id
  ARCCORE_HOST_DEVICE DirCellLocalId _dirCellId(Int32 local_id) const
  {
    ItemDirectionInfo d = m_infos_view[local_id];
    return DirCellLocalId(CellLocalId(d.m_next_lid), CellLocalId(d.m_previous_lid));
  }

  void setNodesIndirection(ConstArrayView<Int8> nodes_indirection);

 protected:

  /*!
   * \internal
   * \brief Internal usage in Arcane. Calculates internal and external entities.
   * Assumes init() has been called.
   */
  void _internalComputeInnerAndOuterItems(const ItemGroup& items);
  void _internalComputeCellGroups(const CellGroup& all_cells, const CellGroup& in_patch_cells, const CellGroup& overlap_cells);

  /*!
   * \internal
   * Initializes the instance.
   */
  void _internalInit(ICartesianMesh* cm, eMeshDirection dir, Integer patch_index);

  /*!
   * \internal
   * Destroys resources associated with the instance.
   */
  void _internalDestroy();

  /*!
   * \internal
   * Positions the local face indices to the next and previous mesh.
   */
  void _internalSetLocalFaceIndex(Int32 next_index, Int32 previous_index)
  {
    m_next_face_index = next_index;
    m_previous_face_index = previous_index;
  }

  /*!
   * \brief Resizes the container holding the \a ItemDirectionInfo.
   *
   * This invalidates current instances of CellDirectionMng.
   */
  void _internalResizeInfos(Int32 new_size);

  void _internalSetOffsetAndNbCellInfos(Int64 global_nb_cell, Int32 own_nb_cell,
                                        Int32 sub_domain_offset, Int64 own_cell_offset);

 public:

  //! Direction value
  eMeshDirection direction() const
  {
    return m_direction;
  }

 private:

  SmallSpan<ItemDirectionInfo> m_infos_view;
  CellInfoListView m_cells;
  eMeshDirection m_direction;
  Int32 m_next_face_index;
  Int32 m_previous_face_index;
  Int8 m_nodes_indirection[MAX_NB_NODE];
  Impl* m_p = nullptr;
  IndexedCellNodeConnectivityView m_cell_node_view;
  IndexedCellFaceConnectivityView m_cell_face_view;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
