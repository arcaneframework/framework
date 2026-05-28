// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FaceDirectionMng.cc                                         (C) 2000-2026 */
/*                                                                           */
/* Info on the faces of a structure mesh in X, Y, or Z direction.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_FACEDIRECTIONMNG_H
#define ARCANE_CARTESIANMESH_FACEDIRECTIONMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Item.h"
#include "arcane/core/VariableTypedef.h"
#include "arcane/core/ItemEnumerator.h"

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
 * \brief Info on the mesh before and after a face along a direction.
 *
 * Instances of this class are temporary and constructed via
 * FaceDirectionMng::face().
 */
class ARCANE_CARTESIANMESH_EXPORT DirFace
{
  friend FaceDirectionMng;

 private:

  DirFace(Cell n, Cell p)
  : m_previous(p)
  , m_next(n)
  {}

 public:

  //! Previous mesh
  Cell previousCell() const { return m_previous; }
  //! Previous mesh
  CellLocalId previousCellId() const { return m_previous.itemLocalId(); }
  //! Next mesh
  Cell nextCell() const { return m_next; }
  //! Next mesh
  CellLocalId nextCellId() const { return m_next.itemLocalId(); }

 private:

  Cell m_previous;
  Cell m_next;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Info on the mesh before and after a face along a direction.
 *
 * Instances of this class are temporary and constructed via
 * FaceDirectionMng::dirFaceId().
 */
class ARCANE_CARTESIANMESH_EXPORT DirFaceLocalId
{
  friend FaceDirectionMng;

 private:

  ARCCORE_HOST_DEVICE DirFaceLocalId(CellLocalId n, CellLocalId p)
  : m_previous(p)
  , m_next(n)
  {}

 public:

  //! Previous mesh
  ARCCORE_HOST_DEVICE CellLocalId previousCell() const { return m_previous; }
  //! Previous mesh
  ARCCORE_HOST_DEVICE CellLocalId previousCellId() const { return m_previous; }
  //! Next mesh
  ARCCORE_HOST_DEVICE CellLocalId nextCell() const { return m_next; }
  //! Next mesh
  ARCCORE_HOST_DEVICE CellLocalId nextCellId() const { return m_next; }

 private:

  CellLocalId m_previous;
  CellLocalId m_next;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Info on the faces of a specific direction X, Y, or Z
 * of a structured mesh.
 */
class ARCANE_CARTESIANMESH_EXPORT FaceDirectionMng
{
  friend CartesianMeshImpl;
  friend CartesianMeshPatch;
  class Impl;

 private:

  using ItemDirectionInfo = impl::CartesianItemDirectionInfo;

 public:

  /*!
   * \brief Creates an empty instance.
   *
   * The instance is not valid until _internalInit() has been called.
   */
  FaceDirectionMng();

 public:

  //! Direction face corresponding to face \a f.
  DirFace face(Face f) const
  {
    return _face(f.localId());
  }
  //! Direction face corresponding to face \a f.
  DirFace face(FaceLocalId f) const
  {
    return _face(f.localId());
  }

  //! Direction face corresponding to face \a f.
  ARCCORE_HOST_DEVICE DirFaceLocalId dirFaceId(FaceLocalId f) const
  {
    return _dirFaceId(f);
  }

  //! Group of all faces in the direction.
  FaceGroup allFaces() const;

  /*!
   * \brief Group of all overlap faces in the direction.
   *
   * These are all faces that have two overlap meshes around them.
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
   * The layer of overlap meshes refers to the layer of meshes of the same
   * level around the patch. These meshes may belong to one or more
   * patches.
   */
  FaceGroup overlapFaces() const;

  /*!
   * \brief Group of all faces within the patch in the direction.
   *
   * These are all faces that do not have two overlap meshes.
   * (`innerFaces() + outerFaces()` or simply `!overlapFaces()`)
   *
   * \warning Faces at the domain boundary (thus having only one
   * "outer" mesh) are included in this group. Therefore, one must not assume
   * that there are two meshes around every face in this group (for that,
   * one must stick with the innerFaces() group).
   */
  FaceGroup inPatchFaces() const;

  /*!
   * \brief Group of all internal faces in the direction.
   *
   * A face is considered internal if its mesh
   * in front and behind is not null and is not an overlap mesh.
   */
  FaceGroup innerFaces() const;

  /*!
   * \brief Group of all external faces in the direction.
   *
   * A face is considered external if its mesh
   * in front or behind is an overlap mesh or is null (if at the domain
   * boundary or if there are no layers of overlap meshes).
   *
   * \note Faces between patches are not duplicated. Therefore, some faces
   * in this group may also be in an outerFaces() of another patch.
   */
  FaceGroup outerFaces() const;

  //! Direction face corresponding to face \a f.
  DirFace operator[](Face f) const
  {
    return _face(f.localId());
  }

  //! Direction face corresponding to face \a f.
  DirFace operator[](FaceLocalId f) const
  {
    return _face(f.localId());
  }

  //! Direction face corresponding to the face iterator \a iface
  DirFace operator[](FaceEnumerator iface) const
  {
    return _face(iface.itemLocalId());
  }

  //! Direction value
  eMeshDirection direction() const
  {
    return m_direction;
  }

 private:

  //! Direction face corresponding to local face number \a local_id
  DirFace _face(Int32 local_id) const
  {
    ItemDirectionInfo d = m_infos_view[local_id];
    return DirFace(m_cells[d.m_next_lid], m_cells[d.m_previous_lid]);
  }

  //! Direction face corresponding to local face number \a local_id
  ARCCORE_HOST_DEVICE DirFaceLocalId _dirFaceId(FaceLocalId local_id) const
  {
    ItemDirectionInfo d = m_infos_view[local_id];
    return DirFaceLocalId(CellLocalId(d.m_next_lid), CellLocalId(d.m_previous_lid));
  }

 private:

  /*!
   * \internal
   * \brief Calculates the information on faces associated with the meshes
   * in the direction \a cell_dm.
   * Assumes that _internalInit() has been called.
   */
  void _internalComputeInfos(const CellDirectionMng& cell_dm,
                             const VariableCellReal3& cells_center,
                             const VariableFaceReal3& faces_center);

  void _internalComputeInfos(const CellDirectionMng& cell_dm);

  /*!
   * \internal
   * Initializes the instance.
   */
  void _internalInit(ICartesianMesh* cm, eMeshDirection dir, Integer patch_index);

  /*!
   * \internal
   * Destroys the resources associated with the instance.
   */
  void _internalDestroy();

  /*!
   * \brief Resizes the container holding the \a ItemDirectionInfo.
   *
   * This invalidates current instances of FaceDirectionMng.
   */
  void _internalResizeInfos(Int32 new_size);

  void _computeCellInfos(const CellDirectionMng& cell_dm,
                         const VariableCellReal3& cells_center,
                         const VariableFaceReal3& faces_center);
  void _computeCellInfos() const;
  bool _hasFace(Cell cell, Int32 face_local_id) const;

 private:

  SmallSpan<ItemDirectionInfo> m_infos_view;
  CellInfoListView m_cells;
  eMeshDirection m_direction;
  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
