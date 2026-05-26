// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianPatch.h                                            (C) 2000-2026 */
/*                                                                           */
/* AMR Patch of a Cartesian mesh.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANPATCH_H
#define ARCANE_CARTESIANMESH_CARTESIANPATCH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

#include "arcane/cartesianmesh/ICartesianMeshPatch.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief AMR Patch of a Cartesian mesh.
 */
class ARCANE_CARTESIANMESH_EXPORT CartesianPatch
{
 public:

  //! Null patch.
  CartesianPatch() = default;

  //! Null patch.
  explicit CartesianPatch(ICartesianMeshPatch* patch_interface)
  : m_patch(patch_interface)
  {
  }
  CartesianPatch& operator=(const CartesianPatch&) = default;
  CartesianPatch& operator=(ICartesianMeshPatch* patch_interface)
  {
    m_patch = patch_interface;
    return (*this);
  }

 public:

  //! Cell group of the patch (including overlap cells).
  CellGroup cells() const;

  /*!
   * \brief Cell group of the patch (excluding overlap cells).
   *
   * Valid only with AMR type 3 (PatchCartesianMeshOnly).
   */
  CellGroup inPatchCells() const;

  /*!
   * \brief Overlap cell group of the patch.
   *
   * Valid only with AMR type 3 (PatchCartesianMeshOnly).
   */
  CellGroup overlapCells() const;

  //! Index of the patch in the patch array.
  Integer index() const;

  /*!
   * \brief Patch level.
   *
   * Valid only with AMR type 3 (PatchCartesianMeshOnly).
   */
  Integer level() const
  {
    ARCANE_CHECK_POINTER(m_patch);
    return m_patch->position().level();
  }

  //! List of cells in direction \a dir
  CellDirectionMng& cellDirection(eMeshDirection dir)
  {
    ARCANE_CHECK_POINTER(m_patch);
    return m_patch->cellDirection(dir);
  }

  //! List of cells in direction \a dir (0, 1 or 2)
  CellDirectionMng& cellDirection(Integer idir)
  {
    ARCANE_CHECK_POINTER(m_patch);
    return m_patch->cellDirection(idir);
  }

  //! List of faces in direction \a dir
  FaceDirectionMng& faceDirection(eMeshDirection dir)
  {
    ARCANE_CHECK_POINTER(m_patch);
    return m_patch->faceDirection(dir);
  }

  //! List of faces in direction \a dir (0, 1 or 2)
  FaceDirectionMng& faceDirection(Integer idir)
  {
    ARCANE_CHECK_POINTER(m_patch);
    return m_patch->faceDirection(idir);
  }

  //! List of nodes in direction \a dir
  NodeDirectionMng& nodeDirection(eMeshDirection dir)
  {
    ARCANE_CHECK_POINTER(m_patch);
    return m_patch->nodeDirection(dir);
  }

  //! List of nodes in direction \a dir (0, 1 or 2)
  NodeDirectionMng& nodeDirection(Integer idir)
  {
    ARCANE_CHECK_POINTER(m_patch);
    return m_patch->nodeDirection(idir);
  }

  //! Performs checks on the instance validity.
  void checkValid() const
  {
    ARCANE_CHECK_POINTER(m_patch);
    m_patch->checkValid();
  }

  /*!
   * \brief Method to retrieve the patch position in the
   * Cartesian mesh.
   *
   * \return A copy of the position.
   */
  AMRPatchPosition position() const
  {
    ARCANE_CHECK_POINTER(m_patch);
    return m_patch->position();
  }

  //! Indicates if the patch is null.
  bool isNull() const { return !m_patch; }

  //! Interface associated with the patch (for compatibility with existing code)
  ICartesianMeshPatch* patchInterface() const { return m_patch; }

 private:

  ICartesianMeshPatch* m_patch = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
