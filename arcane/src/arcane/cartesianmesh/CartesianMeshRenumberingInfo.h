// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshRenumberingInfo.h                              (C) 2000-2024 */
/*                                                                           */
/* Information for renumbering.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHRENUMBERINGINFO_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHRENUMBERINGINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/cartesianmesh/CartesianPatch.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*!
 * \brief Information for renumbering.
 *
 * If both the uniqueId() of the faces and the entities of the
 * patches are renumbered, the face renumbering happens first.
 */
class ARCANE_CARTESIANMESH_EXPORT CartesianMeshRenumberingInfo
{
 public:

  /*!
   * \brief Method to renumber patches.
   *
   * The possible values are as follows:
   * - 0 no renumbering.
   * - 1 default renumbering.
   * - 2 experimental version of renumbering
   * - 3 like version 1 but with a different implementation
   *   for 2D meshes.
   * - 4 like 1, but uses the same numbering with or without de-refinement
   *   of the initial mesh.
   *
   * If renumbering occurs, the uniqueId() of the entities (Node,Face,Cell)
   * of the patches are renumbered to have the same numbering regardless
   * of the subdivision.
   * The numbering is not contiguous. Only the child entities
   * of the parentPatch() entities are renumbered.
   */
  void setRenumberPatchMethod(Int32 v) { m_renumber_patch_method = v; }
  Int32 renumberPatchMethod() const { return m_renumber_patch_method; }

  /*!
   * \brief Method to renumber faces.
   *
   * If 0, there is no renumbering. The only other valid value is 1.
   * In this case, the renumbering is based on a Cartesian numbering.
   */
  void setRenumberFaceMethod(Int32 v) { m_renumber_faces_method = v; }
  Int32 renumberFaceMethod() const { return m_renumber_faces_method; }

  /*!
   * \brief Indicates whether to retrieve the entities after renumbering.
   *
   * The sort calls IItemFamily::compactItems(true) for each family.
   * This also causes a call to ICartesianMesh::computeDirections()
   * to recalculate the directions that are invalidated following the sort.
   */
  void setSortAfterRenumbering(bool v) { m_is_sort = v; }
  bool isSortAfterRenumbering() const { return m_is_sort; }

  /*!
   * \brief Parent patch number for renumbering.
   *
   * Parent patch for renumbering. For renumbering, the child meshes of this patch
   * are recursively traversed and renumbered, as well as the entities
   * associated with these meshes (nodes and faces).
   *
   * The entities of this patch (meshes, nodes, and faces) are not renumbered.
   *
   * If not specified, the implementation will use patch 0 as the parent patch.
   *
   * \note This property is only considered if renumberPatchMethod()==1.
   */
  void setParentPatch(CartesianPatch patch) { m_parent_patch = patch; }
  CartesianPatch parentPatch() const { return m_parent_patch; }

 private:

  Int32 m_renumber_patch_method = 0;
  Int32 m_renumber_faces_method = 0;
  CartesianPatch m_parent_patch;
  bool m_is_sort = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
