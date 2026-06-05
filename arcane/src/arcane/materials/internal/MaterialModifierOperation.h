// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MaterialModifierOperation.h                                 (C) 2000-2024 */
/*                                                                           */
/* Operation to add/remove meshes from a material.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_MATERIALMODIFIEROPERATION_H
#define ARCANE_MATERIALS_INTERNAL_MATERIALMODIFIEROPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/NumArray.h"

#include "arcane/materials/MaterialsGlobal.h"
#include "arcane/materials/internal/MeshMaterialModifierImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Operation to add or remove meshes from a material.
 */
class MaterialModifierOperation
{
 public:

  MaterialModifierOperation();

 private:

  MaterialModifierOperation(IMeshMaterial* mat, SmallSpan<const Int32> ids, bool is_add);

 public:

  static MaterialModifierOperation* createAdd(IMeshMaterial* mat, SmallSpan<const Int32> ids)
  {
    return new MaterialModifierOperation(mat, ids, true);
  }
  static MaterialModifierOperation* createRemove(IMeshMaterial* mat, SmallSpan<const Int32> ids)
  {
    return new MaterialModifierOperation(mat, ids, false);
  }

 public:

  //! Indicates whether the operation is to add or remove meshes from the material
  bool isAdd() const { return m_is_add; }

  //! The material for which meshes are to be added/removed
  IMeshMaterial* material() const { return m_mat; }

  //! List of localId() of meshes to add/remove
  SmallSpan<const Int32> ids() const { return m_ids.view(); }

 public:

  /*!
   * \brief Filters the mesh IDs.
   *
   * If isAdd() is true, filters the IDs to remove those that are already in
   * the material. If isAdd() is false, filters the IDs to remove those that
   * are not in the material.
   *
   * These operations are costly because all present entities must be traversed.
   * Therefore, this method should generally only be used in verification mode.
   */
  void filterIds();

 private:

  static void _filterValidIds(MaterialModifierOperation* operation, Int32Array& valid_ids);
  static Int32 _checkMaterialPresence(MaterialModifierOperation* operation);

 private:

  IMeshMaterial* m_mat = nullptr;
  bool m_is_add = false;
  UniqueArray<Int32> m_ids;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
