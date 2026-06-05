// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialIndirectModifier.h                              (C) 2000-2022 */
/*                                                                           */
/* Object allowing indirect modification of materials.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALINDIRECTMODIFIER_H
#define ARCANE_MATERIALS_MESHMATERIALINDIRECTMODIFIER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IMesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
class IMeshMaterialMng;
class MeshMaterialBackup;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Object allowing indirect modification of materials or media.
 */
class ARCANE_MATERIALS_EXPORT MeshMaterialIndirectModifier
{
 public:

  MeshMaterialIndirectModifier(IMeshMaterialMng*);
  ~MeshMaterialIndirectModifier() noexcept(false);

 public:

  MeshMaterialIndirectModifier(const MeshMaterialIndirectModifier&) = default;
  MeshMaterialIndirectModifier(MeshMaterialIndirectModifier&&) = default;
  MeshMaterialIndirectModifier& operator=(const MeshMaterialIndirectModifier&) = default;
  MeshMaterialIndirectModifier& operator=(MeshMaterialIndirectModifier&&) = default;

 public:

  /*!
   * \brief Prepares an update.
   *
   */
  void beginUpdate();

  /*!
   * \brief Updates the structures after a modification.
   *
   * This method is automatically called in the destructor of
   * the instance if necessary.
   */
  void endUpdate();

  /*!
   * \brief Updates the structures after a modification with sorting
   * of media and material groups.
   *
   * This method is identical to endUpdate() but ensures that
   * the groups associated with the components (IMeshComponent::cells())
   * will be sorted by increasing uniqueId() at the end of the update.
   */
  void endUpdateWithSort();

 private:

  IMeshMaterialMng* m_material_mng;
  MeshMaterialBackup* m_backup;
  bool m_has_update;

 private:

  void _endUpdate(bool do_sort);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
