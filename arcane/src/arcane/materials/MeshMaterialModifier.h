// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialModifier.h                                      (C) 2000-2024 */
/*                                                                           */
/* Object allowing material modification.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALMODIFIER_H
#define ARCANE_MATERIALS_MESHMATERIALMODIFIER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Object allowing modification of materials or media.
 *
 * This class provides methods to modify the list of
 * cells composing a material or a medium.
 *
 * Modifications are made directly on the materials. The corresponding media
 * are automatically updated. It is possible
 * either to add cells to a material (addCells())
 * or to remove them (removeCells()). The modifications are not
 * taken into account until endUpdate() is called. This last method
 * does not need to be called explicitly: it is called automatically
 * when the destructor is called.
 * \todo add example.
 */
class ARCANE_MATERIALS_EXPORT MeshMaterialModifier
{
 public:

  explicit MeshMaterialModifier(IMeshMaterialMng*);
  ~MeshMaterialModifier() ARCANE_NOEXCEPT_FALSE;

 public:

  /*!
   * \brief Adds the cells with local indices \a ids to the material \a mat.
   */
  void addCells(IMeshMaterial* mat, SmallSpan<const Int32> ids);

  /*!
   * \brief Removes the cells with local indices \a ids from the material \a mat.
   */
  void removeCells(IMeshMaterial* mat, SmallSpan<const Int32> ids);

  /*!
   * \brief Updates the structures after a modification.
   *
   * This method is automatically called in the instance's destructor if necessary.
   */
  void endUpdate();

  /*!
   * \brief Indicates whether values are copied between pure and partial states when
   * a change in the cell state occurs.
   *
   * If true (the default), the partial value
   * of a cell is copied into the pure value when the partial cell
   * transitions to a pure cell.
   */
  void setDoCopyBetweenPartialAndPure(bool v);

  /*!
   * \brief Indicates whether newly created material or environment cells are initialized.
   *
   * If true (the default), the newly created constituent cells
   * are initialized. The value used for initialization depends on
   * IMeshMaterialMng::isDataInitialisationWithZero().
   */
  void setDoInitNewItems(bool v);

  /*!
   * \brief Indicates whether work buffers are preserved between modifications.
   *
   * If true (the default), the work buffers are preserved between
   * instances of this class. This prevents these buffers from being reallocated between
   * each use of this class, but consequently, memory consumption may increase.
   */
  void setPersistantWorkBuffer(bool v);

 private:

  MeshMaterialModifierImpl* m_impl = nullptr;
  bool m_has_update = false;
  void _checkHasUpdate();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
