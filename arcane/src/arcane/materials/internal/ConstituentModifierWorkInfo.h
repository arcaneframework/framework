// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentModifierWorkInfo.h                               (C) 2000-2024 */
/*                                                                           */
/* Working structure used during the modification of constituents.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_CONSTITUENTMODIFIERWORKINFO_H
#define ARCANE_MATERIALS_INTERNAL_CONSTITUENTMODIFIERWORKINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/DualUniqueArray.h"

#include "arcane/core/materials/internal/IMeshMaterialVariableInternal.h"
#include "arcane/materials/MaterialsGlobal.h"
#include "arcane/materials/internal/MeshMaterialVariableIndexer.h"
#include "arcane/materials/internal/ComponentItemListBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Working structure used during the modification
 * of constituents (via MeshMaterialModifier).
 *
 * Instances of this class are maintained throughout an entire phase of
 * modification involving several material modification operations
 * (MaterialModifierOperation).
 *
 * initialize() must be called before using the instance.
 */
class ARCANE_MATERIALS_EXPORT ConstituentModifierWorkInfo
{
 public:

  ConstituentModifierWorkInfo(const MemoryAllocationOptions& opts, eMemoryRessource mem);

 public:

  /*!
   * \brief List of pure mesh elements of a constituent
   * added/removed by the current operation.
   */
  DualUniqueArray<Int32> pure_local_ids;
  /*!
   * \brief List of partial mesh elements of a constituent
   * added/removed by the current operation.
   */
  DualUniqueArray<Int32> partial_indexes;
  /*!
   * \brief Array dimensioned by materials, which is true if a material is
   * affected by the current modification.
   */
  DualUniqueArray<bool> m_is_materials_modified;
  /*!
   * \brief Array dimensioned by environments, which is true if an environment is
   * affected by the current modification.
   */
  DualUniqueArray<bool> m_is_environments_modified;
  bool is_verbose = false;

  //! List of mesh elements of an environment that will be added or removed during an operation
  UniqueArray<Int32> cells_changed_in_env;
  //! List of mesh elements of an environment that are already present in an environment during an operation
  UniqueArray<Int32> cells_unchanged_in_env;

  //! List of MatVarIndex and LocalId to save when deleting material mesh elements
  DualUniqueArray<MatVarIndex> m_saved_matvar_indexes;
  DualUniqueArray<Int32> m_saved_local_ids;

  //! Number of materials for the environment currently being evaluated
  UniqueArray<Int16> m_cells_current_nb_material;

  // Filter indicating if a mesh element will be partial after addition.
  // This array is dimensioned by the number of mesh elements added during the current transformation.
  NumArray<bool, MDDim1> m_cells_is_partial;

  ComponentItemListBuilder list_builder;

  //! Information for copies between partial and global values.
  UniqueArray<CopyBetweenDataInfo> m_host_variables_copy_data;

  //! Information for copies between partial and global values.
  NumArray<CopyBetweenDataInfo, MDDim1> m_variables_copy_data;

 public:

  //! Initializes the instance.
  void initialize(Int32 max_local_id, Int32 nb_material, Int32 nb_environment, RunQueue& queue);

 public:

  //! Indicates if the mesh element \a local_id is transformed during the current operation.
  bool isTransformedCell(CellLocalId local_id) const
  {
    return m_cells_to_transform[local_id.localId()];
  }

  //! Sets the transformation status of the mesh element \a local_id for the current operation
  void setTransformedCell(CellLocalId local_id, bool v)
  {
    m_cells_to_transform[local_id.localId()] = v;
  }

  //! Sets the transformation status of the mesh element \a local_id for the current operation
  void resetTransformedCells(ConstArrayView<Int32> local_ids)
  {
    for (Int32 x : local_ids)
      m_cells_to_transform[x] = false;
  }
  //! Indicates if the mesh element \a local_id is removed from the material for the current operation.
  bool isRemovedCell(Int32 local_id) const { return m_removed_local_ids_filter[local_id]; }

  //! Sets the 'Removed' status of the mesh elements \a local_ids to \a value
  void setRemovedCells(ConstArrayView<Int32> local_ids, bool value);

  //! Sets the current operation
  void setCurrentOperation(MaterialModifierOperation* operation);

  //! Indicates if the current operation is an addition (true) or a removal (false) of mesh elements
  bool isAdd() const { return m_is_add; }

  SmallSpan<const bool> transformedCells() const { return m_cells_to_transform.to1DSmallSpan(); }
  SmallSpan<bool> transformedCells() { return m_cells_to_transform.to1DSmallSpan(); }
  SmallSpan<const bool> removedCells() const { return m_removed_local_ids_filter.to1DSmallSpan(); }
  SmallSpan<bool> removedCells() { return m_removed_local_ids_filter.to1DSmallSpan(); }

 private:

  //! Filter indicating the mesh elements that are removed from the constituent
  // This array is dimensioned by the number of mesh elements.
  NumArray<bool, MDDim1> m_removed_local_ids_filter;

  //! Filter indicating the mesh elements that must change status (Pure<->Partial)
  // This array is dimensioned by the number of mesh elements.
  NumArray<bool, MDDim1> m_cells_to_transform;

  bool m_is_add = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
