// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentConnectivityList.h                               (C) 2000-2024 */
/*                                                                           */
/* Management of constituent connectivity lists.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_COMPONENTCONNECTIVITYLIST_H
#define ARCANE_MATERIALS_INTERNAL_COMPONENTCONNECTIVITYLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/DualUniqueArray.h"

#include "arcane/core/VariableTypes.h"
#include "arcane/core/IIncrementalItemConnectivity.h"

#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Management of constituent connectivity lists.
 */
class ConstituentConnectivityList
: public TraceAccessor
, public ReferenceCounterImpl
, public IIncrementalItemSourceConnectivity
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

  class ConstituentContainer;
  class Container;
  class NumberOfMaterialComputer;

 public:

  explicit ConstituentConnectivityList(MeshMaterialMng* mm);
  ~ConstituentConnectivityList();

 public:

  ConstituentConnectivityList(ConstituentConnectivityList&&) = delete;
  ConstituentConnectivityList(const ConstituentConnectivityList&) = delete;
  ConstituentConnectivityList& operator=(ConstituentConnectivityList&&) = delete;
  ConstituentConnectivityList& operator=(const ConstituentConnectivityList&) = delete;

 public:

  void endCreate(bool is_continue);

  void addCellsToEnvironment(Int16 env_id, SmallSpan<const Int32> cell_ids, RunQueue& queue);
  void removeCellsToEnvironment(Int16 env_id, SmallSpan<const Int32> cell_ids, RunQueue& queue);

  void addCellsToMaterial(Int16 mat_id, SmallSpan<const Int32> cell_ids, RunQueue& queue);
  void removeCellsToMaterial(Int16 mat_id, SmallSpan<const Int32> cell_ids, RunQueue& queue);

  //! Arrays of the total number of environments per mesh element (indexed by localId())
  ConstArrayView<Int16> cellsNbEnvironment() const;
  //! Arrays of the total number of materials per mesh element (indexed by localId())
  ConstArrayView<Int16> cellsNbMaterial() const;

  //! Number of materials of mesh element \a cell_id for environment index \a env_id
  Int16 cellNbMaterial(CellLocalId cell_id, Int16 env_id) const;

  //! Removes all connected entities
  void removeAllConnectivities();

  /*!
   * \brief Fills \a cells_nb_material with the number of materials of
   * environment \a env_id
   */
  void fillCellsNbMaterial(SmallSpan<const Int32> cells_local_id, Int16 env_id,
                           SmallSpan<Int16> cells_nb_material, RunQueue& queue);

  /*!
   * \brief Fills \a cells_do_transform indicating if the mesh transitions from
   * pure to partial.
   *
   * Returns the number of transformed meshes
   */
  Int32 fillCellsToTransform(SmallSpan<const Int32> cells_local_id, Int16 env_id,
                             SmallSpan<bool> cells_do_transform, bool is_add, RunQueue& queue);

  /*!
   * \brief Fills \a cells_is_partial indicating if the mesh is partial for
   * environment \a env_id
   */
  void fillCellsIsPartial(SmallSpan<const Int32> cells_local_id, Int16 env_id,
                          SmallSpan<bool> cells_is_partial, RunQueue& queue);

  void fillModifiedConstituents(SmallSpan<const Int32> cells_local_id,
                                SmallSpan<bool> is_modified_materials,
                                SmallSpan<bool> is_modified_environments,
                                int modified_mat_id, bool is_add, const RunQueue& queue);

  void printConstituents(SmallSpan<const Int32> cells_local_id) const;

  /*!
   * \brief Indicates if the instance is active.
   *
   * Valid only after calling endCreate().
   * If the instance is not active, none of the methods that modify the constituent
   * list should be called.
   */
  bool isActive() const { return m_is_active; }

 public:

  // Implémentation de IIncrementalItemSourceConnectivity
  //@{
  IItemFamily* sourceFamily() const override { return m_cell_family; }
  void notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids) override;
  void notifySourceItemAdded(ItemLocalId item) override;
  void reserveMemoryForNbSourceItems(Int32 n, bool pre_alloc_connectivity) override;
  void notifyReadFromDump() override;
  Ref<IIncrementalItemSourceConnectivity> toSourceReference() override;
  //@}

 private:

  MeshMaterialMng* m_material_mng = nullptr;
  IItemFamily* m_cell_family = nullptr;
  Container* m_container = nullptr;

  //! Index of the environment to which a material belongs
  DualUniqueArray<Int16> m_environment_for_materials;
  bool m_is_active = false;
  bool m_is_force_transform_all_constituants = false;

 public:

  void _addCells(Int16 env_id, SmallSpan<const Int32> cell_ids,
                 ConstituentContainer& component, RunQueue& queue);
  void _removeCells(Int16 env_id, SmallSpan<const Int32> cell_ids,
                    ConstituentContainer& component, RunQueue& queue);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
