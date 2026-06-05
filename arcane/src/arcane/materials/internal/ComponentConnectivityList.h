// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentConnectivityList.h                               (C) 2000-2023 */
/*                                                                           */
/* Management of constituent connectivity lists.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_CONSTITUENTCONNECTIVITYLIST_H
#define ARCANE_MATERIALS_INTERNAL_CONSTITUENTCONNECTIVITYLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Ref.h"

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

  class ConstituantContainer;
  class Container;

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

  void addCellsToEnvironment(Int16 env_id, ConstArrayView<Int32> cell_ids);
  void removeCellsToEnvironment(Int16 env_id, ConstArrayView<Int32> cell_ids);

  void addCellsToMaterial(Int16 mat_id, ConstArrayView<Int32> cell_ids);
  void removeCellsToMaterial(Int16 mat_id, ConstArrayView<Int32> cell_ids);

  ConstArrayView<Int16> cellsNbEnvironment() const;
  ConstArrayView<Int16> cellsNbMaterial() const;

  //! Number of materials of the mesh \a cell_id for environment index \a env_id
  Int16 cellNbMaterial(CellLocalId cell_id, Int16 env_id);

 public:

  // Implementation of IIncrementalItemSourceConnectivity
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
  UniqueArray<Int16> m_environment_for_materials;

 private:

  void _addCells(Int16 env_id, ConstArrayView<Int32> cell_ids, ComponentContainer& component);
  void _removeCells(Int16 env_id, ConstArrayView<Int32> cell_ids, ComponentContainer& component);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
