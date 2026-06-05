// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentConnectivityList.cc                              (C) 2000-2024 */
/*                                                                           */
/* Management of constituent connectivity lists.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/ConstituentConnectivityList.h"

#include "arcane/core/IItemFamily.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/core/MeshUtils.h"
#include "arcane/core/internal/IDataInternal.h"
#include "arcane/core/internal/IItemFamilyInternal.h"

#include "arcane/materials/internal/MeshMaterialMng.h"

#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/Scan.h"
#include "arcane/accelerator/Reduce.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  // This method is the same as MeshUtils::removeItemAndKeepOrder().
  // It would be necessary to merge the two.
  // NOTE: With C++20, we should be able to remove this method and
  // use std::erase()

  template <typename DataType>
  ARCCORE_HOST_DEVICE void _removeValueAndKeepOrder(ArrayView<DataType> values, DataType value_to_remove)
  {
    Integer n = values.size();
#ifndef ARCCORE_DEVICE_CODE
    if (n <= 0)
      ARCANE_FATAL("Can not remove item lid={0} because list is empty", value_to_remove);
#endif
    --n;
    if (n == 0) {
      if (values[0] == value_to_remove)
        return;
    }
    else {
      // If the element is the last one, do nothing.
      if (values[n] == value_to_remove)
        return;
      for (Integer i = 0; i < n; ++i) {
        if (values[i] == value_to_remove) {
          for (Integer z = i; z < n; ++z)
            values[z] = values[z + 1];
          return;
        }
      }
    }
#if defined(ARCCORE_DEVICE_CODE)
    // With Intel DPC++ 2024.1, using this function with the
    // CUDA backend causes an error in the generated assembly.
#  if !defined(__INTEL_LLVM_COMPILER)
    assert(false);
#  endif
#else
    ARCANE_FATAL("No value to remove '{0}' found in list {1}", value_to_remove, values);
#endif
  }

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Container for connectivity data for a constituent type.
 */
class ConstituentConnectivityList::ConstituentContainer
{
 public:

  /*!
   * \brief View onto a list of mesh constituents.
   */
  class View
  {
   public:

    explicit View(ConstituentContainer& c)
    : nb_components_view(c.m_nb_component_as_array.view())
    , component_indexes_view(c.m_component_index_as_array.view())
    , component_list_view(c.m_component_list_as_array.view())
    {
    }

   public:

    ARCCORE_HOST_DEVICE SmallSpan<const Int16> components(Int32 item_lid) const
    {
      Int16 n = nb_components_view[item_lid];
      Int32 index = component_indexes_view[item_lid];
      return component_list_view.subPart(index, n);
    }

   private:

    SmallSpan<Int16> nb_components_view;
    SmallSpan<Int32> component_indexes_view;
    SmallSpan<Int16> component_list_view;
  };

 public:

  ConstituentContainer(const MeshHandle& mesh, const String& var_base_name)
  : m_nb_component(VariableBuildInfo(mesh, var_base_name + "NbComponent", IVariable::PPrivate))
  , m_component_index(VariableBuildInfo(mesh, var_base_name + "Index", IVariable::PPrivate))
  , m_component_list(VariableBuildInfo(mesh, var_base_name + "List", IVariable::PPrivate))
  , m_nb_component_as_array(m_nb_component._internalTrueData()->_internalDeprecatedValue())
  , m_component_index_as_array(m_component_index._internalTrueData()->_internalDeprecatedValue())
  , m_component_list_as_array(m_component_list._internalTrueData()->_internalDeprecatedValue())
  {
  }

 public:

  void endCreate(bool is_continue)
  {
    if (!is_continue)
      _resetConnectivities();
  }

  ArrayView<Int16> components(CellLocalId item_lid)
  {
    Int16 n = m_nb_component[item_lid];
    Int32 index = m_component_index[item_lid];
    //Int32 list_size = m_component_list.size();
    //std::cout << "CELL=" << item_lid << " nb_mat=" << n
    //          << " index=" << index << " list_size=" << list_size << "\n";
    return m_component_list_as_array.subView(index, n);
  }

  void checkResize(Int64 size)
  {
    if (MeshUtils::checkResizeArray(m_nb_component_as_array, size, false))
      m_nb_component.updateFromInternal();
    if (MeshUtils::checkResizeArray(m_component_index_as_array, size, false))
      m_component_index.updateFromInternal();
  }

  void reserve(Int64 capacity)
  {
    m_nb_component_as_array.reserve(capacity);
    m_component_index_as_array.reserve(capacity);
  }

  void changeLocalIds(Int32ConstArrayView new_to_old_ids)
  {
    m_nb_component.variable()->compact(new_to_old_ids);
    m_component_index.variable()->compact(new_to_old_ids);
  }

  void notifyUpdateConnectivityList()
  {
    m_component_list.updateFromInternal();
  }

  void removeAllConnectivities()
  {
    _resetConnectivities();
  }

 private:

  //! Number of environments per cell (sized according to the number of cells)
  VariableArrayInt16 m_nb_component;
  //! Index in \a m_component_list (Sized according to the number of cells)
  VariableArrayInt32 m_component_index;
  //! List of constituents
  VariableArrayInt16 m_component_list;

 public:

  VariableArrayInt16::ContainerType& m_nb_component_as_array;
  VariableArrayInt32::ContainerType& m_component_index_as_array;
  VariableArrayInt16::ContainerType& m_component_list_as_array;

 private:

  void _resetConnectivities()
  {
    m_nb_component.fill(0);
    m_component_index.fill(0);
    // The first element of the list is used for empty constituents
    m_component_list.resize(1);
    m_component_list[0] = 0;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class for calculating the number of materials in an environment.
 */
class ConstituentConnectivityList::NumberOfMaterialComputer
{
 public:

  NumberOfMaterialComputer(ConstituentContainer::View view,
                           SmallSpan<const Int16> environment_for_materials)
  : m_view(view)
  , m_environment_for_materials(environment_for_materials)
  {
  }

 public:

  ARCCORE_HOST_DEVICE Int16 cellNbMaterial(Int32 cell_local_id, Int16 env_id) const
  {
    auto mats = m_view.components(cell_local_id);
    Int16 nb_mat = 0;
    for (Int16 mat_id : mats) {
      Int16 current_id = m_environment_for_materials[mat_id];
      if (current_id == env_id)
        ++nb_mat;
    }
    return nb_mat;
  }

 private:

  ConstituentContainer::View m_view;
  //! View indicating the environment associated with the materials
  SmallSpan<const Int16> m_environment_for_materials;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ConstituentConnectivityList::Container
{
 public:

  Container(const MeshHandle& mesh, const String& var_base_name)
  : m_environment(mesh, var_base_name + String("ComponentEnvironment"))
  , m_material(mesh, var_base_name + String("ComponentMaterial"))
  {
  }

 public:

  void checkResize(Int32 lid)
  {
    Int64 wanted_size = lid + 1;
    m_environment.checkResize(wanted_size);
    m_material.checkResize(wanted_size);
  }

  void changeLocalIds(Int32ConstArrayView new_to_old_ids)
  {
    m_environment.changeLocalIds(new_to_old_ids);
    m_material.changeLocalIds(new_to_old_ids);
  }

  void reserve(Int64 capacity)
  {
    m_environment.reserve(capacity);
    m_material.reserve(capacity);
  }

 public:

  ConstituentContainer m_environment;
  ConstituentContainer m_material;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstituentConnectivityList::
ConstituentConnectivityList(MeshMaterialMng* mm)
: TraceAccessor(mm->traceMng())
, m_material_mng(mm)
, m_container(new Container(mm->meshHandle(), String("ComponentEnviroment") + mm->name()))
{
  // Indicates if modification is forced in fillModifiedConstituents()
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_MATERIAL_FORCE_TRANSFORM", true)) {
    m_is_force_transform_all_constituants = (v.value() != 0);
    info() << "Force transformation in 'ConstituentConnectivityList' v=" << m_is_force_transform_all_constituants;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstituentConnectivityList::
~ConstituentConnectivityList()
{
  delete m_container;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
endCreate(bool is_continue)
{
  // Now (February 2024) we always build incremental connectivities
  const bool always_build_connectivity = true;

  // Registers with the family to be notified of changes
  // but only if incremental modification support was requested
  // to avoid unnecessarily consuming memory.
  // Eventually, we will do it all the time
  m_cell_family = m_material_mng->mesh()->cellFamily();
  {
    int opt_flag_value = m_material_mng->modificationFlags();
    bool use_incremental = (opt_flag_value & (int)eModificationFlags::IncrementalRecompute) != 0;
    if (use_incremental || always_build_connectivity) {
      m_cell_family->_internalApi()->addSourceConnectivity(this);
      m_is_active = true;
      info() << "Activating incremental material connectivities";
    }
  }
  if (!is_continue) {
    Int32 max_local_id = m_cell_family->maxLocalId();
    m_container->checkResize(max_local_id + 1);
    m_container->m_environment.endCreate(is_continue);
    m_container->m_material.endCreate(is_continue);
  }

  // Fills an array indicating the index
  // of the corresponding environment for each material index.
  {
    ConstArrayView<MeshMaterial*> materials = m_material_mng->trueMaterials();
    const Int32 nb_mat = materials.size();
    auto environment_for_materials = m_environment_for_materials.hostModifier();
    environment_for_materials.resize(nb_mat);
    auto local_view = environment_for_materials.view();
    for (Int32 i = 0; i < nb_mat; ++i)
      local_view[i] = materials[i]->trueEnvironment()->componentId();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
_addCells(Int16 component_id, SmallSpan<const Int32> cells_local_id,
          ConstituentContainer& component, RunQueue& queue)
{
  const Int32 nb_item = cells_local_id.size();
  if (nb_item == 0)
    return;
  Array<Int16>& nb_component = component.m_nb_component_as_array;
  Array<Int32>& component_index = component.m_component_index_as_array;
  Array<Int16>& component_list = component.m_component_list_as_array;

  SmallSpan<Int16> nb_component_view = component.m_nb_component_as_array.view();

  // TODO: Use persistent working arrays.
  NumArray<Int32, MDDim1> new_indexes(nb_item, queue.memoryRessource());
  // To copy the number of elements to add from the device to the CPU
  NumArray<Int32, MDDim1> new_indexes_to_add(1, eMemoryRessource::HostPinned);

  // Calculates the index of the new elements
  {
    Accelerator::GenericScanner scanner(queue);
    SmallSpan<Int32> new_indexes_view = new_indexes;
    SmallSpan<Int32> new_indexes_to_add_view = new_indexes_to_add;
    auto getter = [=] ARCCORE_HOST_DEVICE(Int32 index) -> Int32 {
      return 1 + nb_component_view[cells_local_id[index]];
    };
    auto setter = [=] ARCCORE_HOST_DEVICE(Int32 index, Int32 value) {
      new_indexes_view[index] = value;
      if (index == (nb_item - 1))
        new_indexes_to_add_view[0] = new_indexes_view[index] + nb_component_view[cells_local_id[index]] + 1;
    };
    Accelerator::ScannerSumOperator<Int32> op;
    scanner.applyWithIndexExclusive(nb_item, 0, getter, setter, op, A_FUNCINFO);
  }
  queue.barrier();

  const Int32 nb_indexes_to_add = new_indexes_to_add[0];
  const Int32 current_list_index = component_list.size();

  MemoryUtils::checkResizeArrayWithCapacity(component_list, current_list_index + nb_indexes_to_add, false);

  {
    auto command = makeCommand(queue);
    SmallSpan<Int16> nb_component_view = nb_component.view();
    SmallSpan<Int32> component_index_view = component_index.view();
    SmallSpan<Int16> component_list_view = component_list.view();
    SmallSpan<const Int32> new_indexes_view = new_indexes;
    command << RUNCOMMAND_LOOP1(iter, nb_item)
    {
      auto [i] = iter();
      Int32 cell_id = cells_local_id[i];
      const Int16 n = nb_component_view[cell_id];
      Int32 new_pos = current_list_index + new_indexes_view[i];
      if (n != 0) {
        // Copies the old values.
        // TODO: This leaves holes in the list that must be removed
        // via compaction.
        Int32 current_pos = component_index_view[cell_id];
        SmallSpan<const Int16> current_values(&component_list_view[current_pos], n);
        SmallSpan<Int16> new_values(&component_list_view[new_pos], n);
        new_values.copy(current_values);
      }
      component_index_view[cell_id] = new_pos;
      component_list_view[new_pos + n] = component_id;
      ++nb_component_view[cell_id];
    };
  }

  component.notifyUpdateConnectivityList();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
_removeCells(Int16 component_id, SmallSpan<const Int32> cells_local_id,
             ConstituentContainer& component, RunQueue& queue)
{
  SmallSpan<Int16> nb_component = component.m_nb_component_as_array.view();
  SmallSpan<Int32> component_index = component.m_component_index_as_array.view();
  SmallSpan<Int16> component_list = component.m_component_list_as_array.view();

  const Int32 n = cells_local_id.size();
  auto command = makeCommand(queue);
  command << RUNCOMMAND_LOOP1(iter, n)
  {
    auto [i] = iter();
    Int32 id = cells_local_id[i];
    //for (Int32 id : cell_ids) {
    CellLocalId cell_id(id);
    const Int32 current_pos = component_index[cell_id];
    const Int32 n = nb_component[cell_id];
    ArrayView<Int16> current_values(n, &component_list[current_pos]);
    // Removes the deleted middle element from the list
    _removeValueAndKeepOrder(current_values, component_id);
    // Sets an invalid value to indicate that the location is free
    current_values[n - 1] = (-1);
    --nb_component[cell_id];
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
addCellsToEnvironment(Int16 env_id, SmallSpan<const Int32> cell_ids, RunQueue& queue)
{
  _addCells(env_id, cell_ids, m_container->m_environment, queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
removeCellsToEnvironment(Int16 env_id, SmallSpan<const Int32> cell_ids, RunQueue& queue)
{
  _removeCells(env_id, cell_ids, m_container->m_environment, queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
addCellsToMaterial(Int16 mat_id, SmallSpan<const Int32> cell_ids, RunQueue& queue)
{
  _addCells(mat_id, cell_ids, m_container->m_material, queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
removeCellsToMaterial(Int16 mat_id, SmallSpan<const Int32> cell_ids, RunQueue& queue)
{
  _removeCells(mat_id, cell_ids, m_container->m_material, queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int16> ConstituentConnectivityList::
cellsNbEnvironment() const
{
  return m_container->m_environment.m_nb_component_as_array;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int16> ConstituentConnectivityList::
cellsNbMaterial() const
{
  return m_container->m_material.m_nb_component_as_array;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int16 ConstituentConnectivityList::
cellNbMaterial(CellLocalId cell_id, Int16 env_id) const
{
  auto environment_for_materials = m_environment_for_materials.hostView();
  Int16 nb_mat = 0;
  ArrayView<Int16> mats = m_container->m_material.components(cell_id);
  for (Int16 mat_id : mats) {
    Int16 current_id = environment_for_materials[mat_id];
    if (current_id == env_id)
      ++nb_mat;
  }
  return nb_mat;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
fillCellsNbMaterial(SmallSpan<const Int32> cells_local_id, Int16 env_id,
                    SmallSpan<Int16> cells_nb_material, RunQueue& queue)
{
  ConstituentContainer::View materials_container_view(m_container->m_material);
  bool is_device = isAcceleratorPolicy(queue.executionPolicy());
  auto environment_for_materials = m_environment_for_materials.view(is_device);
  const Int32 n = cells_local_id.size();
  auto command = makeCommand(queue);
  command << RUNCOMMAND_LOOP1(iter, n)
  {
    auto [i] = iter();
    Int32 cell_id = cells_local_id[i];
    Int16 nb_mat = 0;
    SmallSpan<const Int16> mats = materials_container_view.components(cell_id);
    for (Int16 mat_id : mats) {
      Int16 current_id = environment_for_materials[mat_id];
      if (current_id == env_id)
        ++nb_mat;
    }
    cells_nb_material[i] = nb_mat;
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ConstituentConnectivityList::
fillCellsToTransform(SmallSpan<const Int32> cells_local_id, Int16 env_id,
                     SmallSpan<bool> cells_do_transform, bool is_add, RunQueue& queue)
{
  ConstituentContainer::View materials_container_view(m_container->m_material);
  bool is_device = queue.isAcceleratorPolicy();
  auto environment_for_materials = m_environment_for_materials.view(is_device);

  NumberOfMaterialComputer nb_mat_computer(materials_container_view, environment_for_materials);
  SmallSpan<const Int16> cells_nb_env = cellsNbEnvironment();
  const Int32 n = cells_local_id.size();
  auto command = makeCommand(queue);
  Accelerator::ReducerSum2<Int32> sum_transformed(command);
  command << RUNCOMMAND_LOOP1(iter, n, sum_transformed)
  {
    auto [i] = iter();
    Int32 local_id = cells_local_id[i];
    bool do_transform = false;
    // In case of addition, we switch from pure to partial if there are multiple environments or
    // multiple materials in the environment.
    // In case of removal, we switch from partial to pure if we are the only material
    // and the only environment.
    const Int16 nb_env = cells_nb_env[local_id];
    if (is_add) {
      do_transform = (nb_env > 1);
      if (!do_transform)
        do_transform = nb_mat_computer.cellNbMaterial(local_id, env_id) > 1;
    }
    else {
      do_transform = (nb_env == 1);
      if (do_transform)
        do_transform = nb_mat_computer.cellNbMaterial(local_id, env_id) == 1;
    }
    if (do_transform) {
      cells_do_transform[local_id] = do_transform;
      sum_transformed.combine(1);
    }
  };
  return sum_transformed.reducedValue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
fillCellsIsPartial(SmallSpan<const Int32> cells_local_id, Int16 env_id,
                   SmallSpan<bool> cells_is_partial, RunQueue& queue)
{
  ConstituentContainer::View materials_container_view(m_container->m_material);
  bool is_device = queue.isAcceleratorPolicy();
  auto environment_for_materials = m_environment_for_materials.view(is_device);
  NumberOfMaterialComputer nb_mat_computer(materials_container_view, environment_for_materials);
  SmallSpan<const Int16> cells_nb_env = cellsNbEnvironment();
  const Int32 n = cells_local_id.size();
  auto command = makeCommand(queue);

  command << RUNCOMMAND_LOOP1(iter, n)
  {
    auto [i] = iter();
    Int32 local_id = cells_local_id[i];
    // We only take the global index if we are the only material and the only
    // environment in the mesh. Otherwise, we take a multiple index
    bool is_partial = (cells_nb_env[local_id] > 1 || nb_mat_computer.cellNbMaterial(local_id, env_id) > 1);
    cells_is_partial[i] = is_partial;
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Displays the constituents of a list of entities
 * @param cells_local_id List of local IDs of the entities
 */
void ConstituentConnectivityList::
printConstituents(SmallSpan<const Int32> cells_local_id) const
{
  const ConstituentContainer::View materials_view(m_container->m_material);
  const ConstituentContainer::View environments_view(m_container->m_environment);

  for (Int32 i = 0, n = cells_local_id.size(); i < n; ++i) {
    Int32 lid = cells_local_id[i];
    info() << "Cell index=" << i << " lid=" << lid
           << " materials=" << materials_view.components(lid)
           << " environments=" << environments_view.components(lid);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fills the constituents affected by a modification.
 *
 * The meshes affected by the modification are given by \a cells_local_id.
 * \a modified_mat_id is the ID of the material added (if \a is_add is true)
 * or removed (if \a is_add is false).
 *
 * Sets \a is_modified_materials and \a is_modified_environments to true
 * if they are in one of the meshes in \a cells_local_id.
 *
 * This method allows optimizing the calculation of constituents that will be impacted
 * by a modification by trying to determine in advance which ones will be
 * impacted. This avoids calling calculation kernels (for example, calculating the list
 * of meshes to transform for a constituent) if we know the constituent is not impacted.
 *
 * It is possible to revert to the old mechanism and remove this optimization
 * by filling \a is_modified_materials and \a is_modified_environments with \a true for all
 * constituents. This is the case if m_is_force_transform_all_constituants is true.
 *
 * \note it is important that \a is_modified_materials and \a is_modified_environments is
 * true if the corresponding constituent is modified. Otherwise, there will be an inconsistency
 * in the constituents. However, it is not serious if we mark a constituent as modified
 * and it turns out not to be the case. This just means we will perform unnecessary
 * operations (in MeshMaterialVariableIndexer::transformCells()).
 */
void ConstituentConnectivityList::
fillModifiedConstituents(SmallSpan<const Int32> cells_local_id,
                         SmallSpan<bool> is_modified_materials,
                         SmallSpan<bool> is_modified_environments,
                         int modified_mat_id, bool is_add, const RunQueue& queue)
{
  const Int32 n = cells_local_id.size();
  if (n <= 0)
    return;

  bool is_device = queue.isAcceleratorPolicy();
  const ConstituentContainer::View materials_view(m_container->m_material);
  const ConstituentContainer::View environments_view(m_container->m_environment);
  auto env_for_mat = m_environment_for_materials.view(is_device);
  NumberOfMaterialComputer nb_mat_computer(materials_view, m_environment_for_materials.view(is_device));
  Int16 modified_env_id = m_environment_for_materials.hostView()[modified_mat_id];
  ConstArrayView<Int16> cells_nb_environment = cellsNbEnvironment();

  auto command = makeCommand(queue);
  ITraceMng* tm = traceMng();
  tm->info(4) << "FillModifiedConstituents modified_mat=" << modified_mat_id
              << " modified_env=" << modified_env_id << " is_add=" << is_add;
  const bool force_transform = m_is_force_transform_all_constituants;

  if (force_transform) {
    // If we force the modification for everyone,
    // directly fill the array of modified environments and materials
    command << RUNCOMMAND_LOOP1(iter, n)
    {
      auto [i] = iter();
      const Int32 local_id = cells_local_id[i];
      SmallSpan<const Int16> cell_envs(environments_view.components(local_id));
      for (Int16 x : cell_envs)
        is_modified_environments[x] = true;
      SmallSpan<const Int16> cell_mats(materials_view.components(local_id));
      for (Int16 x : cell_mats)
        is_modified_materials[x] = true;
    };
    return;
  }

  command << RUNCOMMAND_LOOP1(iter, n)
  {
    auto [i] = iter();
    const Int32 local_id = cells_local_id[i];
    const Int16 nb_mat_in_modified_env = nb_mat_computer.cellNbMaterial(local_id, modified_env_id);
    const Int16 nb_env = cells_nb_environment[local_id];
    // No environments in the mesh. This is initialization.
    // No material or environment other than the added one is affected.
    if (nb_env == 0)
      return;
    //tm->info() << "FillModified: Cell lid=" << local_id << " nb_mat_in_modified_env=" << nb_mat_in_modified_env << " nb_env=" << nb_env;
    SmallSpan<const Int16> cell_envs(environments_view.components(local_id));
    for (Int16 x : cell_envs) {
      // Do not process the environment currently being modified
      if (x == modified_env_id)
        continue;
      bool do_transform = false;
      if (is_add) {
        // In case of addition, we transform if there is only one environment and it is us
        // (this is necessarily the case, as we are in the loop of the mesh environments)
        do_transform = nb_env == 1;
      }
      else {
        // I transform if we go from 2 environments to only 1 (I am necessarily this environment
        // if I am in the loop)
        do_transform = nb_env == 2;
      }
      if (do_transform) {
        //tm->info() << "FillModified:   SetTransform Cell lid=" << local_id << " env=" << x;
        is_modified_environments[x] = true;
      }
    }

    SmallSpan<const Int16> cell_mats(materials_view.components(local_id));
    for (Int16 x : cell_mats) {
      // Do not process the material currently being modified
      if (x == modified_mat_id)
        continue;
      bool do_transform = false;
      Int16 my_env_id = env_for_mat[x];
      // TODO: calculate only if necessary.
      Int16 my_mat_nb_env = nb_mat_computer.cellNbMaterial(local_id, my_env_id);
      //tm->info() << "FillModified:   CheckMat lid=" << local_id
      //           << " mat_id=" << x << " mat_env=" << my_env_id
      //           << " my_mat_nb_env=" << my_mat_nb_env;
      if (is_add) {
        if (my_env_id != modified_env_id && (nb_mat_in_modified_env != 0))
          continue;
        do_transform = (nb_env == 1) && (my_mat_nb_env == 1);
      }
      else {
        // I transform if I am the material of the modified environment and there are only two materials in the environment (because I will become the only material
        // of the environment)
        // NOTE: this condition is necessary but does not guarantee that I will
        // necessarily transform. For now, we leave it this way to not be too
        // restrictive.
        do_transform = true;
        if (nb_env == 1)
          do_transform = (my_env_id == modified_env_id) && (my_mat_nb_env == 2);
      }
      if (do_transform) {
        //tm->info() << "FillModified:   SetTransform Cell lid=" << local_id << " mat=" << x;
        is_modified_materials[x] = true;
      }
    }
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
notifySourceFamilyLocalIdChanged([[maybe_unused]] Int32ConstArrayView new_to_old_ids)
{
  m_container->changeLocalIds(new_to_old_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
notifySourceItemAdded(ItemLocalId item)
{
  Int32 lid = item.localId();
  m_container->checkResize(lid + 1);

  m_container->m_environment.m_nb_component_as_array[lid] = 0;
  m_container->m_environment.m_component_index_as_array[lid] = 0;

  m_container->m_material.m_nb_component_as_array[lid] = 0;
  m_container->m_material.m_component_index_as_array[lid] = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
reserveMemoryForNbSourceItems(Int32 n, [[maybe_unused]] bool pre_alloc_connectivity)
{
  info() << "Constituent: reserve=" << n;
  m_container->reserve(n);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
notifyReadFromDump()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IIncrementalItemSourceConnectivity> ConstituentConnectivityList::
toSourceReference()
{
  return Arccore::makeRef<IIncrementalItemSourceConnectivity>(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
removeAllConnectivities()
{
  m_container->m_environment.removeAllConnectivities();
  m_container->m_material.removeAllConnectivities();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
