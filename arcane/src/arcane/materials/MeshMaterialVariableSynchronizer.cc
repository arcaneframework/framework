// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableSynchronizer.cc                         (C) 2000-2024 */
/*                                                                           */
/* Material variable synchronizer.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/MeshMaterialVariableSynchronizer.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/MemoryUtils.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IVariableSynchronizer.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/internal/IParallelMngInternal.h"

#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/IMeshMaterialSynchronizeBuffer.h"
#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"

#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/Scan.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariableSynchronizer::
MeshMaterialVariableSynchronizer(IMeshMaterialMng* material_mng,
                                 IVariableSynchronizer* var_syncer,
                                 MatVarSpace space)
: TraceAccessor(material_mng->traceMng())
, m_material_mng(material_mng)
, m_variable_synchronizer(var_syncer)
, m_timestamp(-1)
, m_var_space(space)
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_MATERIALSYNCHRONIZER_ACCELERATOR_MODE", true))
    m_use_accelerator_mode = v.value();
  _initialize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableSynchronizer* MeshMaterialVariableSynchronizer::
variableSynchronizer()
{
  return m_variable_synchronizer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<MatVarIndex> MeshMaterialVariableSynchronizer::
sharedItems(Int32 index)
{
  return m_shared_items[index];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<MatVarIndex> MeshMaterialVariableSynchronizer::
ghostItems(Int32 index)
{
  return m_ghost_items[index];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Fills \a items with the list of all MatVarIndex from the cells of \a view.
 */
void MeshMaterialVariableSynchronizer::
_fillCells(Array<MatVarIndex>& items, AllEnvCellVectorView view, RunQueue& queue)
{
  items.clear();

  // NOTE: It is possible to optimize by looking at elements that have
  // only one material, because in that case the element value and the
  // material value are the same. Similarly, if there is only one element,
  // then the global value and the element value are the same. In these
  // cases, it is not necessary to add the second MatCell to the list.
  // If we implement this optimization, we will then need to modify the
  // corresponding serialization for the variables
  if (view.size() == 0)
    return;

  bool use_accelerator = queue.isAcceleratorPolicy();
  if (m_use_accelerator_mode == 1)
    use_accelerator = true;
  if (m_use_accelerator_mode == 0)
    use_accelerator = false;

  if (use_accelerator)
    _fillCellsAccelerator(items, view, queue);
  else
    _fillCellsSequential(items, view);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Fills \a items with the list of all MatVarIndex from the cells of \a view.
 */
void MeshMaterialVariableSynchronizer::
_fillCellsSequential(Array<MatVarIndex>& items, AllEnvCellVectorView view)
{
  bool has_mat = m_var_space == MatVarSpace::MaterialAndEnvironment;

  ENUMERATE_ALLENVCELL (iallenvcell, view) {
    AllEnvCell all_env_cell = *iallenvcell;
    for (EnvCell env_cell : all_env_cell.subEnvItems()) {
      items.add(env_cell._varIndex());
      if (has_mat) {
        for (MatCell mat_cell : env_cell.subMatItems()) {
          items.add(mat_cell._varIndex());
        }
      }
    }
    // In principle, adding this information is not necessary because it
    // is possible to retrieve the global variable info.
    items.add(MatVarIndex(0, view.localId(iallenvcell.index())));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Fills \a items with the list of all MatVarIndex from the cells of \a view.
 */
void MeshMaterialVariableSynchronizer::
_fillCellsAccelerator(Array<MatVarIndex>& items, AllEnvCellVectorView view, RunQueue& queue)
{
  bool has_mat = m_var_space == MatVarSpace::MaterialAndEnvironment;
  // Perform a Scan to know the index of each element and the total number
  // The array has (nb_item+1) elements and the last value will contain the
  // total number of elements in the list
  Int32 nb_item = view.size();
  UniqueArray<Int32> indexes(queue.allocationOptions());
  indexes.resize(nb_item + 1);

  Accelerator::GenericScanner scanner(queue);
  Accelerator::ScannerSumOperator<Int32> op;
  Span<Int32> out_indexes = indexes;
  auto getter = [=] ARCCORE_HOST_DEVICE(Int32 index) -> Int32 {
    if (index == nb_item)
      return 0;
    AllEnvCell all_env_cell = view[index];
    Int32 n = 0;
    for (EnvCell env_cell : all_env_cell.subEnvItems()) {
      ++n;
      if (has_mat)
        n += env_cell.nbSubItem();
    }
    ++n;
    return n;
  };

  {
    // Array to store the final sum
    NumArray<Int32, MDDim1> host_total_storage(1, eMemoryRessource::HostPinned);
    SmallSpan<Int32> in_host_total_storage(host_total_storage);

    auto setter = [=] ARCCORE_HOST_DEVICE(Int32 index, Int32 value) {
      out_indexes[index] = value;
      if (index == nb_item)
        in_host_total_storage[0] = value;
    };
    scanner.applyWithIndexExclusive(nb_item + 1, 0, getter, setter, op);
    Int32 total = host_total_storage[0]; //indexes[nb_item];
    items.resize(total);
  }

  {
    auto command = makeCommand(queue);
    Span<const Int32> in_indexes = indexes;
    Span<MatVarIndex> out_mat_var_indexes = items;
    command << RUNCOMMAND_LOOP1(iter, nb_item)
    {
      auto [index] = iter();
      AllEnvCell all_env_cell = view[index];
      Int32 pos = in_indexes[index];
      for (EnvCell env_cell : all_env_cell.subEnvItems()) {
        out_mat_var_indexes[pos] = env_cell._varIndex();
        ++pos;
        if (has_mat) {
          for (MatCell mat_cell : env_cell.subMatItems()) {
            out_mat_var_indexes[pos] = mat_cell._varIndex();
            ++pos;
          }
        }
      }
      // In principle, adding this information is not necessary because it
      // is possible to retrieve the global variable info.
      out_mat_var_indexes[pos] = MatVarIndex(0, view.localId(index));
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableSynchronizer::
checkRecompute()
{
  Int64 ts = m_material_mng->timestamp();
  if (m_timestamp != ts)
    recompute();
  m_timestamp = ts;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableSynchronizer::
recompute()
{
  IVariableSynchronizer* var_syncer = m_variable_synchronizer;

  // Calculation of synchronization information for material cells.
  // NOTE: This version requires that the materials are correctly
  // synchronized between sub-domains.

  IItemFamily* family = var_syncer->itemGroup().itemFamily();
  IParallelMng* pm = var_syncer->parallelMng();
  if (!pm->isParallel())
    return;
  ItemGroup all_items = family->allItems();

  Int32ConstArrayView ranks = var_syncer->communicatingRanks();
  Integer nb_rank = ranks.size();

  m_common_buffer->setNbRank(nb_rank);

  m_shared_items.resize(nb_rank);
  m_ghost_items.resize(nb_rank);

  RunQueue queue = m_material_mng->_internalApi()->runQueue();

  {
    // These arrays must be accessible on the accelerator
    // TODO: eventually, use a single array for sends
    // and a single one for receives.
    MemoryAllocationOptions a = queue.allocationOptions();
    for (Int32 i = 0; i < nb_rank; ++i) {
      m_shared_items[i] = UniqueArray<MatVarIndex>(a);
      m_ghost_items[i] = UniqueArray<MatVarIndex>(a);
    }
  }

  // NOTE: The calls to _fillCells() are independent. We could
  // therefore make them asynchronous.
  for (Integer i = 0; i < nb_rank; ++i) {

    {
      Int32ConstArrayView shared_ids = var_syncer->sharedItems(i);
      CellVectorView shared_cells(family->view(shared_ids));
      AllEnvCellVectorView view = m_material_mng->view(shared_cells);
      Array<MatVarIndex>& items = m_shared_items[i];
      _fillCells(items, view, queue);
      info(4) << "SIZE SHARED FOR rank=" << ranks[i] << " n=" << items.size();
    }

    {
      Int32ConstArrayView ghost_ids = var_syncer->ghostItems(i);
      CellVectorView ghost_cells(family->view(ghost_ids));
      AllEnvCellVectorView view = m_material_mng->view(ghost_cells);
      Array<MatVarIndex>& items = m_ghost_items[i];
      _fillCells(items, view, queue);
      info(4) << "SIZE GHOST FOR rank=" << ranks[i] << " n=" << items.size();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableSynchronizer::
_initialize()
{
  IParallelMng* pm = m_variable_synchronizer->parallelMng();
  if (pm->_internalApi()->isAcceleratorAware()) {
    m_buffer_memory_ressource = eMemoryRessource::Device;
    info() << "MeshMaterialVariableSynchronizer: Using device memory for buffer";
  }
  m_common_buffer = impl::makeOneBufferMeshMaterialSynchronizeBufferRef(m_buffer_memory_ressource);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
