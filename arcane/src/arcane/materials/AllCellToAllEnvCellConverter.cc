// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AllCellToAllEnvCellConverter.cc                             (C) 2000-2024 */
/*                                                                           */
/* Conversion of 'Cell' to 'AllEnvCell'.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/AllCellToAllEnvCellConverter.h"

#include "arcane/utils/NumArray.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"

#include "arcane/materials/internal/AllCellToAllEnvCellContainer.h"

#include "arcane/accelerator/Reduce.h"
#include "arcane/accelerator/RunCommandEnumerate.h"
#include "arcane/accelerator/NumArrayViews.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AllCellToAllEnvCellContainer::Impl
{
 public:

  explicit Impl(IMeshMaterialMng* mm)
  : m_material_mng(mm)
  {
    m_mem_pool.setDebugName("AllCellToAllEnvCellMemPool");
    m_envcell_container.setDebugName("AllCellToAllEnvCellCells");
  }

 public:

  Int32 computeMaxNbEnvPerCell();
  void initialize();
  void bruteForceUpdate();
  void reset();
  AllCellToAllEnvCell view() const { return m_all_cell_to_all_env_cell; }

 public:

  static void updateValues(IMeshMaterialMng* material_mng,
                           Span<ComponentItemLocalId> mem_pool,
                           Span<Span<ComponentItemLocalId>> allcell_allenvcell,
                           Int32 max_nb_env);

 private:

  IMeshMaterialMng* m_material_mng = nullptr;
  Int32 m_size = 0;
  NumArray<Span<ComponentItemLocalId>, MDDim1> m_envcell_container;
  NumArray<ComponentItemLocalId, MDDim1> m_mem_pool;
  Int32 m_current_max_nb_env = 0;
  AllCellToAllEnvCell m_all_cell_to_all_env_cell;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 AllCellToAllEnvCellContainer::Impl::
computeMaxNbEnvPerCell()
{
  IMeshMaterialMng* material_mng = m_material_mng;
  CellToAllEnvCellConverter allenvcell_converter(material_mng);
  RunQueue& queue = material_mng->_internalApi()->runQueue();
  Accelerator::GenericReducer<Int32> reducer(queue);
  auto local_ids = material_mng->mesh()->allCells().internal()->itemsLocalId();
  Int32 nb_item = local_ids.size();
  auto select_func = [=] ARCCORE_HOST_DEVICE(Int32 i) -> Int32 {
    CellLocalId lid(local_ids[i]);
    AllEnvCell all_env_cell = allenvcell_converter[lid];
    return all_env_cell.nbEnvironment();
  };
  reducer.applyMaxWithIndex(nb_item, select_func);
  Int32 max_nb_env = reducer.reducedValue();
  return max_nb_env;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllCellToAllEnvCellContainer::Impl::
updateValues(IMeshMaterialMng* material_mng,
             Span<ComponentItemLocalId> mem_pool,
             Span<Span<ComponentItemLocalId>> allcell_allenvcell,
             Int32 max_nb_env)
{
  // update values
  CellToAllEnvCellConverter all_env_cell_converter(material_mng);
  RunQueue& queue = material_mng->_internalApi()->runQueue();
  auto command = makeCommand(queue);
  command << RUNCOMMAND_ENUMERATE (CellLocalId, cid, material_mng->mesh()->allCells())
  {
    AllEnvCell all_env_cell = all_env_cell_converter[cid];
    Integer offset = cid * max_nb_env;
    Int32 nb_env = all_env_cell.nbEnvironment();
    // Initialize the indices of my mesh's 'mem_pools' to zero.
    for (Int32 x = 0; x < max_nb_env; ++x)
      mem_pool[offset + x] = ComponentItemLocalId();
    if (nb_env != 0) {
      Integer i = 0;
      for (EnvCell ev : all_env_cell.subEnvItems()) {
        mem_pool[offset + i] = ComponentItemLocalId(ev._varIndex());
        ++i;
      }
      allcell_allenvcell[cid] = Span<ComponentItemLocalId>(mem_pool.ptrAt(offset), nb_env);
    }
    else {
      allcell_allenvcell[cid] = {};
    }
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllCellToAllEnvCellContainer::Impl::
initialize()
{
  IMeshMaterialMng* mm = m_material_mng;
  RunQueue queue = mm->_internalApi()->runQueue();
  m_size = mm->mesh()->cellFamily()->maxLocalId() + 1;

  m_envcell_container.resize(m_size);
  m_all_cell_to_all_env_cell.m_allcell_allenvcell_ptr = m_envcell_container.to1DSpan();

  // We force the initial value on all elements because in the ENUMERATE_CELL below
  // m_size (which equals maxLocalId()+1) might be different from allCells().size()
  m_envcell_container.fill(Span<ComponentItemLocalId>(), &queue);

  m_current_max_nb_env = computeMaxNbEnvPerCell();
  // TODO: check for overflow
  Int32 pool_size = m_current_max_nb_env * m_size;
  m_mem_pool.resize(pool_size);
  m_mem_pool.fill(ComponentItemLocalId(), &queue);

  Span<ComponentItemLocalId> mem_pool_view(m_mem_pool.to1DSpan());
  CellToAllEnvCellConverter all_env_cell_converter(mm);
  auto command = makeCommand(queue);
  auto mem_pool = viewOut(command, m_mem_pool);
  auto allcell_allenvcell = viewOut(command, m_envcell_container);
  const Int32 max_nb_env = m_current_max_nb_env;
  command << RUNCOMMAND_ENUMERATE (CellLocalId, cid, mm->mesh()->allCells())
  {
    AllEnvCell all_env_cell = all_env_cell_converter[CellLocalId(cid)];
    Integer nb_env(all_env_cell.nbEnvironment());
    if (nb_env != 0) {
      Integer i = 0;
      Integer offset(cid * max_nb_env);
      ENUMERATE_CELL_ENVCELL (ienvcell, all_env_cell) {
        EnvCell ev = *ienvcell;
        mem_pool[offset + i] = ComponentItemLocalId(ev._varIndex());
        ++i;
      }
      allcell_allenvcell[cid] = Span<ComponentItemLocalId>(mem_pool_view.ptrAt(offset), nb_env);
    }
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllCellToAllEnvCellContainer::Impl::
bruteForceUpdate()
{
  // If the IDs have changed, we must redo everything
  if (m_size != m_material_mng->mesh()->allCells().itemFamily()->maxLocalId() + 1) {
    initialize();
    return;
  }

  Int32 current_max_nb_env(computeMaxNbEnvPerCell());
  // If the IDs have not changed, we check if the maximum number of environments per cell has changed at this moment
  // If it has changed, recreate the mem pool; otherwise, just update the values
  if (current_max_nb_env != m_current_max_nb_env) {
    // Don't forget to update the new value!
    m_current_max_nb_env = current_max_nb_env;
    // we recreate the pool
    Int32 pool_size = CheckedConvert::multiply(m_current_max_nb_env, m_size);
    m_mem_pool.resize(pool_size);
  }
  // Update values
  updateValues(m_material_mng, m_mem_pool.to1DSpan(), m_all_cell_to_all_env_cell.m_allcell_allenvcell_ptr, m_current_max_nb_env);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllCellToAllEnvCellContainer::Impl::
reset()
{
  m_envcell_container.resize(0);
  m_all_cell_to_all_env_cell.m_allcell_allenvcell_ptr = {};
  m_mem_pool.resize(0);
  m_material_mng = nullptr;
  m_size = 0;
  m_current_max_nb_env = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllCellToAllEnvCellContainer::
AllCellToAllEnvCellContainer(IMeshMaterialMng* mm)
: m_p(new Impl(mm))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllCellToAllEnvCellContainer::
~AllCellToAllEnvCellContainer()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllCellToAllEnvCellContainer::
reset()
{
  m_p->reset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 AllCellToAllEnvCellContainer::
computeMaxNbEnvPerCell() const
{
  return m_p->computeMaxNbEnvPerCell();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllCellToAllEnvCellContainer::
initialize()
{
  m_p->initialize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllCellToAllEnvCellContainer::
bruteForceUpdate()
{
  m_p->bruteForceUpdate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllCellToAllEnvCell AllCellToAllEnvCellContainer::
view() const
{
  return m_p->view();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellToAllEnvCellAccessor::
CellToAllEnvCellAccessor(const IMeshMaterialMng* mm)
{
  AllCellToAllEnvCellContainer* c = mm->_internalApi()->getAllCellToAllEnvCellContainer();
  if (c)
    m_cell_allenvcell = c->view();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
