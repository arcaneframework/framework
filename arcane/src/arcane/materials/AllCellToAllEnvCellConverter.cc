// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AllCellToAllEnvCellConverter.cc                             (C) 2000-2024 */
/*                                                                           */
/* Conversion de 'Cell' en 'AllEnvCell'.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/AllCellToAllEnvCellConverter.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"

#include "arcane/accelerator/Reduce.h"
#include "arcane/accelerator/RunCommandEnumerate.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

class AllCellToAllEnvCell::Impl
{
 public:

  static Int32 _computeMaxNbEnvPerCell(IMeshMaterialMng* material_mng);
  static void _updateValues(IMeshMaterialMng* material_mng,
                            ComponentItemLocalId* mem_pool,
                            Span<ComponentItemLocalId>* allcell_allenvcell,
                            Int32 max_nb_env);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 AllCellToAllEnvCell::Impl::
_computeMaxNbEnvPerCell(IMeshMaterialMng* material_mng)
{
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

void AllCellToAllEnvCell::Impl::
_updateValues(IMeshMaterialMng* material_mng,
              ComponentItemLocalId* mem_pool,
              Span<ComponentItemLocalId>* allcell_allenvcell,
              Int32 max_nb_env)
{
  // mise a jour des valeurs
  CellToAllEnvCellConverter all_env_cell_converter(material_mng);
  RunQueue& queue = material_mng->_internalApi()->runQueue();
  auto command = makeCommand(queue);
  command << RUNCOMMAND_ENUMERATE (CellLocalId, cid, material_mng->mesh()->allCells())
  {
    AllEnvCell all_env_cell = all_env_cell_converter[cid];
    Int32 nb_env = all_env_cell.nbEnvironment();
    if (nb_env != 0) {
      Integer i = 0;
      Integer offset = cid * max_nb_env;
      for (EnvCell ev : all_env_cell.subEnvItems()) {
        mem_pool[offset + i] = ComponentItemLocalId(ev._varIndex());
        ++i;
      }
      allcell_allenvcell[cid] = Span<ComponentItemLocalId>(mem_pool + offset, nb_env);
    }
    else {
      allcell_allenvcell[cid] = Span<ComponentItemLocalId>();
    }
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllCellToAllEnvCell::
reset()
{
  if (m_allcell_allenvcell_ptr) {
    m_allcell_allenvcell.resize(0);
    m_allcell_allenvcell_ptr = nullptr;
    m_mem_pool.resize(0);
  }
  m_material_mng = nullptr;
  m_alloc = nullptr;
  m_size = 0;
  m_current_max_nb_env = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllCellToAllEnvCell::
destroy(AllCellToAllEnvCell* instance)
{
  IMemoryAllocator* alloc(instance->m_alloc);
  instance->reset();
  alloc->deallocate(instance);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 AllCellToAllEnvCell::
maxNbEnvPerCell() const
{
  return Impl::_computeMaxNbEnvPerCell(m_material_mng);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllCellToAllEnvCell* AllCellToAllEnvCell::
create(IMeshMaterialMng* mm, IMemoryAllocator* alloc)
{
  AllCellToAllEnvCell* _instance(nullptr);
  _instance = reinterpret_cast<AllCellToAllEnvCell*>(alloc->allocate(sizeof(AllCellToAllEnvCell)));
  if (!_instance)
    ARCANE_FATAL("Unable to allocate memory for AllCellToAllEnvCell instance");
  _instance = new (_instance) AllCellToAllEnvCell();
  _instance->m_material_mng = mm;
  _instance->m_alloc = alloc;
  _instance->m_size = mm->mesh()->cellFamily()->maxLocalId() + 1;

  _instance->m_allcell_allenvcell.resize(_instance->m_size);
  _instance->m_allcell_allenvcell_ptr = _instance->m_allcell_allenvcell.to1DSpan().data();

  // On force la valeur initiale sur tous les elmts car dans le ENUMERATE_CELL ci-dessous
  // il se peut que m_size (qui vaut maxLocalId()+1) soit different de allCells().size()
  _instance->m_allcell_allenvcell.fill(Span<ComponentItemLocalId>());

  _instance->m_current_max_nb_env = _instance->maxNbEnvPerCell();
  auto pool_size(_instance->m_current_max_nb_env * _instance->m_size);
  _instance->m_mem_pool.resize(pool_size);
  _instance->m_mem_pool.fill(ComponentItemLocalId());
  Span<ComponentItemLocalId> mem_pool_view(_instance->m_mem_pool.to1DSpan());
  CellToAllEnvCellConverter all_env_cell_converter(mm);
  ENUMERATE_CELL (icell, mm->mesh()->allCells()) {
    Int32 cid = icell->itemLocalId();
    AllEnvCell all_env_cell = all_env_cell_converter[CellLocalId(cid)];
    Integer nb_env(all_env_cell.nbEnvironment());
    if (nb_env != 0) {
      Integer i = 0;
      Integer offset(cid * _instance->m_current_max_nb_env);
      ENUMERATE_CELL_ENVCELL (ienvcell, all_env_cell) {
        EnvCell ev = *ienvcell;
        _instance->m_mem_pool[offset + i] = ComponentItemLocalId(ev._varIndex());
        ++i;
      }
      _instance->m_allcell_allenvcell[cid] = Span<ComponentItemLocalId>(mem_pool_view.ptrAt(offset), nb_env);
    }
  }
  return _instance;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllCellToAllEnvCell::
bruteForceUpdate()
{
  // Si les ids ont changé, on doit tout refaire
  if (m_size != m_material_mng->mesh()->allCells().itemFamily()->maxLocalId() + 1) {
    AllCellToAllEnvCell* swap_ptr(create(m_material_mng, m_alloc));
    std::swap(this->m_material_mng, swap_ptr->m_material_mng);
    std::swap(this->m_alloc, swap_ptr->m_alloc);
    std::swap(this->m_size, swap_ptr->m_size);
    std::swap(this->m_allcell_allenvcell, swap_ptr->m_allcell_allenvcell);
    std::swap(this->m_mem_pool, swap_ptr->m_mem_pool);
    std::swap(this->m_current_max_nb_env, swap_ptr->m_current_max_nb_env);
    destroy(swap_ptr);
  }
  else {
    Int32 current_max_nb_env(maxNbEnvPerCell());
    // Si les ids n'ont pas changé, on regarde si à cet instant, le nb max d'env par maille a changé
    // Si ca a changé, refaire le mem pool, sinon, juste update les valeurs
    if (current_max_nb_env != m_current_max_nb_env) {
      // On n'oublie pas de mettre a jour la nouvelle valeur !
      m_current_max_nb_env = current_max_nb_env;
      // Si le nb max d'env pour les mailles a changé à cet instant, on doit refaire le memory pool
      ARCANE_ASSERT((m_allcell_allenvcell_ptr), ("Trying to change memory pool within a null structure"));
      // on reinit a un span vide
      m_allcell_allenvcell.fill(Span<ComponentItemLocalId>());
      // on recrée le pool
      auto pool_size(m_current_max_nb_env * m_size);
      m_mem_pool.resize(pool_size);
      m_mem_pool.fill(ComponentItemLocalId());
    }
    // Mise a jour des valeurs
    Impl::_updateValues(m_material_mng, m_mem_pool.to1DSpan().data(), m_allcell_allenvcell_ptr, m_current_max_nb_env);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellToAllEnvCellAccessor::
CellToAllEnvCellAccessor(const IMeshMaterialMng* mmmng)
: m_cell_allenvcell(mmmng->_internalApi()->getAllCellToAllEnvCell())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
