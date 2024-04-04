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

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllCellToAllEnvCell::
reset()
{
  if (m_allcell_allenvcell) {
    for (auto i(m_size - 1); i >= 0; --i)
      m_allcell_allenvcell[i].~Span<ComponentItemLocalId>();
    m_alloc->deallocate(m_allcell_allenvcell);
    m_allcell_allenvcell = nullptr;
    m_alloc->deallocate(m_mem_pool);
    m_mem_pool = nullptr;
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
  // On peut imaginer le faire sur l'accelerator aussi, avec :
  // 1. runcmd_enum_cell pour remplir un array de max env de size max cell id
  // 2. runcmd_loop sur le array avec un reducer max
  // A voir si c'est interessant...
  CellToAllEnvCellConverter allenvcell_converter(m_material_mng);
  Int32 max_nb_env(0);
  ENUMERATE_CELL (icell, m_material_mng->mesh()->allCells()) {
    AllEnvCell all_env_cell = allenvcell_converter[icell];
    if (all_env_cell.nbEnvironment() > max_nb_env)
      max_nb_env = all_env_cell.nbEnvironment();
  }
  return max_nb_env;
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

  _instance->m_allcell_allenvcell = reinterpret_cast<Span<ComponentItemLocalId>*>(
  alloc->allocate(sizeof(Span<ComponentItemLocalId>) * _instance->m_size));
  // On force la valeur initiale sur tous les elmts car dans le ENUMERATE_CELL ci-dessous
  // il se peut que m_size (qui vaut maxLocalId()+1) soit different de allCells().size()
  std::fill_n(_instance->m_allcell_allenvcell, _instance->m_size, Span<ComponentItemLocalId>());

  _instance->m_current_max_nb_env = _instance->maxNbEnvPerCell();
  auto pool_size(_instance->m_current_max_nb_env * _instance->m_size);
  _instance->m_mem_pool = reinterpret_cast<ComponentItemLocalId*>(alloc->allocate(sizeof(ComponentItemLocalId) * pool_size));
  std::fill_n(_instance->m_mem_pool, pool_size, ComponentItemLocalId());

  CellToAllEnvCellConverter all_env_cell_converter(mm);
  ENUMERATE_CELL (icell, mm->mesh()->allCells()) {
    Int32 cid = icell->itemLocalId();
    AllEnvCell all_env_cell = all_env_cell_converter[CellLocalId(cid)];
    Integer nb_env(all_env_cell.nbEnvironment());
    if (nb_env) {
      Integer i(0);
      Integer offset(cid * _instance->m_current_max_nb_env);
      ENUMERATE_CELL_ENVCELL (ienvcell, all_env_cell) {
        EnvCell ev = *ienvcell;
        _instance->m_mem_pool[offset + i] = ComponentItemLocalId(ev._varIndex());
        ++i;
      }
      _instance->m_allcell_allenvcell[cid] = Span<ComponentItemLocalId>(_instance->m_mem_pool + offset, nb_env);
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
      ARCANE_ASSERT((m_allcell_allenvcell), ("Trying to change memory pool within a null structure"));
      // on reinit a un span vide
      std::fill_n(m_allcell_allenvcell, m_size, Span<ComponentItemLocalId>());
      // on recree le pool
      m_alloc->deallocate(m_mem_pool);
      auto pool_size(m_current_max_nb_env * m_size);
      m_mem_pool = reinterpret_cast<ComponentItemLocalId*>(m_alloc->allocate(sizeof(ComponentItemLocalId) * pool_size));
      std::fill_n(m_mem_pool, pool_size, ComponentItemLocalId());
    }
    // mise a jour des valeurs
    CellToAllEnvCellConverter all_env_cell_converter(m_material_mng);
    ENUMERATE_CELL (icell, m_material_mng->mesh()->allCells()) {
      Int32 cid = icell->itemLocalId();
      AllEnvCell all_env_cell = all_env_cell_converter[CellLocalId(cid)];
      Integer nb_env(all_env_cell.nbEnvironment());
      if (nb_env) {
        Integer i(0);
        Integer offset(cid * m_current_max_nb_env);
        ENUMERATE_CELL_ENVCELL (ienvcell, all_env_cell) {
          EnvCell ev = *ienvcell;
          m_mem_pool[offset + i] = ComponentItemLocalId(ev._varIndex());
          ++i;
        }
        m_allcell_allenvcell[cid] = Span<ComponentItemLocalId>(m_mem_pool + offset, nb_env);
      }
      else {
        m_allcell_allenvcell[cid] = Span<ComponentItemLocalId>();
      }
    }
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
