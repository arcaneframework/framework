// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AllCellToAllEnvCellConverter.cc                             (C) 2000-2023 */
/*                                                                           */
/* Conversion de 'Cell' en 'AllEnvCell'.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/AllCellToAllEnvCellConverter.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"

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
    for (auto i(m_size - 1); i >= 0; --i) {
      if (!m_allcell_allenvcell[i].empty())
        m_alloc->deallocate(m_allcell_allenvcell[i].data());
      m_allcell_allenvcell[i].~Span<ComponentItemLocalId>();
      // la memoire sera liberee avec l'appel manuel du dtor a cause du placement new dans create
    }
    m_alloc->deallocate(m_allcell_allenvcell);
    m_allcell_allenvcell = nullptr;
  }
  m_material_mng = nullptr;
  m_alloc = nullptr;
  m_size = 0;
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

  CellToAllEnvCellConverter all_env_cell_converter(mm);
  ENUMERATE_CELL (icell, mm->mesh()->allCells()) {
    Int32 cid = icell->itemLocalId();
    AllEnvCell all_env_cell = all_env_cell_converter[CellLocalId(cid)];
    ComponentItemLocalId* env_cells(nullptr);
    Integer nb_env(all_env_cell.nbEnvironment());
    if (nb_env) {
      env_cells = reinterpret_cast<ComponentItemLocalId*>(alloc->allocate(sizeof(ComponentItemLocalId) * nb_env));
      Integer i(0);
      ENUMERATE_CELL_ENVCELL (ienvcell, all_env_cell) {
        EnvCell ev = *ienvcell;
        env_cells[i] = *(new (env_cells + i) ComponentItemLocalId(ev._varIndex()));
        ++i;
      }
    }
    _instance->m_allcell_allenvcell[cid] = (nb_env ? Span<ComponentItemLocalId>(env_cells, nb_env) : Span<ComponentItemLocalId>());
  }
  return _instance;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllCellToAllEnvCell::
bruteForceUpdate(Int32ConstArrayView ids)
{
  if (m_size != m_material_mng->mesh()->allCells().itemFamily()->maxLocalId() + 1) {
    AllCellToAllEnvCell* swap_ptr(create(m_material_mng, m_alloc));
    std::swap(this->m_material_mng, swap_ptr->m_material_mng);
    std::swap(this->m_alloc, swap_ptr->m_alloc);
    std::swap(this->m_size, swap_ptr->m_size);
    std::swap(this->m_allcell_allenvcell, swap_ptr->m_allcell_allenvcell);
    destroy(swap_ptr);
  }
  else {
    // Si le nb de maille n'a pas changé, on reconstruit en fonction de la liste de maille
    CellToAllEnvCellConverter all_env_cell_converter(m_material_mng);
    for (Int32 i(0), n = ids.size(); i < n; ++i) {
      CellLocalId lid(ids[i]);
      // Si c'est pas vide, on efface et on refait
      if (!m_allcell_allenvcell[lid].empty()) {
        m_alloc->deallocate(m_allcell_allenvcell[lid].data());
        m_allcell_allenvcell[i].~Span<ComponentItemLocalId>();
      }
      AllEnvCell all_env_cell = all_env_cell_converter[lid];
      ComponentItemLocalId* env_cells(nullptr);
      Integer nb_env(all_env_cell.nbEnvironment());
      if (nb_env) {
        env_cells = reinterpret_cast<ComponentItemLocalId*>(m_alloc->allocate(sizeof(ComponentItemLocalId) * nb_env));
        Integer i(0);
        ENUMERATE_CELL_ENVCELL (ienvcell, all_env_cell) {
          EnvCell ev = *ienvcell;
          env_cells[i] = *(new (env_cells + i) ComponentItemLocalId(ev._varIndex()));
          ++i;
        }
      }
      m_allcell_allenvcell[lid] = (nb_env ? Span<ComponentItemLocalId>(env_cells, nb_env) : Span<ComponentItemLocalId>());
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
