// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AllCellToAllEnvCellConverter.cc                             (C) 2000-2023 */
/*                                                                           */
/* Conversion de 'Cell' en 'AllEnvCell'.                                     */
/*---------------------------------------------------------------------------*/

#include "arcane/materials/AllCellToAllEnvCellConverter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
AllCell2AllEnvCell::
AllCell2AllEnvCell()
: m_mm(nullptr), m_alloc(nullptr), m_nb_allcell(0), m_allcell_allenvcell(nullptr)
{
}
*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
AllCell2AllEnvCell::
~AllCell2AllEnvCell()
{
  reset();
}
*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllCell2AllEnvCell::
reset()
{
  if (m_allcell_allenvcell) {
    for (Int64 i(0); i < m_nb_allcell; ++i) {
      if (!m_allcell_allenvcell[i].empty()) {
        m_alloc->deallocate(m_allcell_allenvcell[i].data());
      }
    }
    m_alloc->deallocate(m_allcell_allenvcell);
    m_allcell_allenvcell = nullptr;
  }
  m_mm = nullptr;
  m_alloc = nullptr;
  m_nb_allcell = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllCell2AllEnvCell* AllCell2AllEnvCell::
create(IMeshMaterialMng* mm, IMemoryAllocator* alloc)
{
  AllCell2AllEnvCell *_instance(nullptr);
  _instance = reinterpret_cast<AllCell2AllEnvCell*>(alloc->allocate(sizeof(AllCell2AllEnvCell)));
  if (!_instance)
    ARCANE_FATAL("Unable to allocate memory for AllCell2AllEnvCell instance");

  _instance->m_mm = mm;
  _instance->m_alloc = alloc;
  _instance->m_nb_allcell = mm->mesh()->allCells().size();

  CellToAllEnvCellConverter all_env_cell_converter(mm);

  _instance->m_allcell_allenvcell = reinterpret_cast<Span<ComponentItemLocalId>*>(
    alloc->allocate(sizeof(Span<ComponentItemLocalId>) * _instance->m_nb_allcell));

  ENUMERATE_CELL(icell, mm->mesh()->allCells())
  {
    Int32 cid = icell->itemLocalId();
    AllEnvCell all_env_cell = all_env_cell_converter[CellLocalId(cid)];
    ComponentItemLocalId* env_cells(nullptr);
    Integer nb_env(all_env_cell.nbEnvironment());
    if (nb_env) {
      env_cells = reinterpret_cast<ComponentItemLocalId*>(alloc->allocate(sizeof(ComponentItemLocalId) * nb_env));
      Integer i(0);
      ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell) {
        EnvCell ev = *ienvcell;
        env_cells[i] = ComponentItemLocalId(ev._varIndex());
        ++i;
      }
    }
    _instance->m_allcell_allenvcell[cid] = (nb_env?Span<ComponentItemLocalId>(env_cells, nb_env):Span<ComponentItemLocalId>());
  }
  return _instance;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllCell2AllEnvCell::
bruteForceUpdate(Int32ConstArrayView ids)
{
  // TODO: Je met un fatal, à supprimer une fois bien testé/exploré
  //ARCANE_FATAL("AllCell2AllEnvCell::bruteForceUpdate call !!!");

  // A priori, je ne pense pas que le nb de maille ait changé quand on fait un 
  // ForceRecompute et le updateMaterialDirect. Mais ça doit arriver ailleurs... le endUpdate ?
  if (m_nb_allcell != m_mm->mesh()->allCells().size()) {

    // TODO: Je met un fatal, à supprimer une fois bien testé/exploré
    //ARCANE_FATAL("The number of cells has changed since initialization of AllCell2AllEnvCell.");

    AllCell2AllEnvCell *swap_ptr(create(m_mm, m_alloc));
    std::swap(this->m_nb_allcell, swap_ptr->m_nb_allcell);
    std::swap(this->m_allcell_allenvcell, swap_ptr->m_allcell_allenvcell);
    swap_ptr->reset();
    m_alloc->deallocate(swap_ptr);
  } else {
    // Si le nb de maille n'a pas changé, on reconstruit en fonction de la liste de maille
    CellToAllEnvCellConverter all_env_cell_converter(m_mm);
    for (Int32 i(0), n=ids.size(); i<n; ++i) {
      CellLocalId lid(ids[i]);
      // Si c'est pas vide, on efface et on refait
      if (!m_allcell_allenvcell[lid].empty()) {
        m_alloc->deallocate(m_allcell_allenvcell[lid].data());
      }
      AllEnvCell all_env_cell = all_env_cell_converter[lid];
      ComponentItemLocalId* env_cells(nullptr);
      Span<ComponentItemLocalId> env_cells_span;
      Integer nb_env(all_env_cell.nbEnvironment());
      if (nb_env) {
        env_cells = reinterpret_cast<ComponentItemLocalId*>(m_alloc->allocate(sizeof(ComponentItemLocalId) * nb_env));
        Integer i(0);
        ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell) {
          EnvCell ev = *ienvcell;
          env_cells[i] = ComponentItemLocalId(ev._varIndex());
          ++i;
        }
        env_cells_span = Span<ComponentItemLocalId>(env_cells, nb_env);
      }
      m_allcell_allenvcell[lid] = env_cells_span;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

