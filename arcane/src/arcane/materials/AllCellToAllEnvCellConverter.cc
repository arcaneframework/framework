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

#include <algorithm>
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
old_reset()
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
  m_current_max_nb_env = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllCellToAllEnvCell::
reset()
{
  if (m_allcell_allenvcell) {
    ComponentItemLocalId* env_cells(m_allcell_allenvcell[0].data());
    for (auto i(m_size - 1); i >= 0; --i)
      m_allcell_allenvcell[i].~Span<ComponentItemLocalId>();
    m_alloc->deallocate(env_cells);
    m_alloc->deallocate(m_allcell_allenvcell);
    m_allcell_allenvcell = nullptr;
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

AllCellToAllEnvCell* AllCellToAllEnvCell::
old_create(IMeshMaterialMng* mm, IMemoryAllocator* alloc)
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
  // We force initial value on every elmts  because the ENUMERATE_CELL below
  // can miss some (maxLocalId()+1 can be different from allCells())
  for (auto i(0); i < _instance->m_size; ++i)
    _instance->m_allcell_allenvcell[i] = Span<ComponentItemLocalId>();

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
      _instance->m_allcell_allenvcell[cid] = Span<ComponentItemLocalId>(env_cells, nb_env);
    }
  }
  return _instance;
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

  // TODO(FL) : On peut aussi penser qu'avec un sort on serait plus performant
  // vu que le nb d'environnement max par maille doit etre dans le meme halo... à voir

  CellToAllEnvCellConverter allenvcell_converter(m_material_mng);
  UniqueArray<Int32> max_size_tab(m_material_mng->mesh()->allCells().size(), 0);
  ENUMERATE_CELL(icell, m_material_mng->mesh()->allCells()) {
    AllEnvCell all_env_cell = allenvcell_converter[icell];
    max_size_tab[icell->itemLocalId()] = all_env_cell.nbEnvironment();
  }

  return static_cast<Int32>(*std::max_element(max_size_tab.begin(), max_size_tab.end()));
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
  std::fill(_instance->m_allcell_allenvcell, _instance->m_allcell_allenvcell+_instance->m_size, Span<ComponentItemLocalId>());

  _instance->m_current_max_nb_env = _instance->maxNbEnvPerCell();
  ComponentItemLocalId* env_cells(nullptr);
  // Ici on alloue sur le allCells().size() car on va positionner les span
  // via un ENUMERATE_CELL(...allCells)
  auto pool_size(_instance->m_current_max_nb_env * mm->mesh()->allCells().size());
  env_cells = reinterpret_cast<ComponentItemLocalId*>(alloc->allocate(sizeof(ComponentItemLocalId) * pool_size));
  std::fill(env_cells, env_cells+pool_size, *(new ComponentItemLocalId()));

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
        env_cells[offset + i] = ComponentItemLocalId(ev._varIndex());
        ++i;
      }
      _instance->m_allcell_allenvcell[cid] = Span<ComponentItemLocalId>(env_cells+offset, nb_env);
    }
  }
  return _instance;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllCellToAllEnvCell::
old_bruteForceUpdate()
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
    ENUMERATE_CELL (icell, m_material_mng->mesh()->allCells()) {
      CellLocalId lid(icell->itemLocalId());
      // Si c'est pas vide, on efface et on refait
      if (!m_allcell_allenvcell[lid].empty()) {
        m_alloc->deallocate(m_allcell_allenvcell[lid].data());
        m_allcell_allenvcell[lid].~Span<ComponentItemLocalId>();
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
    destroy(swap_ptr);
  }
  else {
    Int32 current_max_nb_env(maxNbEnvPerCell());
    ComponentItemLocalId* env_cells(nullptr);
    bool mem_pool_has_changed(false);
    // Si les ids n'ont pas changé, on regarde si à cet instant, le nb max d'env par maille a changé
    // TODO: a finir: si ca a changer, refaire le mem pool, sinon, juste update les valeurs
    if (current_max_nb_env != m_current_max_nb_env) {
      // Si le nb max d'env pour les mailles a changé à cet instant, on doit refaire le memory pool
      m_alloc->deallocate(m_allcell_allenvcell[0].data());
      // on reinit a un span vide
      std::fill(m_allcell_allenvcell, m_allcell_allenvcell+m_size, Span<ComponentItemLocalId>());
      // on recree le pool
      auto pool_size(m_current_max_nb_env * m_material_mng->mesh()->allCells().size());
      env_cells = reinterpret_cast<ComponentItemLocalId*>(m_alloc->allocate(sizeof(ComponentItemLocalId) * pool_size));
      std::fill(env_cells, env_cells+pool_size, *(new ComponentItemLocalId()));
      mem_pool_has_changed = true;
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
          // si le pool memoire a changer, on construit l'objet, sinon on met juste a jour
          env_cells[offset+i] = (mem_pool_has_changed?
                                 *(new (env_cells+(offset+i)) ComponentItemLocalId(ev._varIndex())):
                                 ComponentItemLocalId(ev._varIndex()));
          ++i;
        }
        // si le pool memoire a changé on doit aussi recréer le span
        if (mem_pool_has_changed)
          m_allcell_allenvcell[cid] = Span<ComponentItemLocalId>(env_cells+offset, nb_env);
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
