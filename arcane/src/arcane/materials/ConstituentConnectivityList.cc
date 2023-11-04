// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentConnectivityList.cc                              (C) 2000-2023 */
/*                                                                           */
/* Gestion des listes de connectivité des constituants.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/ConstituentConnectivityList.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/MeshUtils.h"
#include "arcane/core/internal/IDataInternal.h"
#include "arcane/core/internal/IItemFamilyInternal.h"

#include "arcane/materials/internal/MeshMaterialMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  // Cette méthode est la même que MeshUtils::removeItemAndKeepOrder().
  // Il faudrait pouvoir fusionner les deux.
  // NOTE: Avec le C++20, on devrait pouvoir supprimer cette méthode et
  // utiliser std::erase()

  template <typename DataType>
  void _removeValueAndKeepOrder(ArrayView<DataType> values, DataType value_to_remove)
  {
    Integer n = values.size();
    if (n <= 0)
      ARCANE_FATAL("Can not remove item lid={0} because list is empty", value_to_remove);

    --n;
    if (n == 0) {
      if (values[0] == value_to_remove)
        return;
    }
    else {
      // Si l'élément est le dernier, ne fait rien.
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
    ARCANE_FATAL("No value to remove '{0}' found in list {1}", value_to_remove, values);
  }

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ConstituentConnectivityList::ConstituentContainer
{
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
    if (!is_continue) {
      m_nb_component.fill(0);
      m_component_index.fill(0);
      // Le premier élément de la liste est utilisé pour les constituants vides
      m_component_list.resize(1);
      m_component_list[0] = 0;
    }
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

 private:

  //! Nombre de milieu par maille (dimensionné au nombre de mailles)
  VariableArrayInt16 m_nb_component;
  //! Indice dans \a m_componente_list (Dimensionné au nombre de mailles)
  VariableArrayInt32 m_component_index;
  //! Liste des constituants
  VariableArrayInt16 m_component_list;

 public:

  VariableArrayInt16::ContainerType& m_nb_component_as_array;
  VariableArrayInt32::ContainerType& m_component_index_as_array;
  VariableArrayInt16::ContainerType& m_component_list_as_array;
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
  // S'enregistre auprès la famille pour être notifié des évolutions
  // mais uniquement si on a demandé le support des modifications incrémentales
  // pour éviter de consommer inutilement de la mémoire.
  // A terme on le fera tout le temps
  m_cell_family = m_material_mng->mesh()->cellFamily();
  {
    int opt_flag_value = m_material_mng->modificationFlags();
    bool use_incremental = (opt_flag_value & (int)eModificationFlags::IncrementalRecompute) != 0;
    if (use_incremental)
      m_cell_family->_internalApi()->addSourceConnectivity(this);
  }
  if (!is_continue) {
    Int32 max_local_id = m_cell_family->maxLocalId();
    m_container->checkResize(max_local_id + 1);
    m_container->m_environment.endCreate(is_continue);
    m_container->m_material.endCreate(is_continue);
  }

  ConstArrayView<MeshMaterial*> materials = m_material_mng->trueMaterials();
  const Int32 nb_mat = materials.size();
  m_environment_for_materials.resize(nb_mat);
  for (Int32 i = 0; i < nb_mat; ++i)
    m_environment_for_materials[i] = materials[i]->trueEnvironment()->componentId();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
_addCells(Int16 component_id, ConstArrayView<Int32> cell_ids, ConstituentContainer& component)
{
  Array<Int16>& nb_component = component.m_nb_component_as_array;
  Array<Int32>& component_index = component.m_component_index_as_array;
  Array<Int16>& component_list = component.m_component_list_as_array;
  for (Int32 id : cell_ids) {
    CellLocalId cell_id(id);
    const Int16 n = nb_component[cell_id];
    if (n == 0) {
      // Pas encore de milieu. On ajoute juste le nouveau milieu
      Int32 pos = component_list.size();
      component_index[cell_id] = pos;
      component_list.add(component_id);
    }
    else {
      // Alloue de la place pour 1 milieu suppl'émentaire et recopie les
      // anciennes valeurs.
      // TODO: cela laisse des trous dans la liste qu'il faudra supprimer
      // via un compactage.
      Int32 current_pos = component_index[cell_id];
      Int32 pos = component_list.size();
      component_index[cell_id] = pos;
      component_list.addRange(component_id, n + 1);
      ArrayView<Int16> current_values(n, &component_list[current_pos]);
      ArrayView<Int16> new_values(n, &component_list[pos]);
      new_values.copy(current_values);
    }
    ++nb_component[cell_id];
  }
  component.notifyUpdateConnectivityList();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
_removeCells(Int16 component_id, ConstArrayView<Int32> cell_ids, ConstituentContainer& component)
{
  Array<Int16>& nb_component = component.m_nb_component_as_array;
  Array<Int32>& component_index = component.m_component_index_as_array;
  Array<Int16>& component_list = component.m_component_list_as_array;
  for (Int32 id : cell_ids) {
    CellLocalId cell_id(id);
    const Int32 current_pos = component_index[cell_id];
    const Int32 n = nb_component[cell_id];
    ArrayView<Int16> current_values(n, &component_list[current_pos]);
    // Enlève de la liste le milieu supprimé
    _removeValueAndKeepOrder(current_values, component_id);
    // Met une valeur invalide pour indiquer que l'emplacement est libre
    current_values[n - 1] = (-1);
    --nb_component[cell_id];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
addCellsToEnvironment(Int16 env_id, ConstArrayView<Int32> cell_ids)
{
  _addCells(env_id, cell_ids, m_container->m_environment);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
removeCellsToEnvironment(Int16 env_id, ConstArrayView<Int32> cell_ids)
{
  _removeCells(env_id, cell_ids, m_container->m_environment);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
addCellsToMaterial(Int16 mat_id, ConstArrayView<Int32> cell_ids)
{
  _addCells(mat_id, cell_ids, m_container->m_material);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentConnectivityList::
removeCellsToMaterial(Int16 mat_id, ConstArrayView<Int32> cell_ids)
{
  _removeCells(mat_id, cell_ids, m_container->m_material);
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

Int16 ConstituentConnectivityList::
cellNbMaterial(CellLocalId cell_id, Int16 env_id)
{
  Int16 nb_mat = 0;
  ArrayView<Int16> mats = m_container->m_material.components(cell_id);
  for (Int16 mat_id : mats) {
    Int16 current_id = m_environment_for_materials[mat_id];
    if (current_id == env_id)
      ++nb_mat;
  }
  return nb_mat;
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

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
