// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentConnectivityList.cc                                (C) 2000-2023 */
/*                                                                           */
/* Gestion des listes de connectivité des milieux et matériaux.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/ComponentConnectivityList.h"

#include "arcane/core/internal/IDataInternal.h"

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

class ComponentConnectivityList::ComponentContainer
{
 public:

  ComponentContainer(const MeshHandle& mesh, const String& var_base_name)
  : m_nb_component(VariableBuildInfo(mesh, var_base_name + "NbComponent", IVariable::PPrivate))
  , m_component_index(VariableBuildInfo(mesh, var_base_name + "Index", IVariable::PPrivate))
  , m_component_list(VariableBuildInfo(mesh, var_base_name + "List", IVariable::PPrivate))
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

 public:

  VariableCellInt16 m_nb_component;
  VariableCellInt32 m_component_index;
  VariableArrayInt16 m_component_list;
  VariableArrayInt16::ContainerType& m_component_list_as_array;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ComponentConnectivityList::Container
{
 public:

  Container(const MeshHandle& mesh, const String& var_base_name)
  : m_environment(mesh, var_base_name + String("ComponentEnvironment"))
  , m_material(mesh, var_base_name + String("ComponentMaterial"))
  {
  }

 public:

  ComponentContainer m_environment;
  ComponentContainer m_material;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentConnectivityList::
ComponentConnectivityList(MeshMaterialMng* mm)
: TraceAccessor(mm->traceMng())
, m_material_mng(mm)
, m_container(new Container(mm->meshHandle(), String("ComponentEnviroment") + mm->name()))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentConnectivityList::
~ComponentConnectivityList()
{
  delete m_container;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentConnectivityList::
endCreate(bool is_continue)
{
  if (!is_continue) {
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

void ComponentConnectivityList::
_addCells(Int16 component_id, ConstArrayView<Int32> cell_ids, ComponentContainer& component)
{
  VariableCellInt16& nb_component = component.m_nb_component;
  VariableCellInt32& component_index = component.m_component_index;
  VariableArrayInt16::ContainerType& component_list = component.m_component_list_as_array;
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
  component.m_component_list.updateFromInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentConnectivityList::
_removeCells(Int16 component_id, ConstArrayView<Int32> cell_ids, ComponentContainer& component)
{
  VariableCellInt16& nb_component = component.m_nb_component;
  VariableCellInt32& component_index = component.m_component_index;
  VariableArrayInt16::ContainerType& component_list = component.m_component_list_as_array;
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

void ComponentConnectivityList::
addCellsToEnvironment(Int16 env_id, ConstArrayView<Int32> cell_ids)
{
  _addCells(env_id, cell_ids, m_container->m_environment);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentConnectivityList::
removeCellsToEnvironment(Int16 env_id, ConstArrayView<Int32> cell_ids)
{
  _removeCells(env_id, cell_ids, m_container->m_environment);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentConnectivityList::
addCellsToMaterial(Int16 mat_id, ConstArrayView<Int32> cell_ids)
{
  _addCells(mat_id, cell_ids, m_container->m_material);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentConnectivityList::
removeCellsToMaterial(Int16 mat_id, ConstArrayView<Int32> cell_ids)
{
  _removeCells(mat_id, cell_ids, m_container->m_material);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VariableCellInt16& ComponentConnectivityList::
cellNbEnvironment() const
{
  return m_container->m_environment.m_nb_component;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VariableCellInt16& ComponentConnectivityList::
cellNbMaterial() const
{
  return m_container->m_material.m_nb_component;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int16 ComponentConnectivityList::
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

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
