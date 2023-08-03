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

class ComponentConnectivityList::Container
{
 public:

  Container(const MeshHandle& mesh, const String& var_name)
  : m_var_name(var_name)
  , m_nb_environment(VariableBuildInfo(mesh, var_name + "ComponentNbEnvironment", IVariable::PPrivate))
  , m_environment_index(VariableBuildInfo(mesh, var_name + "ComponentEnviromentIndex", IVariable::PPrivate))
  , m_environment_list(VariableBuildInfo(mesh, var_name + "ComponentEnvironmentList", IVariable::PPrivate))
  , m_environment_list_array(m_environment_list._internalTrueData()->_internalDeprecatedValue())
  {
  }

 public:

  String m_var_name;

  VariableCellInt16 m_nb_environment;
  VariableCellInt32 m_environment_index;
  VariableArrayInt16 m_environment_list;

  VariableArrayInt16::ContainerType& m_environment_list_array;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentConnectivityList::
ComponentConnectivityList(MeshMaterialMng* mm)
: TraceAccessor(mm->traceMng())
, m_material_mng(mm)
, m_container(new Container(mm->meshHandle(), mm->name()))
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
  if (!is_continue)
    m_container->m_nb_environment.fill(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentConnectivityList::
addCellsToEnvironment(Int16 env_id, ConstArrayView<Int32> cell_ids)
{
  VariableCellInt16& nb_env = m_container->m_nb_environment;
  VariableCellInt32& env_index = m_container->m_environment_index;
  VariableArrayInt16::ContainerType& env_list = m_container->m_environment_list_array;
  for (Int32 id : cell_ids) {
    CellLocalId cell_id(id);
    const Int16 n = nb_env[cell_id];
    if (n == 0) {
      // Pas encore de milieu. On ajoute juste le nouveau milieu
      Int32 pos = env_list.size();
      env_index[cell_id] = pos;
      env_list.add(env_id);
    }
    else {
      // Alloue de la place pour 1 milieu suppl'émentaire et recopie les
      // anciennes valeurs.
      // TODO: cela laisse des trous dans la liste qu'il faudra supprimer
      // via un compactage.
      Int32 current_pos = env_index[cell_id];
      Int32 pos = env_list.size();
      env_index[cell_id] = pos;
      env_list.addRange(env_id, n + 1);
      ArrayView<Int16> current_values(n, &env_list[current_pos]);
      ArrayView<Int16> new_values(n, &env_list[pos]);
      new_values.copy(current_values);
    }
    ++nb_env[cell_id];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentConnectivityList::
removeCellsToEnvironment(Int16 env_id, ConstArrayView<Int32> cell_ids)
{
  VariableCellInt16& nb_env = m_container->m_nb_environment;
  VariableCellInt32& env_index = m_container->m_environment_index;
  VariableArrayInt16::ContainerType& env_list = m_container->m_environment_list_array;
  for (Int32 id : cell_ids) {
    CellLocalId cell_id(id);
    const Int32 current_pos = env_index[cell_id];
    const Int32 n = nb_env[cell_id];
    ArrayView<Int16> current_values(n, &env_list[current_pos]);
    // Enlève de la liste le milieu supprimé
    _removeValueAndKeepOrder(current_values, env_id);
    // Met une valeur invalide pour indiquer que l'emplacement est libre
    current_values[n - 1] = (-1);
    --nb_env[cell_id];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VariableCellInt16& ComponentConnectivityList::
cellNbEnvironment() const
{
  return m_container->m_nb_environment;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
