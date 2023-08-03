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
  for (Int32 id : cell_ids)
    ++nb_env[CellLocalId(id)];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentConnectivityList::
removeCellsToEnvironment(Int16 env_id, ConstArrayView<Int32> cell_ids)
{
  VariableCellInt16& nb_env = m_container->m_nb_environment;
  for (Int32 id : cell_ids)
    --nb_env[CellLocalId(id)];
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
