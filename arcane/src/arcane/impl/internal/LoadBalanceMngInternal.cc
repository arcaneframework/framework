// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LoadBalanceMngInternal.cc                                   (C) 2000-2024 */
/*                                                                           */
/* Classe interne gérant l'équilibre de charge des maillages.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/LoadBalanceMngInternal.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/MultiArray2.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IModule.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/ItemEnumerator.h"


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * @brief Implementation "nulle" de l'interface IProxyItemVariable.
 * Permet d'avoir une variable référence unitaire.
 */
class ProxyItemVariableNull
: public IProxyItemVariable
{
 public:

  ProxyItemVariableNull() {}
  ~ProxyItemVariableNull() {}

  Real operator[](ItemEnumerator) const
  {
    return 1;
  }
  Integer getPos() const
  {
    return 0;
  }

 protected:

  void deleteMe() {}
};

//! Représentant nul de la classe précédente.
static ProxyItemVariableNull nullProxy;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * @brief Implementation de l'interface IProxyItemVariable.
 *
 * Templaté par le type de données associé à la variable.
 */
template <typename DataType> class ProxyItemVariable
: public IProxyItemVariable
{
 public:

  ProxyItemVariable(IVariable* var, Integer pos = 0)
  : m_var(var)
  , m_pos(pos)
  {
  }
  ~ProxyItemVariable() {}

  Real operator[](ItemEnumerator i) const
  {
    return static_cast<Real>(m_var[i]);
  }
  Integer getPos() const
  {
    return m_pos;
  }

 private:

  ItemVariableScalarRefT<DataType> m_var;
  Integer m_pos;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * @brief Factory pour la construction de variable proxy.
 *
 * Actuellement, seules les variables de type Real et Int32 ont une signification.
 */
IProxyItemVariable* StoreIProxyItemVariable::
proxyItemVariableFactory(IVariable* var, Integer pos)
{
  if (!var)
    return &nullProxy;
  switch (var->dataType()) {
  case DT_Real:
    return new ProxyItemVariable<Real>(var, pos);
    break;
  case DT_Int32:
    return new ProxyItemVariable<Int32>(var, pos);
    break;
  default:
    // TODO : throw an exception
    return &nullProxy;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LoadBalanceMngInternal::
LoadBalanceMngInternal(bool mass_as_criterion)
: m_default_mass_criterion(mass_as_criterion)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMngInternal::
reset(IMesh* mesh)
{
  m_mesh_criterion[mesh].resetCriteria();
  if(mesh) {
    mesh->traceMng()->debug() << "LoadBalanceInternal -- Mesh : " << mesh->name() << " -- reset()";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMngInternal::
initAccess(IMesh* mesh)
{
  if (!mesh)
    ARCANE_FATAL("Null mesh");

  m_mesh_handle = mesh->handle();

  CriteriaMng& mesh_criterion = m_mesh_criterion[mesh];
  mesh_criterion.init(mesh);
  mesh_criterion.defaultMassCriterion(m_default_mass_criterion);

  mesh->traceMng()->info() << "LoadBalanceMngInternal::initAccess(): use_memory=" << mesh_criterion.useMassAsCriterion();

  mesh_criterion.computeCriteria();

  mesh->traceMng()->debug() << "LoadBalanceInternal -- Mesh : " << mesh->name() << " -- initAccess()";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMngInternal::
endAccess()
{
  IMesh* mesh = m_mesh_handle.mesh();
  if (!mesh)
    ARCANE_FATAL("Null mesh");

  m_mesh_criterion[mesh].clearVariables();

  mesh->traceMng()->debug() << "LoadBalanceInternal -- Mesh : " << mesh->name() << " -- clearVariables()";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMngInternal::
addMass(VariableCellInt32& count, IMesh* mesh, const String& entity)
{
  StoreIProxyItemVariable cvar(count.variable(), m_mesh_criterion[mesh].addEntity(entity));
  m_mesh_criterion[mesh].addMass(cvar);
  mesh->traceMng()->debug() << "Set mass (name=" << count.name() << ") criterion to mesh (name=" << mesh->name() << ")";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMngInternal::
addCriterion(VariableCellInt32& count, IMesh* mesh)
{
  StoreIProxyItemVariable cvar(count.variable());
  m_mesh_criterion[mesh].addCriterion(cvar);
  mesh->traceMng()->debug() << "Set criterion (name=" << count.name() << ") criterion to mesh (name=" << mesh->name() << ")";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMngInternal::
addCriterion(VariableCellReal& count, IMesh* mesh)
{
  //std::cerr << "Adding var " << count.variable()->fullName() << " ref # " << count.variable()->nbReference() << std::endl;
  StoreIProxyItemVariable cvar(count.variable());
  //std::cerr << "Adding var (2)" << count.variable()->fullName() << " ref # " << count.variable()->nbReference() << std::endl;
  m_mesh_criterion[mesh].addCriterion(cvar);
  mesh->traceMng()->debug() << "Set criterion (name=" << count.name() << ") criterion to mesh (name=" << mesh->name() << ")";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMngInternal::
addCommCost(VariableFaceInt32& count, IMesh* mesh, const String& entity)
{
  StoreIProxyItemVariable cvar(count.variable(), m_mesh_criterion[mesh].addEntity(entity));
  m_mesh_criterion[mesh].addCommCost(cvar);
  mesh->traceMng()->debug() << "Set CommCost (name=" << count.name() << ") criterion to mesh (name=" << mesh->name() << ")";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer LoadBalanceMngInternal::
nbCriteria(IMesh* mesh)
{
  return m_mesh_criterion[mesh].nbCriteria();
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMngInternal::
setMassAsCriterion(IMesh* mesh, bool active)
{
  m_mesh_criterion[mesh].setMassCriterion(active);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMngInternal::
setNbCellsAsCriterion(IMesh* mesh, bool active)
{
  m_mesh_criterion[mesh].setNbCellsAsCriterion(active);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMngInternal::
setCellCommContrib(IMesh* mesh, bool active)
{
  m_mesh_criterion[mesh].setCellCommContrib(active);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool LoadBalanceMngInternal::
cellCommContrib(IMesh* mesh) const
{
  if(m_mesh_criterion.find(mesh) == m_mesh_criterion.end()){
    return true;
  }
  return m_mesh_criterion.at(mesh).cellCommContrib();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMngInternal::
setComputeComm(IMesh* mesh, bool active)
{
  m_mesh_criterion[mesh].setComputeComm(active);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VariableFaceReal& LoadBalanceMngInternal::
commCost(IMesh* mesh) const
{
  if(m_mesh_criterion.find(mesh) == m_mesh_criterion.end()){
    ARCANE_FATAL("CriteriaMng not found for the targeted mesh (name={0}). Use MeshCriteriaLoadBalanceMng class or call initAccess() before.", (mesh ? mesh->name() : "Unknown"));
  }
  if (!m_mesh_criterion.at(mesh).isInit()) {
    ARCANE_FATAL("CriteriaMng is not initialized for the targeted mesh (name={0}). Use MeshCriteriaLoadBalanceMng class or call initAccess() before.", (mesh ? mesh->name() : "Unknown"));
  }
  return m_mesh_criterion.at(mesh).commCost();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VariableCellReal& LoadBalanceMngInternal::
massWeight(IMesh* mesh) const
{
  if(m_mesh_criterion.find(mesh) == m_mesh_criterion.end()){
    ARCANE_FATAL("CriteriaMng not found for the targeted mesh (name={0}). Use MeshCriteriaLoadBalanceMng class or call initAccess() before.", (mesh ? mesh->name() : "Unknown"));
  }
  if (!m_mesh_criterion.at(mesh).isInit()) {
    ARCANE_FATAL("CriteriaMng is not initialized for the targeted mesh (name={0}). Use MeshCriteriaLoadBalanceMng class or call initAccess() before.", (mesh ? mesh->name() : "Unknown"));
  }
  return m_mesh_criterion.at(mesh).massWeight();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VariableCellReal& LoadBalanceMngInternal::
massResWeight(IMesh* mesh) const
{
  if(m_mesh_criterion.find(mesh) == m_mesh_criterion.end()){
    ARCANE_FATAL("CriteriaMng not found for the targeted mesh (name={0}). Use MeshCriteriaLoadBalanceMng class or call initAccess() before.", (mesh ? mesh->name() : "Unknown"));
  }
  if (!m_mesh_criterion.at(mesh).isInit()) {
    ARCANE_FATAL("CriteriaMng is not initialized for the targeted mesh (name={0}). Use MeshCriteriaLoadBalanceMng class or call initAccess() before.", (mesh ? mesh->name() : "Unknown"));
  }
  return m_mesh_criterion.at(mesh).massResWeight();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VariableCellArrayReal& LoadBalanceMngInternal::
mCriteriaWeight(IMesh* mesh) const
{
  if(m_mesh_criterion.find(mesh) == m_mesh_criterion.end()){
    ARCANE_FATAL("CriteriaMng not found for the targeted mesh (name={0}). Use MeshCriteriaLoadBalanceMng class or call initAccess() before.", (mesh ? mesh->name() : "Unknown"));
  }
  if (!m_mesh_criterion.at(mesh).isInit()) {
    ARCANE_FATAL("CriteriaMng is not initialized for the targeted mesh (name={0}). Use MeshCriteriaLoadBalanceMng class or call initAccess() before.", (mesh ? mesh->name() : "Unknown"));
  }
  return m_mesh_criterion.at(mesh).criteriaWeight();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMngInternal::
notifyEndPartition()
{
  IMesh* mesh = m_mesh_handle.mesh();
  m_mesh_criterion[mesh].fillCellNewOwner();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
