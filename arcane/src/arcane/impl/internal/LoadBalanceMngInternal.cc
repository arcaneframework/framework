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

  explicit ProxyItemVariable(IVariable* var, Integer pos = 0)
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

void CriteriaMng::
init(IMesh* mesh)
{
  MeshHandle mesh_handle = mesh->handle();
  int vflags = IVariable::PExecutionDepend | IVariable::PNoDump | IVariable::PTemporary;

  m_cell_new_owner = new VariableCellInt32(VariableBuildInfo(mesh_handle, "CellFamilyNewOwner", IVariable::PExecutionDepend | IVariable::PNoDump));
  m_comm_costs = new VariableFaceReal(VariableBuildInfo(mesh_handle, "LbMngCommCost", vflags));
  m_mass_over_weight = new VariableCellReal(VariableBuildInfo(mesh_handle, "LbMngOverallMass", vflags));
  m_mass_res_weight = new VariableCellReal(VariableBuildInfo(mesh_handle, "LbMngResidentMass", vflags));
  m_event_weights = new VariableCellArrayReal(VariableBuildInfo(mesh_handle, "LbMngMCriteriaWgt", vflags));

  m_comm_costs->fill(1);
  m_mass_over_weight->fill(1);
  m_mass_res_weight->fill(1);
  m_is_init = true;
  m_mesh = mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CriteriaMng::
resetCriteria()
{
  m_mass_vars.clear();
  m_comm_vars.clear();
  m_event_vars.resize(1); // First slot booked by MemoryOverAll

  clearVariables();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CriteriaMng::
clearVariables()
{
  m_event_weights = nullptr;
  m_mass_res_weight = nullptr;
  m_mass_over_weight = nullptr;
  m_comm_costs = nullptr;
  m_is_init = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CriteriaMng::
nbCriteria()
{
  Integer count;

  count = m_event_vars.size();
  count -= ((m_use_mass_as_criterion) ? 0 : 1); // First event is mass !
  count += ((m_nb_cells_as_criterion) ? 1 : 0);
  return count;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayView<StoreIProxyItemVariable> CriteriaMng::
criteria()
{
  if (m_use_mass_as_criterion) {
    StoreIProxyItemVariable cvar(m_mass_over_weight->variable());
    m_event_vars[0] = cvar;
    return m_event_vars;
  }
  return m_event_vars.subView(1, m_event_vars.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CriteriaMng::
computeCriteria()
{
  if (needComputeComm() || useMassAsCriterion()) { // Memory useful only for communication cost or mass lb criterion
    m_criteria->computeMemory(m_mesh->variableMng());
    _computeResidentMass();
  }
  if (needComputeComm()) {
    _computeComm();
  }
  if (useMassAsCriterion()) {
    _computeOverallMass();
  }
  _computeEvents();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CriteriaMng::
_computeOverallMass()
{
  VariableCellReal& mass_over_weigth = *m_mass_over_weight;
  ENUMERATE_CELL (icell, m_mesh->ownCells()) {
    mass_over_weigth[icell] = m_criteria->getOverallMemory(*icell);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CriteriaMng::
_computeComm()
{
  VariableFaceReal& comm_costs = *m_comm_costs;
  VariableCellReal& mass_res_weight = *m_mass_res_weight;

  Integer penalty = 2; // How many times we do synchronization ?

  if (!m_comm_vars.empty())
    comm_costs.fill(0);

  for (auto& commvar : m_comm_vars) {
    ENUMERATE_FACE (iface, m_mesh->ownFaces()) {
      comm_costs[iface] += commvar[iface] * m_criteria->getResidentMemory(commvar.getPos());
    }
  }
  if (m_cell_comm) {
    ENUMERATE_CELL (icell, m_mesh->ownCells()) {
      Real mem = mass_res_weight[icell];
      for (Face face : icell->faces()) {
        comm_costs[face] += mem * penalty;
      }
    }
  }

  // Make sure that ghosts contribution is used
  IVariable* ivar = m_comm_costs->variable();
  ivar->itemFamily()->reduceFromGhostItems(ivar, Parallel::ReduceSum);
  m_comm_costs->synchronize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CriteriaMng::
_computeResidentMass()
{
  VariableCellReal& mass_res_weight = *m_mass_res_weight;
  ENUMERATE_CELL (icell, m_mesh->ownCells()) {
    mass_res_weight[icell] = m_criteria->getResidentMemory(*icell);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CriteriaMng::
_computeEvents()
{
  m_event_weights->resize(nbCriteria());

  ArrayView<StoreIProxyItemVariable> event_vars = criteria();

  VariableCellArrayReal& event_weights = *(m_event_weights);

  for (Integer i = 0; i < event_vars.size(); ++i) {
    ENUMERATE_CELL (icell, m_mesh->ownCells()) {
      Integer count = i;
      if (m_nb_cells_as_criterion) {
        count += 1;
        event_weights(icell, 0) = 1;
      }
      event_weights(icell, count) = event_vars[i][icell];
    }
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

  mesh->traceMng()->info() << "LoadBalanceMngInternal::initAccess(): use_memory=" << mesh_criterion.useMassAsCriterion()
                           << " nb_criteria=" << mesh_criterion.nbCriteria();

  // Si aucun critère n'est défini, utilise le nombre de mailles.
  // Il faut au moins un critère sinon il n'y aura pas de poids dans le graphe de partitionnement.
  if (mesh_criterion.nbCriteria() == 0)
    mesh_criterion.setNbCellsAsCriterion(true);

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
    return;

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
