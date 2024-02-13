// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LoadBalanceMng.cc                                           (C) 2000-2024 */
/*                                                                           */
/* Module standard de description du probleme pour l'equilibrage de charge.  */
/* Est utilise par le MeshPartioner comme entree.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/LoadBalanceMng.h"

#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/MultiArray2.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IModule.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/CommonVariables.h"
#include "arcane/core/VariableCollection.h"

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

/*!
 * \brief Classe de gestion des criteres de partitionnement.
 *
 * Sert essentiellement à fournir les informations mémoire associées à chaque
 * entité.
 * Permet d'obtenir le numéro d'entité à partir de son nom.
 * \note This class is not thread safe.
 */
class CriteriaMng
{
 public:

  //! Construction en fonction du IVariableMng.
  explicit CriteriaMng(IVariableMng* varMng);
  ~CriteriaMng() {}

  //! Ajoute une entité et lui attribue un numéro. Un même nom n'est pas dupliqué.
  Integer addEntity(const String& entity);

  //! Précalcule la mémoire associée à chaque entité. (estimation)
  void computeMemory();

  //! Retourne la mémoire totale associée à une entité.
  Real getOverallMemory(const String& entity) const;
  Real getOverallMemory(Integer offset) const
  {
    return m_overall_memory[offset];
  }
  Real getOverallMemory(const Cell& cell)
  {
    _computeMemCell(cell);
    return m_buffer.overall_memory;
  }

  //! Retourne la mémoire "résidente" (à transférer) associée à une entité.
  Real getResidentMemory(const String& entity) const;
  Real getResidentMemory(Integer offset) const
  {
    return m_resident_memory[offset];
  }
  Real getResidentMemory(const Cell& cell)
  {
    _computeMemCell(cell);
    return m_buffer.resident_memory;
  }

  //! Gestion des entités et de leur nom.
  Integer operator[](const String& entity) const
  {
    return _findEntity(entity);
  }
  const String& operator[](unsigned int i) const
  {
    return m_family_names[i];
  }

 private:

  Integer _findEntity(const String& entity) const;
  void _computeMemCell(Cell cell);

  //! Calcule de la contribution d'un entité sur les mailles adjacentes.
  template <typename ItemKind>
  Real _computeMemContrib(ItemConnectedListViewTypeT<ItemKind> list)
  {
    Real contrib = 0.0;
    //ItemEnumeratorT<ItemKind> iterator = list.enumerator();
    for (const auto& item : list) {
      contrib += 1.0 / (Real)(item.nbCell());
    }
    return contrib;
  }

  IVariableMng* m_variable_mng;
  UniqueArray<String> m_family_names;
  UniqueArray<Int32> m_overall_memory;
  UniqueArray<Int32> m_resident_memory;

  //! Système de cache pour l'accès aux mémoires relatives à une maille.
  struct MemInfo
  {
    Int32 id = -1;
    Real overall_memory = 0;
    Real resident_memory = 0;
  };
  MemInfo m_buffer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CriteriaMng::
CriteriaMng(IVariableMng* varMng)
: m_variable_mng(varMng)
, m_family_names(IK_Unknown + 1, "__special__") // +1 car certaines variables sont associées à IK_Unknown
{
  m_family_names[IK_Cell] = "Cell";
  m_family_names[IK_Face] = "Face";
  m_family_names[IK_Edge] = "Edge";
  m_family_names[IK_Node] = "Node";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CriteriaMng::
addEntity(const String& entity)
{
  Integer pos;
  pos = _findEntity(entity);
  if (pos < 0) {
    pos = m_family_names.size();
    m_family_names.add(entity);
  }
  return pos;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Calcul de la consommation mémoire pour chaque type d'entité.
// Les mailles bénéficient ensuite des contributions des autres entités adjacentes.
void CriteriaMng::
computeMemory()
{
  Int32 length = m_family_names.size();
  m_overall_memory.resize(length);
  m_resident_memory.resize(length);
  m_buffer.id = -1;
  m_overall_memory.fill(0);
  m_resident_memory.fill(0);

  // For each variable, compute the size for one object.
  for (VariableCollectionEnumerator vc(m_variable_mng->usedVariables()); ++vc;) {
    const IVariable* var = *vc;
    Integer memory = 0;
    try {
      if (var->dataType() != DT_String)
        memory = dataTypeSize(var->dataType());
    }
    catch (const ArgumentException&) {
      memory = 0; // Cannot know memory used for that ...
      continue;
    }
    Int32 family_index = -1;
    Integer kind = var->itemKind();
    if (kind == IK_Particle) { // Not the same counter for all items
      family_index = _findEntity(var->itemFamilyName());
      if (family_index >= 0) {
        m_overall_memory[family_index] += memory;
      }
    }
    m_overall_memory[kind] += memory;

    int properties = var->property();
    if ((properties & IVariable::PNoExchange) ||
        (properties & IVariable::PNoNeedSync) ||
        (properties & IVariable::PTemporary) ||
        (properties & IVariable::PSubDomainPrivate)) {
      continue;
    }
    m_resident_memory[kind] += memory;
    if (family_index >= 0) {
      m_resident_memory[family_index] += memory;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CriteriaMng::
_findEntity(const String& entity) const
{
  for (int i = 0; i < m_family_names.size(); ++i) {
    if (m_family_names[i] == entity)
      return i;
  }
  return -1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real CriteriaMng::
getOverallMemory(const String& entity) const
{
  Integer pos = _findEntity(entity);
  if (pos >= 0)
    return getOverallMemory(pos);
  else
    return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real CriteriaMng::
getResidentMemory(const String& entity) const
{
  Integer pos = _findEntity(entity);
  if (pos >= 0)
    return getResidentMemory(pos);
  else
    return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Pour les mailles, on calcule la contribution mémoire des noeuds, aretes et faces.
void CriteriaMng::
_computeMemCell(Cell cell)
{
  Real contrib;
  if (cell.localId() == m_buffer.id) // already computed
    return;
  m_buffer.id = cell.localId();
  m_buffer.overall_memory = m_overall_memory[IK_Cell];
  m_buffer.resident_memory = m_resident_memory[IK_Cell];

  contrib = _computeMemContrib<Node>(cell.nodes());
  m_buffer.overall_memory += contrib * m_overall_memory[IK_Node];
  m_buffer.resident_memory += contrib * m_resident_memory[IK_Node];

  contrib = _computeMemContrib<Face>(cell.faces());
  m_buffer.overall_memory += contrib * m_overall_memory[IK_Face];
  m_buffer.resident_memory += contrib * m_resident_memory[IK_Face];

  contrib = _computeMemContrib<Edge>(cell.edges());
  m_buffer.overall_memory += contrib * m_overall_memory[IK_Edge];
  m_buffer.resident_memory += contrib * m_resident_memory[IK_Edge];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LoadBalanceMng::
LoadBalanceMng(ISubDomain* sd, bool mass_as_criterion)
: m_mesh_handle(sd->defaultMeshHandle())
, m_criteria(new CriteriaMng(sd->variableMng()))
, m_mass_criterion(mass_as_criterion)
{
  reset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
reset()
{
  m_mass_vars.clear();
  m_comm_vars.clear();
  m_event_vars.resize(1); // First slot booked by MemoryOverAll

  endAccess();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
initAccess(IMesh* mesh)
{
  if (!mesh)
    ARCANE_FATAL("Null mesh");
  m_mesh_handle = mesh->handle();
  int vflags = IVariable::PExecutionDepend | IVariable::PNoDump | IVariable::PTemporary;
  m_cell_new_owner = std::make_unique<VariableCellInt32>(VariableBuildInfo(m_mesh_handle, "CellFamilyNewOwner", IVariable::PExecutionDepend | IVariable::PNoDump));
  m_comm_costs = new VariableFaceReal(VariableBuildInfo(m_mesh_handle, "LbMngCommCost", vflags));
  m_mass_over_weigth = new VariableCellReal(VariableBuildInfo(m_mesh_handle, "LbMngOverallMass", vflags));
  m_mass_res_weight = new VariableCellReal(VariableBuildInfo(m_mesh_handle, "LbMngResidentMass", vflags));
  m_event_weights = new VariableCellArrayReal(VariableBuildInfo(m_mesh_handle, "LbMngMCriteriaWgt", vflags));
  m_comm_costs->fill(1);
  m_mass_over_weigth->fill(1);
  m_mass_res_weight->fill(1);

  mesh->traceMng()->info() << "LoadBalanceMng::initAccess(): use_memory=" << m_mass_criterion;

  if (m_compute_comm || m_mass_criterion) { // Memory useful only for communication cost or mass lb criterion
    m_criteria->computeMemory();
    _computeResidentMass();
  }
  if (m_compute_comm)
    _computeComm();
  if (m_mass_criterion)
    _computeOverallMass();
  _computeEvents();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
endAccess()
{
  m_event_weights = nullptr;
  m_mass_res_weight = nullptr;
  m_mass_over_weigth = nullptr;
  m_comm_costs = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
addMass(VariableCellInt32& count, const String& entity)
{
  StoreIProxyItemVariable cvar(count.variable(), m_criteria->addEntity(entity));
  m_mass_vars.add(cvar);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
addCriterion(VariableCellInt32& count)
{
  StoreIProxyItemVariable cvar(count.variable());
  m_event_vars.add(cvar);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
addCriterion(VariableCellReal& count)
{
  //std::cerr << "Adding var " << count.variable()->fullName() << " ref # " << count.variable()->nbReference() << std::endl;
  StoreIProxyItemVariable cvar(count.variable());
  //std::cerr << "Adding var (2)" << count.variable()->fullName() << " ref # " << count.variable()->nbReference() << std::endl;
  m_event_vars.add(cvar);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
addCommCost(VariableFaceInt32& count, const String& entity)
{
  StoreIProxyItemVariable cvar(count.variable(), m_criteria->addEntity(entity));
  m_comm_vars.add(cvar);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer LoadBalanceMng::
nbCriteria()
{
  Integer count;

  count = m_event_vars.size();
  count -= ((m_mass_criterion) ? 0 : 1); // First event is mass !
  count += ((m_nb_criterion) ? 1 : 0);
  return count;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
notifyEndPartition()
{
  IMesh* mesh = m_mesh_handle.mesh();
  if (m_cell_new_owner)
    m_cell_new_owner->fill(mesh->parallelMng()->commRank(), mesh->ownCells());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
_computeOverallMass()
{
  ENUMERATE_CELL (icell, m_mesh_handle.mesh()->ownCells()) {
    (*m_mass_over_weigth)[icell] = m_criteria->getOverallMemory(*icell);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
_computeResidentMass()
{
  ENUMERATE_CELL (icell, m_mesh_handle.mesh()->ownCells()) {
    (*m_mass_res_weight)[icell] = m_criteria->getResidentMemory(*icell);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
_computeComm()
{
  IMesh* mesh = m_mesh_handle.mesh();
  Integer penalty = 2; // How many times we do synchronization ?

  if (m_comm_vars.size())
    m_comm_costs->fill(0);

  for (int i = 0; i < m_comm_vars.size(); ++i) {
    StoreIProxyItemVariable& commvar = m_comm_vars[i];
    ENUMERATE_FACE (iface, mesh->ownFaces()) {
      (*m_comm_costs)[iface] += commvar[iface] * m_criteria->getResidentMemory(commvar.getPos());
    }
  }
  if (cellCommContrib()) {
    ENUMERATE_CELL (icell, mesh->ownCells()) {
      Real mem = (*m_mass_res_weight)[icell];
      for (Face face : icell->faces()) {
        (*m_comm_costs)[face] += mem * penalty;
      }
    }
  }

  IVariable* ivar;
  // Make sure that ghosts contribution is used
  ivar = m_comm_costs->variable();
  ivar->itemFamily()->reduceFromGhostItems(ivar, Parallel::ReduceSum);
  m_comm_costs->synchronize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
_computeEvents()
{
  m_event_weights->resize(nbCriteria());

  ArrayView<StoreIProxyItemVariable> eventVars = m_event_vars;
  if (m_mass_criterion) {
    StoreIProxyItemVariable cvar(m_mass_over_weigth->variable());
    m_event_vars[0] = cvar;
  }
  else {
    eventVars = m_event_vars.subView(1, m_event_vars.size());
  }

  for (Integer i = 0; i < eventVars.size(); ++i) {
    ENUMERATE_CELL (icell, m_mesh_handle.mesh()->ownCells()) {
      Integer count = i;
      if (m_nb_criterion) {
        count += 1;
        (*m_event_weights)[icell][0] = 1;
      }
      (*m_event_weights)[icell][count] = eventVars[i][icell];
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ILoadBalanceMng*
arcaneCreateLoadBalanceMng(ISubDomain* sd)
{
  ILoadBalanceMng* lbm = new LoadBalanceMng(sd);
  return lbm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
