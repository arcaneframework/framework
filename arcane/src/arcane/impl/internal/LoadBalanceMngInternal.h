// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LoadBalanceMngInternal.h                                    (C) 2000-2024 */
/*                                                                           */
/* Internal class managing the load balance of meshes.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNAL_LOADBALANCEMNGINTERNAL_H
#define ARCANE_IMPL_INTERNAL_LOADBALANCEMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/internal/ILoadBalanceMngInternal.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/CommonVariables.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IParallelMng.h"

#include "arcane/utils/ObjectImpl.h"
#include "arcane/utils/AutoRef.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ArgumentException.h"

#include <unordered_map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class PartitionerMemoryInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface proxy to access variables defining weights.
 *
 * It is independent of the variable type (Integer, Real).
 * It is automatically memory managed (via ObjectImpl).
 * Allows noting which object family the variable is associated with.
 */
class ARCANE_IMPL_EXPORT IProxyItemVariable
: public ObjectImpl
{
 public:
  virtual ~IProxyItemVariable() { }

  //! Access to the value associated with a mesh entity, in the form of a Real.
  virtual Real operator[](ItemEnumerator i) const =0;

  //! Access to the associated family number.
  virtual Integer getPos() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 *  @brief Class for accessing the proxy without dereferencing in the code.
 *
 *  It is independent of the variable type (Integer, Real).
 *  It is automatically memory managed (via AutoRefT).
 */
class ARCANE_IMPL_EXPORT StoreIProxyItemVariable
{
 public:

  StoreIProxyItemVariable(IVariable* var = nullptr, Integer pos = 0)
  {
    m_var = StoreIProxyItemVariable::proxyItemVariableFactory(var,pos);
  }

  StoreIProxyItemVariable(const StoreIProxyItemVariable& src) {
    if (m_var != src.m_var)
      m_var = src.m_var;
  }

  //! Access to the value associated with a mesh entity, in the form of a Real.
  Real operator[](ItemEnumerator i) const {
    return ((*m_var)[i]);
  }

  StoreIProxyItemVariable& operator=(const StoreIProxyItemVariable& src) {
    /* if (m_var != src.m_var) */
    m_var = src.m_var;
    return *this;
  }

  Integer getPos() const {
    return m_var->getPos();
  }

 protected:
  //! Factory for construction based on the initial variable type.
  static IProxyItemVariable* proxyItemVariableFactory(IVariable* var, Integer pos=0);

 private:
  //! Pointer to the variable.
  AutoRefT<IProxyItemVariable> m_var;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class for managing partitioning criteria.
 *
 * It essentially serves to provide memory information associated with each
 * entity.
 * Allows obtaining the entity number from its name.
 * \note This class is not thread safe.
 */
class PartitionerMemoryInfo
{
 public:

  //! Construction based on the IVariableMng.
  explicit PartitionerMemoryInfo()
  : m_family_names(IK_Unknown + 1, "__special__") // +1 because some variables are associated with IK_Unknown
  {
    m_family_names[IK_Cell] = "Cell";
    m_family_names[IK_Face] = "Face";
    m_family_names[IK_Edge] = "Edge";
    m_family_names[IK_Node] = "Node";
  }
  ~PartitionerMemoryInfo() = default;

  //! Adds an entity and assigns it a number. The same name is not duplicated.
  Integer addEntity(const String& entity)
  {
    Integer pos;
    pos = _findEntity(entity);
    if (pos < 0) {
      pos = m_family_names.size();
      m_family_names.add(entity);
    }
    return pos;
  }

  // Calculates the memory consumption for each entity type.
  // The meshes then benefit from the contributions of other adjacent entities.
  void computeMemory(IVariableMng* varMng)
  {
    Int32 length = m_family_names.size();
    m_overall_memory.resize(length);
    m_resident_memory.resize(length);
    m_buffer.id = -1;
    m_overall_memory.fill(0);
    m_resident_memory.fill(0);

    // For each variable, compute the size for one object.
    for (VariableCollectionEnumerator vc(varMng->usedVariables()); ++vc;) {
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

  //! Returns the total memory associated with an entity.
  Real getOverallMemory(const String& entity) const
  {
    Integer pos = _findEntity(entity);
    if (pos >= 0)
      return getOverallMemory(pos);
    else
      return 0;
  }
  Real getOverallMemory(Integer offset) const
  {
    return m_overall_memory[offset];
  }
  Real getOverallMemory(const Cell& cell)
  {
    _computeMemCell(cell);
    return m_buffer.overall_memory;
  }

  //! Returns the "resident" (to be transferred) memory associated with an entity.
  Real getResidentMemory(const String& entity) const
  {
    Integer pos = _findEntity(entity);
    if (pos >= 0)
      return getResidentMemory(pos);
    else
      return 0;
  }
  Real getResidentMemory(Integer offset) const
  {
    return m_resident_memory[offset];
  }
  Real getResidentMemory(const Cell& cell)
  {
    _computeMemCell(cell);
    return m_buffer.resident_memory;
  }

  //! Management of entities and their names.
  Integer operator[](const String& entity) const
  {
    return _findEntity(entity);
  }
  const String& operator[](unsigned int i) const
  {
    return m_family_names[i];
  }

 private:

  Integer _findEntity(const String& entity) const
  {
    for (int i = 0; i < m_family_names.size(); ++i) {
      if (m_family_names[i] == entity)
        return i;
    }
    return -1;
  }

  // For meshes, we calculate the memory contribution of nodes, edges, and faces.
  void _computeMemCell(Cell cell)
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

  //! Calculates the contribution of an entity to adjacent meshes.
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

  UniqueArray<String> m_family_names;
  UniqueArray<Int32> m_overall_memory;
  UniqueArray<Int32> m_resident_memory;

  //! Cache system for accessing memory related to a mesh.
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

class CriteriaMng
{
 public:

  CriteriaMng(bool use_legacy_init);

 public:

  void init(IMesh* mesh);

  void defaultMassCriterion(bool mass_criterion)
  {
    if (!m_is_edited_mass_criterion)
      m_use_mass_as_criterion = mass_criterion;
  }

  void setMassCriterion(bool mass_criterion)
  {
    m_is_edited_mass_criterion = true;
    m_use_mass_as_criterion = mass_criterion;
  }

  bool useMassAsCriterion() const { return m_use_mass_as_criterion; }
  bool useNbCellsAsCriterion() const { return m_nb_cells_as_criterion; }

  void resetCriteria();
  void clearVariables();

  void addCriterion(const StoreIProxyItemVariable& criterion)
  {
    m_event_vars.add(criterion);
  }

  Integer nbCriteria();
  ArrayView<StoreIProxyItemVariable> criteria();

  const VariableCellArrayReal& criteriaWeight() const
  {
    return *m_event_weights;
  }

  void addCommCost(const StoreIProxyItemVariable& comm_cost) { m_comm_vars.add(comm_cost); }
  const VariableCellReal& massResWeight() const { return *m_mass_res_weight; }
  const VariableCellReal& massWeight() const { return *m_mass_over_weight; }

  void fillCellNewOwner()
  {
    if (!m_cell_new_owner.isNull())
      m_cell_new_owner->fill(m_mesh->parallelMng()->commRank(), m_mesh->ownCells());
  }

  const VariableFaceReal& commCost() const { return *m_comm_costs; }

  void setComputeComm(bool active) { m_need_compute_comm = active; }
  bool cellCommContrib() const { return m_cell_comm; }
  void setCellCommContrib(bool active) { m_cell_comm = active; }
  void setNbCellsAsCriterion(bool active) { m_nb_cells_as_criterion = active; }
  void addMass(const StoreIProxyItemVariable& mass) { m_mass_vars.add(mass); }
  bool needComputeComm() const { return m_need_compute_comm; }
  bool isInit() const { return m_is_init; }
  Integer addEntity(const String& entity) { return m_criteria->addEntity(entity); }
  void computeCriteria();

 private:

  void _computeOverallMass();
  void _computeComm();
  void _computeResidentMass();
  void _computeEvents();

 private:

  UniqueArray<StoreIProxyItemVariable> m_mass_vars;
  UniqueArray<StoreIProxyItemVariable> m_comm_vars;
  UniqueArray<StoreIProxyItemVariable> m_event_vars;

  bool m_use_mass_as_criterion = false;
  bool m_nb_cells_as_criterion = true;
  bool m_cell_comm = false;
  bool m_need_compute_comm = false;
  bool m_is_edited_mass_criterion = false;
  bool m_is_init = false;

  ScopedPtrT<VariableFaceReal> m_comm_costs;
  ScopedPtrT<VariableCellReal> m_mass_over_weight;
  ScopedPtrT<VariableCellReal> m_mass_res_weight;
  ScopedPtrT<VariableCellArrayReal> m_event_weights;
  ScopedPtrT<VariableCellInt32> m_cell_new_owner; // SdC This variable is a problem when using a custom mesh

  IMesh* m_mesh = nullptr;
  ScopedPtrT<PartitionerMemoryInfo> m_criteria;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_IMPL_EXPORT LoadBalanceMngInternal
: public ILoadBalanceMngInternal
{
 public:

  explicit LoadBalanceMngInternal(bool mass_as_criterion, bool is_legacy_init);

 public:

  void addMass(VariableCellInt32& count, IMesh* mesh, const String& entity) override;
  void addCriterion(VariableCellInt32& count, IMesh* mesh) override;
  void addCriterion(VariableCellReal& count, IMesh* mesh) override;
  void addCommCost(VariableFaceInt32& count, IMesh* mesh, const String& entity) override;

  void setMassAsCriterion(IMesh* mesh, bool active) override;
  void setNbCellsAsCriterion(IMesh* mesh, bool active) override;
  void setCellCommContrib(IMesh* mesh, bool active) override;
  void setComputeComm(IMesh* mesh, bool active) override;
  const VariableFaceReal& commCost(IMesh* mesh) override;
  const VariableCellReal& massWeight(IMesh* mesh) override;
  const VariableCellReal& massResWeight(IMesh* mesh) override;
  const VariableCellArrayReal& mCriteriaWeight(IMesh* mesh) override;

  bool cellCommContrib(IMesh* mesh) override;
  Integer nbCriteria(IMesh* mesh) override;

  void reset(IMesh* mesh) override;
  void initAccess(IMesh* mesh) override;
  void endAccess() override;
  void notifyEndPartition() override;

 private:

  MeshHandle m_mesh_handle;
  bool m_default_mass_criterion = false;
  bool m_is_legacy_init = false;
  std::unordered_map<IMesh*, Ref<CriteriaMng>> m_mesh_criterion;

 private:

  CriteriaMng& _getCriteria(IMesh* mesh);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
