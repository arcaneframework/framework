// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LoadBalanceMngInternal.h                                    (C) 2000-2024 */
/*                                                                           */
/* Classe interne gérant un TODO.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNAL_LOADBALANCEMNGINTERNAL_H
#define ARCANE_IMPL_INTERNAL_LOADBALANCEMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/



#include "arcane/core/IMesh.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/internal/ILoadBalanceMngInternal.h"
#include "arcane/utils/ObjectImpl.h"
#include "arcane/utils/AutoRef.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/utils/ScopedPtr.h"
#include <unordered_map>
#include "arcane/core/IVariableMng.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/CommonVariables.h"
#include "arcane/core/VariableCollection.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class PartitionerMemoryInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface proxy pour acceder aux variables definissant les poids.
 *
 * Est indépendante du type de variable (Integer, Real).
 * Est à libération automatique de mémoire (via ObjectImpl).
 * Permet de noter à quelle famille d'objets est associée la variable.
 */
class ARCANE_IMPL_EXPORT IProxyItemVariable
: public ObjectImpl
{
 public:
  virtual ~IProxyItemVariable() { }

  //! Accès à la valeur associée à une entité du maillage, sous forme d'un Real.
  virtual Real operator[](ItemEnumerator i) const =0;

  //! Accès au numéro de la famille associée.
  virtual Integer getPos() const =0;
};


/*!
 *  @brief Classe pour accéder au proxy sans déférencement dans le code.
 *
 *  Est indepedante du type de variable (Integer, Real).
 *  Est à libération automatique de mémoire (via AutoRefT).
 */
class ARCANE_IMPL_EXPORT StoreIProxyItemVariable
{
 public:
  StoreIProxyItemVariable(IVariable* var=NULL, Integer pos=0) {
    m_var = StoreIProxyItemVariable::proxyItemVariableFactory(var,pos);
  }

  StoreIProxyItemVariable(const StoreIProxyItemVariable& src) {
    if (m_var != src.m_var)
      m_var = src.m_var;
  }

  ~StoreIProxyItemVariable() {
  }

  //! Accès à la valeur associée à une entité du maillage, sous forme d'un Real.
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
  //! Factory pour la constructions selon le type de variable initiale.
  static IProxyItemVariable* proxyItemVariableFactory(IVariable* var, Integer pos=0);

 private:
  //! Pointeur vers la variable.
  AutoRefT<IProxyItemVariable> m_var;
};

struct MeshCriteria
{
  MeshCriteria()
  : m_mass_criterion(false)
  , m_nb_criterion(false)
  , m_cell_comm(true)
  , m_compute_comm(true)
  , is_edited_mass_criterion(false)
  , is_init(false)
  {
    m_mass_vars.clear();
    m_comm_vars.clear();
    m_event_vars.resize(1); // First slot booked by MemoryOverAll

    m_event_weights = nullptr;
    m_mass_res_weight = nullptr;
    m_mass_over_weigth = nullptr;
    m_comm_costs = nullptr;
  };

  explicit MeshCriteria(IMesh* mesh)
  : MeshCriteria()
  {
    init(mesh);
  }

  void init(IMesh* mesh)
  {
    MeshHandle mesh_handle = mesh->handle();
    int vflags = IVariable::PExecutionDepend | IVariable::PNoDump | IVariable::PTemporary;

    m_cell_new_owner = std::make_unique<VariableCellInt32>(VariableBuildInfo(mesh_handle, "CellFamilyNewOwner", IVariable::PExecutionDepend | IVariable::PNoDump));
    m_comm_costs = new VariableFaceReal(VariableBuildInfo(mesh_handle, "LbMngCommCost", vflags));
    m_mass_over_weigth = new VariableCellReal(VariableBuildInfo(mesh_handle, "LbMngOverallMass", vflags));
    m_mass_res_weight = new VariableCellReal(VariableBuildInfo(mesh_handle, "LbMngResidentMass", vflags));
    m_event_weights = new VariableCellArrayReal(VariableBuildInfo(mesh_handle, "LbMngMCriteriaWgt", vflags));

    m_comm_costs->fill(1);
    m_mass_over_weigth->fill(1);
    m_mass_res_weight->fill(1);
    is_init = true;
  }

  void defaultMassCriterion(bool mass_criterion)
  {
    if(!is_edited_mass_criterion) m_mass_criterion = mass_criterion;
  }

  void setMassCriterion(bool mass_criterion)
  {
    is_edited_mass_criterion = true;
    m_mass_criterion = mass_criterion;
  }

  bool massCriterion()
  {
    return m_mass_criterion;
  }

  void reset(){
    m_mass_vars.clear();
    m_comm_vars.clear();
    m_event_vars.resize(1); // First slot booked by MemoryOverAll

    endAccess();
  }

  void endAccess(){
    m_event_weights = nullptr;
    m_mass_res_weight = nullptr;
    m_mass_over_weigth = nullptr;
    m_comm_costs = nullptr;
    is_init = false;
  }

  UniqueArray<StoreIProxyItemVariable> m_mass_vars;
  UniqueArray<StoreIProxyItemVariable> m_event_vars;
  UniqueArray<StoreIProxyItemVariable> m_comm_vars;

 private:
  bool m_mass_criterion;
 public:
  bool m_nb_criterion;
  bool m_cell_comm;
  bool m_compute_comm;
  bool is_edited_mass_criterion;
  bool is_init;

  ScopedPtrT<VariableFaceReal> m_comm_costs;
  ScopedPtrT<VariableCellReal> m_mass_over_weigth;
  ScopedPtrT<VariableCellReal> m_mass_res_weight;
  ScopedPtrT<VariableCellArrayReal> m_event_weights;

  std::unique_ptr<VariableCellInt32> m_cell_new_owner; // SdC This variable is a problem when using a custom mesh
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class LoadBalanceMngInternal
: public ILoadBalanceMngInternal
{
 public:

  explicit LoadBalanceMngInternal(ISubDomain* sd, bool massAsCriterion);

 public:
  /*!
   * Methodes utilisees par les modules clients pour definir les criteres
   * de partitionnement.
   */
  void addMass(VariableCellInt32& count, IMesh* mesh, const String& entity) override;
  void addCriterion(VariableCellInt32& count, IMesh* mesh) override;
  void addCriterion(VariableCellReal& count, IMesh* mesh) override;
  void addCommCost(VariableFaceInt32& count, IMesh* mesh, const String& entity) override;

  void reset(IMesh* mesh) override;

  /*!
   * Methodes utilisees par le MeshPartitioner pour acceder a la description
   * du probleme.
   */
  void setMassAsCriterion(IMesh* mesh, bool active) override;
  void setNbCellsAsCriterion(IMesh* mesh, bool active) override;
  void setCellCommContrib(IMesh* mesh, bool active) override;
  bool cellCommContrib(IMesh* mesh) const override;
  void setComputeComm(IMesh* mesh, bool active) override;
  Integer nbCriteria(IMesh* mesh) override;
  void initAccess(IMesh* mesh) override;
  const VariableFaceReal& commCost(IMesh* mesh) const override;
  const VariableCellReal& massWeight(IMesh* mesh) const override;
  const VariableCellReal& massResWeight(IMesh* mesh) const override;
  const VariableCellArrayReal& mCriteriaWeight(IMesh* mesh) const override;
  void endAccess() override;
  void notifyEndPartition() override;

 private:

  void _computeOverallMass();
  void _computeResidentMass();
  void _computeComm();
  void _computeEvents();

  MeshHandle m_mesh_handle;
  ScopedPtrT<PartitionerMemoryInfo> m_criteria;

  bool m_default_mass_criterion;

  std::unordered_map<IMesh*, MeshCriteria> m_mesh_criterion;


};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
