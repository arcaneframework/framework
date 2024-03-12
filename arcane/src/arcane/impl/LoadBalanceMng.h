// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LoadBalanceMng.h                                            (C) 2000-2024 */
/*                                                                           */
/* Module standard de description du probleme pour l'equilibrage de charge.  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_LOADBALANCEMNG_H
#define ARCANE_IMPL_LOADBALANCEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <memory>

#include "arcane/ILoadBalanceMng.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/AutoRef.h"
#include "arcane/utils/ObjectImpl.h"
#include "arcane/MeshHandle.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CriteriaMng;

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


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implantation standard d'une interface d'enregistrement des variables
 * pour l'equilibrage de charge.
 *
 */
class ARCANE_IMPL_EXPORT LoadBalanceMng
: public ILoadBalanceMng
{
 public:

  explicit LoadBalanceMng(ISubDomain* sd, bool massAsCriterion = true);

 public:
  /*!
   * Methodes utilisees par les modules clients pour definir les criteres
   * de partitionnement.
   */
  void addMass(VariableCellInt32& count, const String& entity="") override;
  void addCriterion(VariableCellInt32& count) override;
  void addCriterion(VariableCellReal& count) override;
  void addCommCost(VariableFaceInt32& count, const String& entity="") override;

  void reset() override;

  /*!
   * Methodes utilisees par le MeshPartitioner pour acceder a la description
   * du probleme.
   */
  void setMassAsCriterion(bool active=true) override { m_mass_criterion = active; }
  void setNbCellsAsCriterion(bool active=true) override { m_nb_criterion = active; }
  void setCellCommContrib(bool active=true) override { m_cell_comm = active; }
  bool cellCommContrib() const override { return m_cell_comm; }
  void setComputeComm(bool active=true) override { m_compute_comm = active; }
  Integer nbCriteria() override;
  void initAccess(IMesh* mesh=nullptr) override;
  const VariableFaceReal& commCost() const override { return *m_comm_costs; }
  const VariableCellReal& massWeight() const override {return *m_mass_over_weigth;}
  const VariableCellReal& massResWeight() const override {return *m_mass_res_weight;}
  const VariableCellArrayReal& mCriteriaWeight() const override {return *m_event_weights; }
  void endAccess() override;
  void notifyEndPartition() override;

 private:

  void _computeOverallMass();
  void _computeResidentMass();
  void _computeComm();
  void _computeEvents();

  MeshHandle m_mesh_handle;
  ScopedPtrT<CriteriaMng> m_criteria;
  UniqueArray<StoreIProxyItemVariable> m_mass_vars;
  UniqueArray<StoreIProxyItemVariable> m_event_vars;
  UniqueArray<StoreIProxyItemVariable> m_comm_vars;
  bool m_mass_criterion = false;
  bool m_nb_criterion = false;
  bool m_cell_comm = true;
  bool m_compute_comm = true;

  ScopedPtrT<VariableFaceReal> m_comm_costs;
  ScopedPtrT<VariableCellReal> m_mass_over_weigth;
  ScopedPtrT<VariableCellReal> m_mass_res_weight;
  ScopedPtrT<VariableCellArrayReal> m_event_weights;

  std::unique_ptr<VariableCellInt32> m_cell_new_owner; // SdC This variable is a problem when using a custom mesh
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
