// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneLoadBalanceModule.cc                                  (C) 2000-2010 */
/*                                                                           */
/* Module d'équilibrage de charge.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/EntryPoint.h"
#include "arcane/ISubDomain.h"
#include "arcane/IParallelMng.h"
#include "arcane/ModuleFactory.h"
#include "arcane/IMeshPartitioner.h"
#include "arcane/ServiceUtils.h"
#include "arcane/CommonVariables.h"
#include "arcane/ITimeStats.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/ITimeHistoryMng.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshModifier.h"
#include "arcane/ItemPrinter.h"
#include "arcane/IItemFamily.h"

#include "arcane/std/ArcaneLoadBalance_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

#define OLD_LOADBALANCE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module d'équilibrage de charge
 */
class ArcaneLoadBalanceModule
: public ArcaneArcaneLoadBalanceObject
{
 public:

  ArcaneLoadBalanceModule(const ModuleBuildInfo& mb);
  ~ArcaneLoadBalanceModule();

 public:

  virtual VersionInfo versionInfo() const { return VersionInfo(1,0,0); }

 public:

  void checkLoadBalance();
  void loadBalanceInit();

 private:

  VariableScalarReal m_elapsed_computation_time;
  /*! \brief Temps de calcul depuis le dernier équilibrage
   * Note: cette valeur doit être synchronisée.
   */
  Real m_computation_time;
#ifdef OLD_LOADBALANCE
   Integer m_nb_weight;
  UniqueArray<float> m_cells_weight;
#endif // OLD_LOADBALANCE

 private:

  void _checkInit();
  Real _computeImbalance();
#ifdef OLD_LOADBALANCE
   void _computeWeights(RealConstArrayView compute_times,Real max_compute_time);
#endif // OLD_LOADBALANCE
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_ARCANELOADBALANCE(ArcaneLoadBalanceModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneLoadBalanceModule::
ArcaneLoadBalanceModule(const ModuleBuildInfo& mb)
: ArcaneArcaneLoadBalanceObject(mb)
, m_elapsed_computation_time(VariableBuildInfo(this,"ArcaneLoadBalanceElapsedComputationTime",
                                               IVariable::PNoDump|IVariable::PNoRestore))
, m_computation_time(0.0)
#ifdef OLD_LOADBALANCE
 , m_nb_weight(2)
#endif // OLD_LOADBALANCE
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneLoadBalanceModule::
~ArcaneLoadBalanceModule()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneLoadBalanceModule::
loadBalanceInit()
{
  m_elapsed_computation_time = 0;
  _checkInit();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneLoadBalanceModule::
_checkInit()
{
  if (options()->period()==0 || !options()->active()){
    info() << "Load balance deactivated.";
    return;
  }

  if (!subDomain()->parallelMng()->isParallel()){
    info() << "Load balance required but inactive during serial execution";
    return;
  }

  info() << "Load balance active with maximum unbalance: "
         << options()->maxImbalance();
  // Indique au maillage qu'il peut évoluer
  defaultMesh()->modifier()->setDynamic(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneLoadBalanceModule::
checkLoadBalance()
{
  ISubDomain* sd = subDomain();
  Integer global_iteration = sd->commonVariables().globalIteration();
  int period = options()->period();
  if (period==0)
    return;
  if (global_iteration==0)
    return;
  if ((global_iteration % period) != 0)
    return;
  
  Real imbalance = _computeImbalance();

  if (!options()->active())
    return;
  if (imbalance<options()->maxImbalance())
    return;
  Real min_cpu_time = options()->minCpuTime();
  if (min_cpu_time!=0 || m_computation_time<min_cpu_time)
    return;

  m_computation_time = 0;
  info() << "Programme un repartitionnement du maillage";
#ifdef OLD_LOADBALANCE
  IMeshPartitioner* p = options()->partitioner();
  _computeWeights(p->computationTimes(),p->maximumComputationTime());
  p->setCellsWeight(m_cells_weight,m_nb_weight);
#endif // OLD_LOADBALANCE
  subDomain()->timeLoopMng()->registerActionMeshPartition(options()->partitioner());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifdef OLD_LOADBALANCE
/*!
 * \brief Calcule le poids de chaque maille et le range dans m_cells_weight
 */
void ArcaneLoadBalanceModule::
_computeWeights(RealConstArrayView compute_times,Real max_compute_time)
{
  ISubDomain* sd = subDomain();
  IMesh* mesh = this->mesh();
  IParallelMng* pm = sd->parallelMng();
  Int32 nb_sub_domain = pm->commSize();
  CellGroup own_cells = mesh->ownCells();
  IntegerUniqueArray global_nb_own_cell(nb_sub_domain);
  Integer nb_own_cell = own_cells.size();
  Integer nb_weight = m_nb_weight;

  pm->allGather(IntegerConstArrayView(1,&nb_own_cell),global_nb_own_cell);
  //Integer total_nb_cell = 0;

  // compute_times[0] contient le temps global (sans tracking)
  // compute_times[1] contient le temps de tracking
  //Real max_compute_time = maximumComputationTime();
  //RealConstArrayView compute_times = computationTimes();
  bool has_compute_time = compute_times.size()!=0;
  bool has_cell_time = compute_times.size()==2;
  if (math::isZero(max_compute_time))
    max_compute_time = 1.0;
  Real compute_times0 = 1.0;
  Real compute_times1 = 0.0;
  if (has_compute_time){
    compute_times0 = compute_times[0];
    if (has_cell_time)
      compute_times1 = compute_times[1];
  }

  bool dump_info = true;

  Real time_ratio = compute_times0 / max_compute_time;
  Real time_ratio2 = compute_times1 / max_compute_time;

  if (dump_info){
    info() << " MAX_COMPUTE=" << max_compute_time;
    info() << " COMPUTE 0=" << compute_times0;
    info() << " COMPUTE 1=" << compute_times1;
    info() << " TIME RATIO 0=" << time_ratio;
    info() << " TIME RATIO 2=" << time_ratio2;
  }
  Real proportional_time = compute_times0 / (nb_own_cell+1);

  if (dump_info){
    info() << " PROPORTIONAL TIME=" << proportional_time;
  }

  Real max_weight = 0.0;
  IItemFamily* cell_family = mesh->cellFamily();
  m_cells_weight.resize(cell_family->maxLocalId()*nb_weight);
  ENUMERATE_CELL(iitem,own_cells){
    const Cell& cell = *iitem;
    Real v0 = proportional_time;
    Real w = (v0);
    if (dump_info && iitem.index()<10){
      info() << "Weight " << ItemPrinter(cell)
             << " v0=" << v0
             << " w=" << w;
    }
    // Pour test si on a un deuxieme poids, le multiplie par i+1;
    for( Integer i=0; i<nb_weight; ++i ){
      m_cells_weight[(nb_weight*iitem->localId())+i] = (float)(w*((Real)(i+1)));
    }
    if (w>max_weight)
      max_weight = w;
  }

  Real total_max_weight = pm->reduce(Parallel::ReduceMax,max_weight);
  if (math::isZero(total_max_weight))
    total_max_weight = 1.0;

  if (dump_info){
    info() << " TOTAL MAX WEIGHT=" << total_max_weight;
  }

  ENUMERATE_CELL(iitem,own_cells){
    for( Integer i=0; i<nb_weight; ++i ){
      Integer idx = (nb_weight*iitem->localId())+i;
      m_cells_weight[idx] = (float)(m_cells_weight[idx] / total_max_weight);
    }
  }
}
#endif // OLD_LOADBALANCE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real ArcaneLoadBalanceModule::
_computeImbalance()
{
  //TODO: rendre la méthode compatible avec le retour-arrière
  ITimeStats* time_stats = subDomain()->timeStats();
  IParallelMng* pm = subDomain()->parallelMng();

  // Temps écoulé depuis le début de l'exécution
  Real elapsed_computation_time = time_stats->elapsedTime(TP_Computation);
  Real computation_time = elapsed_computation_time - m_elapsed_computation_time();

  m_elapsed_computation_time = elapsed_computation_time;

  if (options()->statistics()){
    // Optionnel:
    // Récupère le temps de calcul de chaque sous-domaine pour en sortir
    // les historiques.
    // TODO: dans ce cas, le reduce standard pour le min et le max est
    // inutilise -> le supprimer
    Integer nb_sub_domain = pm->commSize();
    RealUniqueArray compute_times(nb_sub_domain);
    Real my_time = computation_time;
    RealConstArrayView my_time_a(1,&my_time);
    pm->allGather(my_time_a,compute_times);
    ITimeHistoryMng* thm = subDomain()->timeHistoryMng();
    thm->addValue("SubDomainComputeTime",compute_times);
  }

  Real reduce_times[2];
  reduce_times[0] = computation_time;
  reduce_times[1] = -computation_time;
  // Tous les replicats doivent avoir les mêmes infos de temps pour tous lancer
  // le repartitionnement (même si au final un seul réplica fait le repartionnement,
  // l'opération de demande doit être collective)
  subDomain()->allReplicaParallelMng()->reduce(Parallel::ReduceMin,RealArrayView(2,reduce_times));
  Real min_computation_time = reduce_times[0];
  Real max_computation_time = -reduce_times[1];
  if (math::isZero(max_computation_time))
    max_computation_time = 1.;
  if (math::isZero(min_computation_time))
    min_computation_time = 1.;
  RealUniqueArray computation_times(1);
  computation_times[0] = computation_time;

  m_computation_time += max_computation_time;
  
  Real ratio = computation_time / max_computation_time;
  Real imbalance = (max_computation_time - min_computation_time) / min_computation_time;
  info() << "Computing time used (" << pm->commRank() << ") :"
         << " nb_owncell=" << ownCells().size()
         << " current=" << computation_time
         << " min=" << min_computation_time
         << " max=" << max_computation_time
         << " ratio=" << ratio
         << " imbalance=" << imbalance
         << " full-compute=" << m_computation_time;

  IMeshPartitioner* p = options()->partitioner();
  p->setMaximumComputationTime(max_computation_time);
  p->setComputationTimes(computation_times);
  p->setImbalance(imbalance);
  p->setMaxImbalance(options()->maxImbalance());
  return imbalance;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
