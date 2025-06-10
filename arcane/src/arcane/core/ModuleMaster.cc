// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ModuleMaster.cc                                             (C) 2000-2025 */
/*                                                                           */
/* Module maître.                                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Iterator.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/Collection.h"
#include "arcane/utils/IOnlineDebuggerService.h"

#include "arcane/core/ModuleMaster.h"
#include "arcane/core/EntryPoint.h"
#include "arcane/core/CaseOptionsMain.h"
#include "arcane/core/ITimeHistoryMng.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/IModuleMng.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ICaseMng.h"
#include "arcane/core/IModuleMng.h"
#include "arcane/core/ModuleBuildInfo.h"
#include "arcane/core/ITimeLoopService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT IModuleMaster*
arcaneCreateModuleMaster(ISubDomain* sd)
{
  ModuleBuildInfo mbi(sd,sd->defaultMeshHandle(),"ArcaneMasterInternal");
  ModuleMaster* m = new ModuleMaster(mbi);
  return m;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleMaster::
ModuleMaster(const ModuleBuildInfo& mb)
: AbstractModule(mb)
, CommonVariables(this)
{
  m_case_options_main = new CaseOptionsMain(mb.subDomain()->caseMng());

  addEntryPoint(this,"ArcaneTimeLoopBegin",
                &ModuleMaster::timeLoopBegin,
                IEntryPoint::WComputeLoop,IEntryPoint::PAutoLoadBegin);
  addEntryPoint(this,"ArcaneTimeLoopEnd",
                &ModuleMaster::timeLoopEnd,
                IEntryPoint::WComputeLoop,IEntryPoint::PAutoLoadEnd);
  addEntryPoint(this,"ArcaneMasterStartInit",
                &ModuleMaster::masterStartInit,
                IEntryPoint::WStartInit,IEntryPoint::PAutoLoadBegin);
  addEntryPoint(this,"ArcaneMasterInit",
                &ModuleMaster::masterInit,
                IEntryPoint::WInit,IEntryPoint::PAutoLoadBegin);
  addEntryPoint(this,"ArcaneMasterContinueInit",
                &ModuleMaster::masterContinueInit,
                IEntryPoint::WContinueInit,
                IEntryPoint::PAutoLoadBegin);
  addEntryPoint(this,"ArcaneMasterLoopExit",
                &ModuleMaster::_masterLoopExit,
                IEntryPoint::WExit,
                IEntryPoint::PAutoLoadBegin);
  addEntryPoint(this,"ArcaneMasterLoopMeshChanged",
                &ModuleMaster::_masterMeshChanged,
                IEntryPoint::WOnMeshChanged,
                IEntryPoint::PAutoLoadBegin);
  addEntryPoint(this,"ArcaneMasterLoopRestore",
                &ModuleMaster::_masterRestore,
                IEntryPoint::WRestore,
                IEntryPoint::PAutoLoadBegin);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleMaster::
~ModuleMaster()
{
  delete m_case_options_main;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleMaster::
masterStartInit()
{
  m_global_iteration = 0;
  m_global_time = 0.0;
  m_global_old_time = m_global_time();
  m_global_old_cpu_time = 0.0;
  m_global_cpu_time = 0.0;
  m_global_old_elapsed_time = 0.0;
  m_global_elapsed_time = 0.0;
  m_global_old_deltat = 0.0;

  // Met a jour les options du jeu de données qui dépendent d'une table de marche.
  ICaseMng* com = subDomain()->caseMng();
  com->updateOptions(0.0,0.0,0);

  _masterStartInit();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleMaster::
masterInit()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleMaster::
masterContinueInit()
{
  // En reprise, l'initialisation est considérée comme ayant lieu avant
  // la prochaine itération, donc avec les anciennes valeurs du pas de temps
  // et du temps de la simulation.

  ICaseMng* com = subDomain()->caseMng();
  Int32 opt_iteration = m_global_iteration() - 1;
  info() << "Initialization to restore the functions of the input data: "
         << " time=" << m_global_old_time()
         << " dt=" << m_global_old_deltat()
         << " iteration=" << opt_iteration;
  com->updateOptions(m_global_old_time(),m_global_old_deltat(),opt_iteration);

  _masterContinueInit();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleMaster::
timeIncrementation()
{
  m_global_time = m_global_time() + m_global_deltat();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleMaster::
timeStepInformation()
{
  Integer precision = FloatInfo<Real>::maxDigit();
  Integer digit = FloatInfo<Real>::maxDigit()+5;

  info()<<" ";
  info(0) << "***"
          << " ITERATION " << Trace::Width(8) << m_global_iteration()
          << "  TIME " << Trace::Width(digit) << Trace::Precision(precision,m_global_time(),true)
          << "  LOOP " << Trace::Width(8) << m_nb_loop
          << "  DELTAT " << Trace::Width(digit) << Trace::Precision(precision,m_global_deltat(),true)
          << " ***";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


void ModuleMaster::
timeLoopBegin()
{
  // Première boucle, initialise le temps et affiche les infos
  // sur la consommation
  if (m_is_first_loop){
    info() << "Information on consumption (unit:second): Con(R=...,I=...,C=...)";
    info() << " R -> consumption in real time (clock) since the beginning of the computation";
    info() << " I -> real time (clock) spent during the last iteration";
    info() << " C -> consumption in CPU time since the beginning of the computation";
    info() << "Information on memory consumption (unit: Mo): Mem=(X,m=X1:R1,M=X2:R2,avg=Z)";
    info() << " X -> memory consumption of the process";
    info() << " X1 -> memory consumption of the least memory hungry process and R1 is its rank";
    info() << " X2 -> memory consumption of the most memory hungry process and R2 is its rank";
    info() << " Z -> average memory consumption of all process";
  }

  ++m_nb_loop;

  // Met a jour les options du jeu de données qui dépendent d'une table de marche.
  ICaseMng* com = subDomain()->caseMng();
  com->updateOptions(m_global_time(),m_global_deltat(),m_global_iteration());

  Real mem_used = platform::getMemoryUsed();

  IParallelMng* pm = parallelMng();

  m_global_old_time = m_global_time();

  // Infos pour les courbes
  m_thm_mem_used = mem_used;
  
  // Ajoute le deltat au temps courant
  //info() << "[ModuleMaster::timeLoopBegin] Add deltat to the current time";
  timeIncrementation();

  m_thm_global_time = m_global_time();
  //thm->addValue(m_global_time.name(),m_global_time());

  //info() << "[ModuleMaster::timeLoopBegin] breakpoint_requested?";
  IOnlineDebuggerService* hyoda = platform::getOnlineDebuggerService();
 
  // Il faut absolument que cette valeur soit la même pour tous les sous-domaines
  Real cpu_time = (Real)platform::getCPUTime();
  Real elapsed_time = platform::getRealTime();
  {
    Real values[3];
    ArrayView<Real> vals(3,values);
    values[0] = cpu_time;
    values[1] = elapsed_time;
    values[2] = (hyoda)?hyoda->loopbreak(subDomain()):0.0;
    subDomain()->allReplicaParallelMng()->reduce(Parallel::ReduceMax,vals);
    cpu_time = values[0];
    elapsed_time = values[1];
    if (hyoda && values[2]>0.0)
      hyoda->hook(subDomain(),values[2]);
  }
  if (m_is_first_loop){
    m_old_cpu_time = cpu_time;
    m_old_elapsed_time = elapsed_time;
  }
  Real diff_cpu =  (cpu_time - m_old_cpu_time) / CLOCKS_PER_SEC;
  Real diff_elapsed =  (elapsed_time - m_old_elapsed_time);
  m_global_old_cpu_time = diff_cpu;
  m_global_old_elapsed_time = diff_elapsed;
  m_global_cpu_time = m_global_cpu_time() + diff_cpu;
  m_global_elapsed_time = m_global_elapsed_time() + diff_elapsed;

  m_thm_diff_cpu = diff_cpu;
  m_thm_global_cpu_time = m_global_cpu_time();
  m_thm_diff_elapsed = diff_elapsed;
  m_thm_global_elapsed_time = m_global_elapsed_time();

  m_old_cpu_time = cpu_time;
  m_old_elapsed_time = elapsed_time;
  
  timeStepInformation();
  
  Real mem_sum = 0.0;
  Real mem_min = 0.0;
  Real mem_max = 0.0;
  Int32 mem_min_rank = 0;
  Int32 mem_max_rank = 0;
  pm->computeMinMaxSum(mem_used,mem_min,mem_max,mem_sum,mem_min_rank,mem_max_rank);
  // Tronque à la milli-seconde
  Int64 i_elapsed = (Int64)(m_global_elapsed_time() * 1000.0);
  Int64 i_diff_elapsed = (Int64)(diff_elapsed * 1000.0);
  Int64 i_cpu = (Int64)(m_global_cpu_time() * 1000.0);
  info(1) << "Date: " << platform::getCurrentDateTime()
          << " Conso=(R=" << (((Real)i_elapsed) / 1000.0)
          << ",I=" << (((Real)i_diff_elapsed) / 1000.0)
          << ",C=" << ((Real)i_cpu / 1000.0)
          << ") Mem=(" << (Int64)(mem_used/1e6)
          << ",m=" << (Int64)(mem_min/1e6)
          << ":" << mem_min_rank
          << ",M=" << (Int64)(mem_max/1e6)
          << ":" << mem_max_rank
          << ",avg=" << (Int64)(mem_sum/1e6) / pm->commSize()
          << ")";

  _masterBeginLoop();
  m_is_first_loop = false;
  m_has_thm_dump_at_iteration = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleMaster::
timeLoopEnd()
{
  _masterEndLoop();

  // Effectue les sorties courbes
  // Comme il est possible de désactiver les sorties courbes
  // à une itération donnée, il faut faire les sorties en fin d'itération
  // et non pas au début pour permettre aux autres modules d'activer ou non
  // les sorties. Il est aussi possible pour un code utilisateur d'appeler
  // explicitement cette méthode pour qu'il fasse les sorties quand il le
  // souhaite au cours de l'itération.
  dumpStandardCurves();
 
  // Incrémente le compteur d'itération
  m_global_iteration = m_global_iteration() + 1;

  if (subDomain()->timeLoopMng()->finalTimeReached()){
    info() <<"===============================================================";
    info() <<"======   END OF COMPUTATION REACHED...  =======================";
    info() <<"===============================================================";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleMaster::
dumpStandardCurves()
{
  if (m_has_thm_dump_at_iteration)
    return;

  // Si cette méthode est appelée à la fin de l'itération, alors les
  // temps cpu (m_thm_diff_cpu) et elapsed (m_thm_global_elapsed_time)
  // ne sont pas bon car ils ont été calculés en début d'itération et donc
  // ils ne sont pas bon. On les recalcule donc en prenant soin
  // de ne pas mettre à jour les variables m_global* pour éviter toute
  // incohérence car ces valeurs ne sont pas synchronisées entre tous
  // les sous-domaines (et de plus cette méthode n'est pas toujours
  // appelée)

  Real cpu_time = (Real)platform::getCPUTime();
  Real elapsed_time = platform::getRealTime();
  Real diff_cpu =  (cpu_time - m_old_cpu_time) / 1000000.0;
  Real diff_elapsed =  (elapsed_time - m_old_elapsed_time);
  Real mem_used = platform::getMemoryUsed();
  Real global_cpu_time = diff_cpu + m_global_cpu_time();
  Real global_elapsed_time = diff_elapsed + m_global_elapsed_time();

  ITimeHistoryMng* thm = subDomain()->timeHistoryMng();
  thm->addValue("TotalMemory",mem_used);
  thm->addValue("CpuTime",diff_cpu);
  thm->addValue("GlobalCpuTime",global_cpu_time);
  thm->addValue("ElapsedTime",diff_elapsed);
  thm->addValue("GlobalElapsedTime",global_elapsed_time);
  thm->addValue(m_global_time.name(),m_thm_global_time);
  m_has_thm_dump_at_iteration = true;

  Int64 i_elapsed = (Int64)(global_elapsed_time * 1000.0);
  Int64 i_diff_elapsed = (Int64)(diff_elapsed * 1000.0);
  Int64 i_cpu = (Int64)(global_cpu_time * 1000.0);

  info(4) << "EndIter: Date: " << platform::getCurrentDateTime()
          << " Conso=(R=" << (((Real)i_elapsed) / 1000.0)
          << ",I=" << (((Real)i_diff_elapsed) / 1000.0)
          << ",C=" << ((Real)i_cpu / 1000.0)
          << ") Mem=(" << (Int64)(mem_used/1e6)
          << ")";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleMaster::
addTimeLoopService(ITimeLoopService* tls)
{
  m_timeloop_services.add(tls);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleMaster::
_masterBeginLoop()
{
  for( Integer i=0, is=m_timeloop_services.size(); i<is; ++i ){
    ITimeLoopService* service = m_timeloop_services[i];
    service->onTimeLoopBeginLoop();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleMaster::
_masterEndLoop()
{
  for( Integer i=0, is=m_timeloop_services.size(); i<is; ++i ){
    ITimeLoopService* service = m_timeloop_services[i];
    service->onTimeLoopEndLoop();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleMaster::
_masterStartInit()
{
  for( Integer i=0, is=m_timeloop_services.size(); i<is; ++i ){
    ITimeLoopService* service = m_timeloop_services[i];
    service->onTimeLoopStartInit();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleMaster::
_masterContinueInit()
{
  for( Integer i=0, is=m_timeloop_services.size(); i<is; ++i ){
    ITimeLoopService* service = m_timeloop_services[i];
    service->onTimeLoopContinueInit();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleMaster::
_masterLoopExit()
{
  for( Integer i=0, is=m_timeloop_services.size(); i<is; ++i ){
    ITimeLoopService* service = m_timeloop_services[i];
    service->onTimeLoopExit();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleMaster::
_masterMeshChanged()
{
  for( Integer i=0, is=m_timeloop_services.size(); i<is; ++i ){
    ITimeLoopService* service = m_timeloop_services[i];
    service->onTimeLoopMeshChanged();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleMaster::
_masterRestore()
{
  for( Integer i=0, is=m_timeloop_services.size(); i<is; ++i ){
    ITimeLoopService* service = m_timeloop_services[i];
    service->onTimeLoopRestore();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
