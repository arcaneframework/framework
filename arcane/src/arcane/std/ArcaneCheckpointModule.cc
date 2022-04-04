// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneCheckpointModule.cc                                   (C) 2000-2020 */
/*                                                                           */
/* Module gérant les protections/reprises.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/ISubDomain.h"
#include "arcane/EntryPoint.h"
#include "arcane/Timer.h"
#include "arcane/ITimeHistoryMng.h"
#include "arcane/ModuleFactory.h"
#include "arcane/ServiceUtils.h"
#include "arcane/ICheckpointWriter.h"
#include "arcane/ICheckpointMng.h"
#include "arcane/Directory.h"
#include "arcane/IParallelMng.h"

#include "arcane/OutputChecker.h"
#include "arcane/std/ArcaneCheckpoint_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module gérant les protections (mécanisme de protections/reprises).
*/
class ArcaneCheckpointModule
: public ArcaneArcaneCheckpointObject
{
 public:

  ArcaneCheckpointModule(const ModuleBuilder& cb);
  ~ArcaneCheckpointModule();

 public:

  virtual VersionInfo versionInfo() const { return VersionInfo(0,9,1); }

 public:

  virtual void checkpointCheckAndWriteData();
  virtual void checkpointStartInit();
  virtual void checkpointInit();
  virtual void checkpointExit();


 private:

  OutputChecker m_output_checker;
  Timer* m_checkpoint_timer;
  ICheckpointWriter* m_checkpoint_writer;
  String m_checkpoint_dirname;

 private:
  
  void _doCheckpoint(bool save_history);
  void _dumpStats();
  void _getCheckpointService();
  void _setDirectoryName();
  bool _checkHasOutput();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_ARCANECHECKPOINT(ArcaneCheckpointModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneCheckpointModule::
ArcaneCheckpointModule(const ModuleBuildInfo& mb)
: ArcaneArcaneCheckpointObject(mb)
, m_output_checker(mb.m_sub_domain,"CheckpointRestart")
, m_checkpoint_timer(0)
, m_checkpoint_writer(0)
, m_checkpoint_dirname(".")
{
  m_checkpoint_timer = new Timer(mb.m_sub_domain,"Checkpoint",Timer::TimerReal);
  m_output_checker.assignIteration(&m_next_iteration,&options()->period);
  m_output_checker.assignGlobalTime(&m_next_global_time,&options()->frequency);
  m_output_checker.assignCPUTime(&m_next_cpu_time,&options()->frequencyCpu);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneCheckpointModule::
~ArcaneCheckpointModule()
{
  delete m_checkpoint_timer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCheckpointModule::
checkpointInit()
{
  // Contrairement aux autres types de sortie, il faut remettre le temps
  // CPU de la prochaine sortie à zéro car le temps CPU utilisé est de
  // l'exécution courante.
  m_next_cpu_time = options()->frequencyCpu();
  info() << " -------------------------------------------";
  info() << "|            PROTECTION-REPRISE             |";
  info() << " -------------------------------------------";
  info() << " ";
  //info() << " Utilise le service '" << options()->checkpointServiceName()
  //     << "' pour les protections";
  info() << " ";
  m_output_checker.initialize();
  info() << " ";
  // Si les protections sont actives, vérifie que le service spécifié
  // existe bien
  if (options()->doDumpAtEnd()){
    info() << "Protection required at the end of computations";
    _getCheckpointService();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ArcaneCheckpointModule::
_checkHasOutput()
{
  Real old_time = m_global_old_time();
  Real current_time = m_global_time();
  Integer iteration = m_global_iteration();
  Integer cpu_used  = Convert::toInteger(m_global_cpu_time());

  bool do_output = m_output_checker.check(old_time,current_time,iteration,cpu_used);
  return do_output;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCheckpointModule::
checkpointStartInit()
{
  m_next_global_time = 0.;
  m_next_iteration = 0;
  m_next_cpu_time = 0;

  // Initialise le vérificateur de sortie. Il faut le faire ici si on
  // veut sauver des choses à l'itération 1.
  _checkHasOutput();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Opérations de fin de calcul
 *
 * - Effectue une protection de fin de calcul (si demandée)
 */
void ArcaneCheckpointModule::
checkpointExit()
{
  if (!options()->doDumpAtEnd())
    return;

  _doCheckpoint(false);
  _dumpStats();

  if (m_checkpoint_writer)
    m_checkpoint_writer->close();
  m_checkpoint_writer = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCheckpointModule::
_dumpStats()
{
  // Affiche statistiques d'exécutions
  Real total_time = m_checkpoint_timer->totalTime();
  info() << "Total time spent in protection output (second): " << total_time;
  Integer nb_time = m_checkpoint_timer->nbActivated();
  if (nb_time!=0)
    info() << "Average time per output (second): " << total_time / nb_time
           << " (for " << nb_time << " outputs";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCheckpointModule::
_setDirectoryName()
{
  Directory export_dir = subDomain()->storageDirectory();
  if (export_dir.path().null())
    export_dir = subDomain()->exportDirectory();

  Directory output_directory = Directory(export_dir,"protection");
  IParallelMng* pm = parallelMng();
  if (pm->isMasterIO())
    output_directory.createDirectory();
  pm->barrier();

  m_checkpoint_dirname = output_directory.path();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Effectue une protection.
 *
 * Affiche le temps passé lors de l'écriture de la protection.
 */
void ArcaneCheckpointModule::
_doCheckpoint(bool save_history)
{
  {
    Timer::Sentry sentry(m_checkpoint_timer);
    Timer::Phase tp(subDomain(),TP_InputOutput);
    if (!m_checkpoint_writer)
      _getCheckpointService();
    if (m_checkpoint_writer){
      Integer nb_checkpoint = m_checkpoints_time.size();
      m_checkpoints_time.resize(nb_checkpoint+1);
      Real checkpoint_time = m_global_time();
      m_checkpoints_time[nb_checkpoint] = checkpoint_time;
      m_checkpoint_writer->setCheckpointTimes(m_checkpoints_time);
      m_checkpoint_writer->setBaseDirectoryName(m_checkpoint_dirname);
      info() << "****  Protection active at time " << checkpoint_time
             << " directory=" << m_checkpoint_dirname
             <<" numéro " << nb_checkpoint << " ******";
      subDomain()->checkpointMng()->writeDefaultCheckpoint(m_checkpoint_writer);
    }
  }
  info() << "Protection write time (second): "
         << m_checkpoint_timer->lastActivationTime();

  // Sauve les historiques
  if (save_history)
    subDomain()->timeHistoryMng()->dumpHistory(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCheckpointModule::
_getCheckpointService()
{
  ICheckpointWriter* checkpoint = options()->checkpointService();
  if (!checkpoint){
    StringUniqueArray valid_values;
    options()->checkpointService.getAvailableNames(valid_values);
    pfatal() << "Protections required but protection/restore service selected ("
             << options()->checkpointService.serviceName() << ") not available "
             << "(valid values: " << String::join(", ",valid_values) << ")";
  }
  m_checkpoint_writer = checkpoint;
  _setDirectoryName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie s'il faut faire une protection à cet instant et l'effectue
 * si nécessaire.
 */
void ArcaneCheckpointModule::
checkpointCheckAndWriteData()
{
  bool do_output = _checkHasOutput();
  if (!do_output)
    return;

  info() << "Protection required.";
  // TEMPORAIRE:
  // Contrairement a la protection en fin d'execution, celle-ci a lieu
  // dans à la fin de la boucle en temps, mais avant que le numéro
  // d'itération ne soit incrémenté. Si on reprend à ce temps,
  // le numéro d'itération ne sera pas bon. Pour corriger ce problème,
  // on incrémente ce nombre avant la protection et on le remet
  // à sa bonne valeur après.
  // SDC : ce probleme n'est pas constaté. Il me semble que c'est bien le numérod
  // de l'itération courante qu'il faut sauver. Changé pour un problème de restart 
  // pour une application IFPEN (en interne Bugzilla 778.
  //  m_global_iteration = m_global_iteration() + 1;
  _doCheckpoint(true);
  //  m_global_iteration = m_global_iteration() - 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
