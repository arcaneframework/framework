// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneCheckpointModule.cc                                   (C) 2000-2020 */
/*                                                                           */
/* Module managing protections/restorations.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/core/ISubDomain.h"
#include "arcane/core/EntryPoint.h"
#include "arcane/core/Timer.h"
#include "arcane/core/ITimeHistoryMng.h"
#include "arcane/core/ModuleFactory.h"
#include "arcane/core/ServiceUtils.h"
#include "arcane/core/ICheckpointWriter.h"
#include "arcane/core/ICheckpointMng.h"
#include "arcane/core/Directory.h"
#include "arcane/core/IParallelMng.h"

#include "arcane/core/OutputChecker.h"
#include "arcane/std/ArcaneCheckpoint_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Module managing protections (checkpoint/restart mechanism).
*/
class ArcaneCheckpointModule
: public ArcaneArcaneCheckpointObject
{
 public:

  ArcaneCheckpointModule(const ModuleBuilder& cb);
  ~ArcaneCheckpointModule();

 public:

  virtual VersionInfo versionInfo() const { return VersionInfo(0, 9, 1); }

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
, m_output_checker(mb.subDomain(), "CheckpointRestart")
, m_checkpoint_timer(0)
, m_checkpoint_writer(0)
, m_checkpoint_dirname(".")
{
  m_checkpoint_timer = new Timer(mb.subDomain(), "Checkpoint", Timer::TimerReal);
  m_output_checker.assignIteration(&m_next_iteration, &options()->period);
  m_output_checker.assignGlobalTime(&m_next_global_time, &options()->frequency);
  m_output_checker.assignCPUTime(&m_next_cpu_time, &options()->frequencyCpu);
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
  // Unlike other output types, the CPU time for the next output must be reset
  // to zero because the CPU time used belongs to the current execution.
  m_next_cpu_time = options()->frequencyCpu();
  info() << " -------------------------------------------";
  info() << "|            PROTECTION-REPRISE             |";
  info() << " -------------------------------------------";
  info() << " ";
  //info() << " Uses the service '" << options()->checkpointServiceName()
  //     << "' for protections";
  info() << " ";
  m_output_checker.initialize();
  info() << " ";
  // If protections are active, check that the specified service
  // exists
  if (options()->doDumpAtEnd()) {
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
  Integer cpu_used = Convert::toInteger(m_global_cpu_time());

  bool do_output = m_output_checker.check(old_time, current_time, iteration, cpu_used);
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

  // Initializes the output checker. This must be done here if we
  // want to save things at iteration 1.
  _checkHasOutput();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief End-of-calculation operations
 *
 * - Performs an end-of-calculation checkpoint (if requested)
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
  // Displays execution statistics
  Real total_time = m_checkpoint_timer->totalTime();
  info() << "Total time spent in protection output (second): " << total_time;
  Integer nb_time = m_checkpoint_timer->nbActivated();
  if (nb_time != 0)
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

  Directory output_directory = Directory(export_dir, "protection");
  IParallelMng* pm = parallelMng();
  if (pm->isMasterIO())
    output_directory.createDirectory();
  pm->barrier();

  m_checkpoint_dirname = output_directory.path();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Performs a checkpoint.
 *
 * Displays the time spent during checkpoint writing.
 */
void ArcaneCheckpointModule::
_doCheckpoint(bool save_history)
{
  {
    Timer::Sentry sentry(m_checkpoint_timer);
    Timer::Phase tp(subDomain(), TP_InputOutput);
    if (!m_checkpoint_writer)
      _getCheckpointService();
    if (m_checkpoint_writer) {
      Integer nb_checkpoint = m_checkpoints_time.size();
      m_checkpoints_time.resize(nb_checkpoint + 1);
      Real checkpoint_time = m_global_time();
      m_checkpoints_time[nb_checkpoint] = checkpoint_time;
      m_checkpoint_writer->setCheckpointTimes(m_checkpoints_time);
      m_checkpoint_writer->setBaseDirectoryName(m_checkpoint_dirname);
      info() << "****  Protection active at time " << checkpoint_time
             << " directory=" << m_checkpoint_dirname
             << " number " << nb_checkpoint << " ******";
      subDomain()->checkpointMng()->writeDefaultCheckpoint(m_checkpoint_writer);
    }
  }
  info() << "Checkpoint write time (second): "
         << m_checkpoint_timer->lastActivationTime();

  // Saves histories
  if (save_history)
    subDomain()->timeHistoryMng()->dumpHistory(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCheckpointModule::
_getCheckpointService()
{
  ICheckpointWriter* checkpoint = options()->checkpointService();
  if (!checkpoint) {
    StringUniqueArray valid_values;
    options()->checkpointService.getAvailableNames(valid_values);
    pfatal() << "Protections required but protection/restore service selected ("
             << options()->checkpointService.serviceName() << ") not available "
             << "(valid values: " << String::join(", ", valid_values) << ")";
  }
  m_checkpoint_writer = checkpoint;
  _setDirectoryName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Checks if a checkpoint should be performed at this moment and
 * performs it if necessary.
 */
void ArcaneCheckpointModule::
checkpointCheckAndWriteData()
{
  bool do_output = _checkHasOutput();
  if (!do_output)
    return;

  info() << "Protection required.";
  // TEMPORARY:
  // Unlike end-of-execution checkpointing, this occurs at the end of the
  // time loop, but before the iteration number is incremented. If we resume
  // at this time, the iteration number will be incorrect. To correct this
  // problem, we increment this number before the checkpoint and reset it
  // to the correct value afterward.
  // SDC: this problem has not been observed. It seems to me that it is the
  // current iteration number that must be saved. Changed for a restart issue
  // in an IFPEN application (internally Bugzilla 778.
  //  m_global_iteration = m_global_iteration() + 1;
  _doCheckpoint(true);
  //  m_global_iteration = m_global_iteration() - 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
