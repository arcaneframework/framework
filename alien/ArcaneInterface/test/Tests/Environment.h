// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef TESTS_ENVIRONMENT_H
#define TESTS_ENVIRONMENT_H

#include <alien/AlienLegacyConfig.h>
#include <alien/data/Universe.h>

#include <arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h>
#include <arccore/trace/ITraceMng.h>
#include <arccore/trace/TraceClassConfig.h>

namespace Environment {

struct Private
{
  Arccore::ITraceMng* tm;
  Arccore::MessagePassing::IMessagePassingMng* pm;
} global_alien_env_info;

extern void
initialize(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  // Gestionnaire de parallélisme
  global_alien_env_info.pm = Arccore::MessagePassing::Mpi::StandaloneMpiMessagePassingMng::create(
      MPI_COMM_WORLD);

  // Gestionnaire de trace
  global_alien_env_info.tm = Arccore::arccoreCreateDefaultTraceMng();

  // Initialize the instance of TraceMng.
  // Only the rank 0 will display the listing
  bool is_master_io = (global_alien_env_info.pm->commRank()==0);
  Arccore::TraceClassConfig trace_config;
  trace_config.setActivated(is_master_io);

  global_alien_env_info.tm->setClassConfig("*",trace_config);
  global_alien_env_info.tm->setMaster(is_master_io);
  global_alien_env_info.tm->finishInitialize();
}

extern void
finalize()
{
  MPI_Finalize();
}

extern Arccore::MessagePassing::IMessagePassingMng*
parallelMng()
{
  Arccore::MessagePassing::IMessagePassingMng* pm = global_alien_env_info.pm;
  return pm;
}

extern Arccore::ITraceMng*
traceMng()
{
  Arccore::ITraceMng* tm = global_alien_env_info.tm;
  return tm;
}
}

namespace Environment {

template <typename T>
extern int
execute(int argc, char** argv, T&& t)
{
  initialize(argc, argv);

  int ret = t();

  finalize();

  return ret;
}
}

#endif
