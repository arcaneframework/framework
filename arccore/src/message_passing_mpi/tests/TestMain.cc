// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
#include "arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
{
Ref<Arcane::MessagePassing::IMessagePassingMng> global_message_passing_mng;
}

using namespace Arccore::MessagePassing::Mpi;

namespace
{
char*** global_argv = nullptr;
int* global_argc = nullptr;

class MPIEnvironment
: public ::testing::Environment
{
 public:

  void SetUp() override
  {
    int thread_required = MPI_THREAD_SERIALIZED;
    global_message_passing_mng = StandaloneMpiMessagePassingMng::createAndInitializeMPI(global_argc,global_argv, thread_required);
  }
  void TearDown() override
  {
    global_message_passing_mng.reset();
  }
};

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int main(int argc, char* argv[])
{
  global_argc = &argc;
  global_argv = &argv;
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment());
  return RUN_ALL_TESTS();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
