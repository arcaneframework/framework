// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
{
MPI_Comm global_mpi_comm_world = MPI_COMM_NULL;
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
    int thread_provided = 0;
    int mpi_error = ::MPI_Init_thread(global_argc, global_argv, thread_required, &thread_provided);
    ASSERT_EQ(mpi_error, MPI_SUCCESS);

    global_mpi_comm_world = MPI_COMM_WORLD;
    int comm_rank = 0;
    ::MPI_Comm_rank(global_mpi_comm_world, &comm_rank);
    if (comm_rank == 0)
      std::cout << "SETUP MPI result thread_required=" << thread_required << " provided=" << thread_provided << "\n";
  }
  void TearDown() override
  {
    int mpi_error = MPI_Finalize();
    ASSERT_EQ(mpi_error,MPI_SUCCESS);
  }
};

}

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
