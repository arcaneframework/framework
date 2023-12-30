// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
    std::cout << "SETUP MPI\n";
    int mpi_error = MPI_Init(global_argc, global_argv);
    global_mpi_comm_world = MPI_COMM_WORLD;
    ASSERT_EQ(mpi_error,MPI_SUCCESS);
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
