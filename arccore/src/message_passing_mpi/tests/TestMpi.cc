// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h"
#include "arccore/base/Ref.h"
#include "arccore/collections/Array.h"
#include "arccore/message_passing/Messages.h"

#include "TestMain.h"

#include <iostream>

using namespace Arccore;
using namespace Arccore::MessagePassing;
using namespace Arccore::MessagePassing::Mpi;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(MessagePassingMpi, Simple)
{
  Ref<IMessagePassingMng> pm(StandaloneMpiMessagePassingMng::createRef(global_mpi_comm_world));
  std::cout << "Rank=" << pm->commRank() << "\n";
  if (pm->commSize() == 2) {
    UniqueArray<Int32> send_buf = { 1, 7, -4 };
    if (pm->commRank() == 0)
      mpSend(pm.get(), send_buf, 1);
    else {
      UniqueArray<Int32> receive_buf(3);
      mpReceive(pm.get(), receive_buf, 0);
      ASSERT_EQ(receive_buf.size(), 3);
      ASSERT_EQ(receive_buf[0], 1);
      ASSERT_EQ(receive_buf[1], 7);
      ASSERT_EQ(receive_buf[2], -4);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
