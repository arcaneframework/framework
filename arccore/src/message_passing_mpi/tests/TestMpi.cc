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
#include "arccore/serialize/ISerializer.h"

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

TEST(MessagePassingMpi, SerializeGather)
{
  // Teste uniquement l'appel. Les tests avec des vraies valeurs
  // sont faits dans Arcane.
  Ref<IMessagePassingMng> pm(StandaloneMpiMessagePassingMng::createRef(global_mpi_comm_world));
  ASSERT_EQ(pm->commSize(), 3);
  Int32 my_rank = pm->commRank();
  Ref<ISerializer> send_serializer(createSerializer());
  UniqueArray<UniqueArray<Int32>> send_buf(3);

  send_buf[0] = UniqueArray<Int32>{ 1, 12, 4, -3 };
  send_buf[1] = UniqueArray<Int32>{ 27, 0 };
  send_buf[2] = UniqueArray<Int32>{ -5, 32, 8, -1, 8932 };

  send_serializer->setMode(ISerializer::ModeReserve);
  send_serializer->reserveArray(send_buf[my_rank]);
  send_serializer->allocateBuffer();
  send_serializer->setMode(ISerializer::ModePut);
  send_serializer->putArray(send_buf[my_rank]);

  Ref<ISerializer> receive_serializer(createSerializer());
  mpAllGather(pm.get(), send_serializer.get(), receive_serializer.get());
  receive_serializer->setMode(ISerializer::ModeGet);
  UniqueArray<UniqueArray<Int32>> receive_buf(3);
  receive_serializer->getArray(receive_buf[0]);
  receive_serializer->getArray(receive_buf[1]);
  receive_serializer->getArray(receive_buf[2]);

  //UniqueArray<Int32> expected_receive_buf = { 1, 12, 4, -3, 27, 0, -5, 32, 8, -1, 8932 };

  ASSERT_EQ(send_buf[0], receive_buf[0]);
  ASSERT_EQ(send_buf[1], receive_buf[1]);
  ASSERT_EQ(send_buf[2], receive_buf[2]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
