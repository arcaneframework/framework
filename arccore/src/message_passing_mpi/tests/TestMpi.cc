// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h"
#include "arccore/base/Ref.h"
#include "arccore/base/BFloat16.h"
#include "arccore/base/Float16.h"
#include "arccore/collections/Array.h"
#include "arccore/message_passing/Messages.h"
#include "arccore/message_passing/Communicator.h"
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
  Communicator c(pm->communicator());
  ASSERT_TRUE(c.isValid());
  MPI_Comm comm = c;
  std::cout << "Communicator=" << comm << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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

namespace
{

template <typename DataType>
void _doTestFloat16(IMessagePassingMng* pm)
{
  Int32 nb_rank = pm->commSize();
  ASSERT_EQ(pm->commSize(), 2);
  Int32 my_rank = pm->commRank();
  UniqueArray<DataType> send_buf = { DataType{ -1.2f }, DataType{ 4.5f }, DataType{ 3.2e5f } };
  UniqueArray<DataType> receive_buf(send_buf.size());
  if (my_rank == 0) {
    mpSend(pm, send_buf, 1);
  }
  else {
    mpReceive(pm, receive_buf, 0);
    ASSERT_EQ(send_buf, receive_buf);
  }

  // Teste les réductions
  {
    UniqueArray<DataType> values(nb_rank);
    DataType expected_sum(0.0f);
    for (Int32 i = 0; i < nb_rank; ++i) {
      values[i] = (static_cast<float>(i) - 1.2f) * 3.4f;
      expected_sum = expected_sum + values[i];
    }
    DataType expected_min = values[0];
    DataType expected_max = values[nb_rank - 1];
    DataType my_value(values[my_rank]);
    DataType sum_result = mpAllReduce(pm, eReduceType::ReduceSum, my_value);
    std::cout << "Sum=" << sum_result << "\n";
    ASSERT_EQ(expected_sum, sum_result);

    DataType max_result = mpAllReduce(pm, eReduceType::ReduceMax, my_value);
    std::cout << "Max=" << max_result << "\n";
    ASSERT_EQ(expected_max, max_result);

    DataType min_result = mpAllReduce(pm, eReduceType::ReduceMin, my_value);
    std::cout << "Min=" << min_result << "\n";
    ASSERT_EQ(expected_min, min_result);
  }
}

} // namespace

TEST(MessagePassingMpi, Float16)
{
  // Teste uniquement l'appel. Les tests avec des vraies valeurs
  // sont faits dans Arcane.
  Ref<IMessagePassingMng> pm(StandaloneMpiMessagePassingMng::createRef(global_mpi_comm_world));
  _doTestFloat16<Float16>(pm.get());
  _doTestFloat16<BFloat16>(pm.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
