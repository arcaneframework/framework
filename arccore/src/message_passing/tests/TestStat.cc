// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/message_passing/Stat.h"

using namespace Arccore;
using namespace Arccore::MessagePassing;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(MessagePassing, OneStat)
{
  String name1("Message1");
  const Int64 message_size1 = 324;
  const double message_time1 = 1.2;
  const Int64 message_size2 = 1923;
  const double message_time2 = 4.5;

  {
    OneStat stat1(name1);
    ASSERT_EQ(stat1.name(), name1);
  }
  {
    OneStat stat1(name1, message_size1, message_time1);
    ASSERT_EQ(stat1.name(), name1);
    ASSERT_EQ(stat1.nbMessage(), 0);
    ASSERT_EQ(stat1.totalSize(), message_size1);
    ASSERT_EQ(stat1.totalTime(), message_time1);
  }
  {
    OneStat stat1(name1);

    stat1.addMessage(message_size1, message_time1);
    ASSERT_EQ(stat1.nbMessage(), 1);
    ASSERT_EQ(stat1.totalSize(), message_size1);
    ASSERT_EQ(stat1.totalTime(), message_time1);
    ASSERT_EQ(stat1.cumulativeNbMessage(), 1);
    ASSERT_EQ(stat1.cumulativeTotalSize(), message_size1);
    ASSERT_EQ(stat1.cumulativeTotalTime(), message_time1);

    stat1.resetCurrentStat();

    ASSERT_EQ(stat1.nbMessage(), 0);
    ASSERT_EQ(stat1.totalSize(), 0);
    ASSERT_EQ(stat1.totalTime(), 0);
    ASSERT_EQ(stat1.cumulativeNbMessage(), 1);
    ASSERT_EQ(stat1.cumulativeTotalSize(), message_size1);
    ASSERT_EQ(stat1.cumulativeTotalTime(), message_time1);

    stat1.addMessage(message_size2, message_time2);

    ASSERT_EQ(stat1.nbMessage(), 1);
    ASSERT_EQ(stat1.totalSize(), message_size2);
    ASSERT_EQ(stat1.totalTime(), message_time2);
    ASSERT_EQ(stat1.cumulativeNbMessage(), 2);
    ASSERT_EQ(stat1.cumulativeTotalSize(), message_size1 + message_size2);
    ASSERT_EQ(stat1.cumulativeTotalTime(), message_time1 + message_time2);

    stat1.print(std::cout);
  }

  {
    OneStat stat1(name1);

    stat1.setNbMessage(3);
    ASSERT_EQ(stat1.nbMessage(), 3);

    stat1.setTotalSize(message_size1);
    ASSERT_EQ(stat1.totalSize(), message_size1);

    stat1.setTotalTime(message_time1);
    ASSERT_EQ(stat1.totalTime(), message_time1);

    stat1.setCumulativeNbMessage(25);
    ASSERT_EQ(stat1.cumulativeNbMessage(), 25);

    stat1.setCumulativeTotalSize(message_size2);
    ASSERT_EQ(stat1.cumulativeTotalSize(), message_size2);

    stat1.setCumulativeTotalTime(message_time2);
    ASSERT_EQ(stat1.cumulativeTotalTime(), message_time2);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
