/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <alien/utils/time_stamp/Timestamp.h>
#include <alien/utils/time_stamp/TimestampMng.h>
#include <alien/utils/time_stamp/TimestampObserver.h>

TEST(TestTimestamp, UpdateTimestamp)
{
  Alien::TimestampMng mng;
  Alien::Timestamp ts1(&mng), ts2(&mng);
  ts1.updateTimestamp();
  ASSERT_EQ(1, ts1.timestamp());
  ASSERT_EQ(0, ts2.timestamp());
  ASSERT_EQ(1, mng.timestamp());
  ts2.updateTimestamp();
  ASSERT_EQ(1, ts1.timestamp());
  ASSERT_EQ(2, ts2.timestamp());
  ASSERT_EQ(2, mng.timestamp());
}

TEST(TestTimestamp, UpdateTimestampWithObserver)
{
  Alien::TimestampMng mng1;
  Alien::Timestamp ts1(&mng1), ts11(&mng1);
  Alien::TimestampMng mng2;
  Alien::Timestamp ts2(&mng2);
  mng2.addObserver(std::make_shared<Alien::TimestampObserver>(ts1));
  ts1.updateTimestamp();
  ASSERT_EQ(1, ts1.timestamp());
  ASSERT_EQ(0, ts11.timestamp());
  ASSERT_EQ(1, mng1.timestamp());
  ASSERT_EQ(0, ts2.timestamp());
  ASSERT_EQ(0, mng2.timestamp());
  ts2.updateTimestamp();
  ASSERT_EQ(2, ts1.timestamp());
  ASSERT_EQ(0, ts11.timestamp());
  ASSERT_EQ(2, mng1.timestamp());
  ASSERT_EQ(1, ts2.timestamp());
  ASSERT_EQ(1, mng2.timestamp());
  ts11.updateTimestamp();
  ASSERT_EQ(2, ts1.timestamp());
  ASSERT_EQ(3, ts11.timestamp());
  ASSERT_EQ(3, mng1.timestamp());
  ASSERT_EQ(1, ts2.timestamp());
  ASSERT_EQ(1, mng2.timestamp());
  ts2.updateTimestamp();
  ASSERT_EQ(4, ts1.timestamp());
  ASSERT_EQ(3, ts11.timestamp());
  ASSERT_EQ(4, mng1.timestamp());
  ASSERT_EQ(2, ts2.timestamp());
  ASSERT_EQ(2, mng2.timestamp());
}
