#include <gtest/gtest.h>

#include <alien/Alien.h>

namespace Environment {
extern Arccore::MessagePassing::IMessagePassingMng* parallelMng();
}

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
