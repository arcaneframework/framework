#include <gtest/gtest.h>

#include <alien/ref/AlienRefSemantic.h>

namespace Environment {
extern Arccore::MessagePassing::IMessagePassingMng* parallelMng();
}

TEST(TestVectorBuilder, ReleaseTest)
{
  Alien::Vector v(3, Environment::parallelMng());
  {
    Alien::VectorWriter writer(v);
  }
  ASSERT_EQ(3, v.space().size());
}

TEST(TestVectorBuilder, WriterTest)
{
  Alien::Vector v(3, Environment::parallelMng());
  {
    Alien::VectorWriter writer(v);
    writer[0] = 0.;
    writer[1] = 1.;
    writer[2] = 2.;
  }
  ASSERT_EQ(3, v.space().size());
}

TEST(TestVectorBuilder, ReaderWriterTest)
{
  Alien::Vector v(3, Environment::parallelMng());
  {
    Alien::VectorWriter writer(v);
    writer[0] = 0.;
    writer[1] = 1.;
    writer[2] = 2.;
  }
  ASSERT_EQ(3, v.space().size());
  {
    Alien::LocalVectorReader reader(v);
    ASSERT_EQ(0., reader[0]);
    ASSERT_EQ(1., reader[1]);
    ASSERT_EQ(2., reader[2]);
    ASSERT_EQ(3, reader.size());
  }
}
