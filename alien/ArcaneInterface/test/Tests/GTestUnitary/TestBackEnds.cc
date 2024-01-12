#include <gtest/gtest.h>

#include <alien/Alien.h>

namespace Environment {
extern Arccore::MessagePassing::IMessagePassingMng* parallelMng();
}

TEST(TestBackEnds, Constructor)
{
  Alien::VectorDistribution dist(3, Environment::parallelMng());
  Alien::Space sp(3);
  Alien::MultiVectorImpl impl(std::make_shared<Alien::Space>(sp), dist.clone());
  ASSERT_EQ(nullptr, impl.block());
  ASSERT_EQ(sp, impl.space());
  ASSERT_EQ(dist, impl.distribution());
}

TEST(TestBackEnds, CompositeVector)
{
  Alien::CompositeKernel::MultiVectorImpl impl;
  ASSERT_EQ(nullptr, impl.block());
  ASSERT_EQ(Alien::Space(), (const Alien::ISpace&)impl.space());
  ASSERT_EQ(Alien::VectorDistribution(), impl.distribution());
}
