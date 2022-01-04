#include <gtest/gtest.h>

#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/data/CompositeVector.h>
#include <alien/distribution/VectorDistribution.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>

#include <alien/data/Space.h>
#include <alien/functional/Cast.h>
#include <alien/ref/handlers/scalar/VectorReader.h>
#include <alien/ref/handlers/scalar/VectorWriter.h>

#include <arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h>

namespace Environment {
extern Arccore::MessagePassing::IMessagePassingMng* parallelMng();
}

TEST(TestCompositeVector, DefaultConstructor)
{
  Alien::CompositeVector v;
  std::cout << v.size();
  ASSERT_EQ(0, v.size());
  ASSERT_TRUE(v.hasUserFeature("composite"));
  ASSERT_EQ(0, v.space().size());
}

TEST(TestCompositeVector, ConstructorWithSize)
{
  Alien::CompositeVector v(3);
  ASSERT_TRUE(v.hasUserFeature("composite"));
  ASSERT_EQ(3, v.size());
  ASSERT_EQ(0, v.space().size());
  for (int i = 0; i < 3; ++i) {
    auto& c = v[i];
    // ASSERT_THROW(c.impl(), Alien::FatalErrorException);
    ASSERT_EQ(0, c.space().size());
  }
}

TEST(TestCompositeVector, CompositeConstructorsTest)
{
  Alien::CompositeVector v(2);
  Alien::CompositeElement(v, 0) = Alien::Vector(4, Environment::parallelMng());
  Alien::CompositeElement(v, 1) = Alien::Vector(5, Environment::parallelMng());
  ASSERT_EQ(9, v.space().size());
  auto& c0 = v[0];
  ASSERT_EQ(4, c0.space().size());
  auto& c1 = v[1];
  ASSERT_EQ(5, c1.space().size());
}

TEST(TestCompositeVector, CompositeResize)
{
  Alien::CompositeVector v;
  ASSERT_EQ(0, v.size());
  ASSERT_TRUE(v.hasUserFeature("composite"));
  ASSERT_EQ(0, v.space().size());
  v.resize(2);
  Alien::CompositeElement(v, 0) = Alien::Vector(4, Environment::parallelMng());
  Alien::CompositeElement(v, 1) = Alien::Vector(5, Environment::parallelMng());
  ASSERT_EQ(9, v.space().size());
  auto& c0 = v[0];
  ASSERT_EQ(4, c0.space().size());
  auto& c1 = v[1];
  ASSERT_EQ(5, c1.space().size());
}

TEST(TestCompositeVector, CompositeMultipleResize)
{
  Alien::CompositeVector v;
  ASSERT_EQ(0, v.size());
  ASSERT_TRUE(v.hasUserFeature("composite"));
  ASSERT_EQ(0, v.space().size());
  v.resize(2);
  Alien::CompositeElement(v, 0) = Alien::Vector(4, Environment::parallelMng());
  Alien::CompositeElement(v, 1) = Alien::Vector(5, Environment::parallelMng());
  ASSERT_EQ(9, v.space().size());
  {
    auto& c0 = v[0];
    ASSERT_EQ(4, c0.space().size());
    auto& c1 = v[1];
    ASSERT_EQ(5, c1.space().size());
  }
  v.resize(3);
  ASSERT_EQ(3, v.size());
  for (int i = 0; i < 3; ++i) {
    auto& c = v[i];
    // ASSERT_THROW(c.impl(), Alien::FatalErrorException);
    ASSERT_EQ(0, c.space().size());
  }
  Alien::CompositeElement(v, 0) = Alien::Vector(4, Environment::parallelMng());
  Alien::CompositeElement(v, 1) = Alien::Vector(3, Environment::parallelMng());
  ASSERT_EQ(7, v.space().size());
  {
    auto& c0 = v[0];
    ASSERT_EQ(4, c0.space().size());
    auto& c1 = v[1];
    ASSERT_EQ(3, c1.space().size());
    auto& c2 = v[2];
    // ASSERT_THROW(c2.impl(), Alien::FatalErrorException);
    ASSERT_EQ(0, c2.space().size());
  }
}

auto fill = [](Alien::IVector& c, Arccore::Real shift) {
  auto& dist = c.impl()->distribution();
  Arccore::Integer lsize = dist.localSize();
  Arccore::Integer offset = dist.offset();
  auto& cc = Alien::cast<Alien::Vector>(c);
  Alien::VectorWriter writer(cc);
  for (Arccore::Integer i = 0; i < lsize; ++i)
    writer[i] = offset + i + shift;
};

auto check = [](Alien::IVector& c, Arccore::Real shift) {
  auto& dist = c.impl()->distribution();
  Arccore::Integer lsize = dist.localSize();
  Arccore::Integer offset = dist.offset();
  auto& v = Alien::cast<Alien::Vector>(c);
  Alien::LocalVectorReader reader(v);
  for (Arccore::Integer i = 0; i < lsize; ++i)
    ASSERT_DOUBLE_EQ((offset + i + shift), reader[i]);
};

// ATTENTION
// Le test suivant utilise le convertisseur composite -> csr
// Comme on ne donne pas de distribution à l'objet composite
// cela ne fonctionne qu'en sequentiel !!

TEST(TestCompositeVector, CompositeReaderWriterTest)
{
  if (Environment::parallelMng()->commSize() > 1)
    return;
  Alien::CompositeVector v(2);
  Alien::CompositeElement(v, 0) = Alien::Vector(4, Environment::parallelMng());
  Alien::CompositeElement(v, 1) = Alien::Vector(5, Environment::parallelMng());
  ASSERT_EQ(9, v.space().size());
  std::cout << "auto& c0 = v[0];" << std::endl;
  auto& c0 = v[0];
  std::cout << "fill(c0, 1.);" << std::endl;
  ;
  fill(c0, 1.);
  std::cout << "auto& c1 = v[1];" << std::endl;
  ;
  auto& c1 = v[1];
  std::cout << "fill(c1, 2.);" << std::endl;
  ;
  fill(c1, 2.);
  std::cout << "check(c0, 1.);" << std::endl;
  ;
  check(c0, 1.);
  std::cout << "fill(c1, 3.);" << std::endl;
  ;
  fill(c1, 3.);
  std::cout << "fill(c0, 2.);" << std::endl;
  ;
  fill(c0, 2.);
  std::cout << "check(c0, 2.);" << std::endl;
  ;
  check(c0, 2.);
  std::cout << "check(c1, 3.);" << std::endl;
  ;
  check(c1, 3.);
}

TEST(TestCompositeVector, TimeStampTest)
{
  Alien::CompositeVector v;
  auto* impl = v.impl();
  ASSERT_EQ(0, impl->timestamp());
  std::cout << "main ts = " << impl->timestamp() << std::endl;
  v.resize(2);
  Alien::CompositeElement(v, 0) = Alien::Vector(4, Environment::parallelMng());
  Alien::CompositeElement(v, 1) = Alien::Vector(4, Environment::parallelMng());
  ASSERT_EQ(0, impl->timestamp());
  std::cout << "main ts = " << impl->timestamp() << std::endl;
  auto& c0 = v[0];
  auto* impl0 = c0.impl();
  ASSERT_EQ(0, impl0->timestamp());
  std::cout << "sub0 ts = " << impl0->timestamp() << std::endl;
  impl0->get<Alien::BackEnd::tag::simplecsr>(true);
  impl0->get<Alien::BackEnd::tag::simplecsr>(true);
  ASSERT_EQ(2, impl0->timestamp());
  std::cout << "sub0 ts = " << impl0->timestamp() << std::endl;
  ASSERT_EQ(2, impl->timestamp());
  std::cout << "main ts = " << impl->timestamp() << std::endl;
}
