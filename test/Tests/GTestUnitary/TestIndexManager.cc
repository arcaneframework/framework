#include <ALIEN/IndexManager/Functional/DefaultAbstractFamily.h>

#include "gtest/gtest.h"

#include <ALIEN/IndexManager/IndexManager.h>

namespace Environment {
extern Alien::IParallelMng* parallelMng();
extern Alien::ITraceMng* traceMng();
}

TEST(IndexManager, Constructor)
{
  Alien::IndexManager m(Environment::parallelMng());
  ASSERT_FALSE(m.isPrepared());
}

TEST(IndexManager, EmptyPrepare)
{
  Alien::IndexManager m(Environment::parallelMng());
  ASSERT_FALSE(m.isPrepared());
  m.prepare(); // sort);
  ASSERT_TRUE(m.isPrepared());
}

TEST(IndexManager, SimpleAbstractFamily)
{
  Alien::IndexManager m(Environment::parallelMng());
  m.setVerboseMode(true);
  ASSERT_FALSE(m.isPrepared());
  Alien::Int64UniqueArray uids(3);
  uids[0] = 122;
  uids[1] = 137;
  uids[2] = 33;
  Alien::IntegerUniqueArray owners(3, 0);
  Alien::DefaultAbstractFamily family(uids, owners, Environment::parallelMng());
  auto entry = m.buildScalarIndexSet("Simple", family, 0);
  m.prepare();
  ASSERT_TRUE(m.isPrepared());
  auto indexes = m.getIndexes(entry);
  ASSERT_EQ(3, indexes.size());
  for (auto& i : indexes)
    std::cout << indexes[i] << std::endl;
}

TEST(IndexManager, Assertion)
{
  Alien::Int64UniqueArray uids(1, 0);
  Alien::DefaultAbstractFamily s(uids, Environment::parallelMng());
  {
    Alien::IndexManager m(Environment::parallelMng());
    m.setVerboseMode(true);
    ASSERT_FALSE(m.isPrepared());
    m.prepare(); // sort);
    ASSERT_TRUE(m.isPrepared());
    // Appel prepare() déjà fait
    ASSERT_THROW(m.buildScalarIndexSet("Simple", s, 0), Alien::FatalErrorException);
  }
  {
    Alien::IndexManager m(Environment::parallelMng());
    m.setVerboseMode(true);
    ASSERT_FALSE(m.isPrepared());
    // 2 familles avec même label
    m.buildScalarIndexSet("Simple", s, 0);
    ASSERT_THROW(m.buildScalarIndexSet("Simple", s, 0), Alien::FatalErrorException);
  }
  {
    Alien::IndexManager m(Environment::parallelMng());
    m.setVerboseMode(true);
    ASSERT_FALSE(m.isPrepared());
    // Appel prepare() non effectué avant getIndexes()
    auto entry = m.buildScalarIndexSet("Simple", s, 0);
    ASSERT_THROW(m.getIndexes(entry), Alien::FatalErrorException);
  }
}

TEST(IndexManager, MultipleSimpleAbstractFamily0)
{
  Alien::IndexManager m(Environment::parallelMng());
  m.setVerboseMode(true);
  ASSERT_FALSE(m.isPrepared());
  Alien::Int64UniqueArray uids1(3);
  uids1[0] = 122;
  uids1[1] = 137;
  uids1[2] = 33;
  Alien::IntegerUniqueArray owners1(3, 0);
  Alien::DefaultAbstractFamily family1(uids1, owners1, Environment::parallelMng());
  auto entry1 = m.buildScalarIndexSet("Simple1", family1, 0);
  Alien::Int64UniqueArray uids2(2);
  uids2[0] = 112;
  uids2[1] = 33;
  Alien::IntegerUniqueArray owners2(2, 0);
  Alien::DefaultAbstractFamily family2(uids2, owners2, Environment::parallelMng());
  auto entry2 = m.buildScalarIndexSet("Simple2", family2, 0);
  m.prepare(); // sort);
  ASSERT_TRUE(m.isPrepared());
  auto indexes1 = m.getIndexes(entry1);
  ASSERT_EQ(3, indexes1.size());
  for (auto& i : indexes1)
    std::cout << i << std::endl;
  auto indexes2 = m.getIndexes(entry2);
  ASSERT_EQ(2, indexes2.size());
  for (auto& i : indexes2)
    std::cout << i << std::endl;
  for (auto e : m) {
    std::cout << "Scalar index set name = " << e.getName() << std::endl;
  }
}

TEST(IndexManager, MultipleSimpleAbstractFamily1)
{
  Alien::IndexManager m(Environment::parallelMng());
  m.setVerboseMode(true);
  ASSERT_FALSE(m.isPrepared());
  Alien::Int64UniqueArray uids1(3);
  uids1[0] = 122;
  uids1[1] = 137;
  uids1[2] = 33;
  Alien::IntegerUniqueArray owners1(3, 0);
  Alien::DefaultAbstractFamily family1(uids1, owners1, Environment::parallelMng());
  auto entry1 = m.buildScalarIndexSet("Simple1", family1, 0);
  Alien::Int64UniqueArray uids2(2);
  uids2[0] = 112;
  uids2[1] = 33;
  Alien::IntegerUniqueArray owners2(2, 0);
  Alien::DefaultAbstractFamily family2(uids2, owners2, Environment::parallelMng());
  auto entry2 = m.buildScalarIndexSet("Simple2", family2, 1);
  m.prepare();
  ASSERT_TRUE(m.isPrepared());
  auto indexes1 = m.getIndexes(entry1);
  ASSERT_EQ(3, indexes1.size());
  for (auto& i : indexes1)
    std::cout << i << std::endl;
  auto indexes2 = m.getIndexes(entry2);
  ASSERT_EQ(2, indexes2.size());
  for (auto& i : indexes2)
    std::cout << i << std::endl;
  for (auto e : m) {
    std::cout << "Scalar index set name = " << e.getName() << std::endl;
  }
}

TEST(IndexManager, MultipleSimpleAbstractFamily2)
{
  Alien::IndexManager m(Environment::parallelMng());
  m.setVerboseMode(true);
  ASSERT_FALSE(m.isPrepared());
  Alien::Int64UniqueArray uids1(3);
  uids1[0] = 122;
  uids1[1] = 137;
  uids1[2] = 33;
  Alien::IntegerUniqueArray owners1(3, 0);
  Alien::DefaultAbstractFamily family1(uids1, owners1, Environment::parallelMng());
  auto entry1 = m.buildScalarIndexSet("Simple1", family1, 1);
  Alien::Int64UniqueArray uids2(2);
  uids2[0] = 112;
  uids2[1] = 33;
  Alien::IntegerUniqueArray owners2(2, 0);
  Alien::DefaultAbstractFamily family2(uids2, owners2, Environment::parallelMng());
  auto entry2 = m.buildScalarIndexSet("Simple2", family2, 0);
  m.prepare();
  ASSERT_TRUE(m.isPrepared());
  auto indexes1 = m.getIndexes(entry1);
  ASSERT_EQ(3, indexes1.size());
  for (auto& i : indexes1)
    std::cout << i << std::endl;
  auto indexes2 = m.getIndexes(entry2);
  ASSERT_EQ(2, indexes2.size());
  for (auto& i : indexes2)
    std::cout << i << std::endl;
  for (auto e : m) {
    std::cout << "Scalar index set name = " << e.getName() << std::endl;
  }
}
