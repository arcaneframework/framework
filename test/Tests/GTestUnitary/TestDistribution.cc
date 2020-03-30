#include <gtest/gtest.h>

#include <alien/Alien.h>

using namespace Arccore;

namespace Environment {
extern Arccore::MessagePassing::IMessagePassingMng* parallelMng();
}

TEST(TestDistribution, VectorDefaultConstructor)
{
  Alien::VectorDistribution vd;
  ASSERT_FALSE(vd.isParallel());
}

TEST(TestDistribution, VectorGlobalSizeConstructor)
{
  Alien::VectorDistribution vd(10, Environment::parallelMng());
  ASSERT_EQ(vd.isParallel(), Environment::parallelMng()->commSize() > 1);
  ASSERT_EQ(vd.parallelMng(), Environment::parallelMng());
  ASSERT_EQ(10, vd.globalSize());
  auto gsize = Arccore::MessagePassing::mpAllReduce(
      Environment::parallelMng(), Arccore::MessagePassing::ReduceSum, vd.localSize());
  ASSERT_EQ(10, gsize);
}

TEST(TestDistribution, VectorGlobalLocalSizeConstructor)
{
  auto* pm = Environment::parallelMng();
  auto np = pm->commSize();
  auto rk = pm->commRank();
  auto global_size = 2 * np;
  Alien::VectorDistribution vd(global_size, 2, pm);
  ASSERT_EQ(vd.isParallel(), Environment::parallelMng()->commSize() > 1);
  ASSERT_EQ(vd.parallelMng(), Environment::parallelMng());
  ASSERT_EQ(global_size, vd.globalSize());
  ASSERT_EQ(2, vd.localSize());
  ASSERT_EQ(2 * rk, vd.offset());
}

TEST(TestDistribution, MatrixDefaultConstructor)
{
  Alien::MatrixDistribution vd;
  ASSERT_FALSE(vd.isParallel());
}

TEST(TestDistribution, MatrixGlobalSizeConstructor)
{
  Alien::MatrixDistribution vd(10, 5, Environment::parallelMng());
  ASSERT_EQ(vd.isParallel(), Environment::parallelMng()->commSize() > 1);
  ASSERT_EQ(vd.parallelMng(), Environment::parallelMng());
  ASSERT_EQ(10, vd.globalRowSize());
  auto gsize = Arccore::MessagePassing::mpAllReduce(
      Environment::parallelMng(), Arccore::MessagePassing::ReduceSum, vd.localRowSize());
  ASSERT_EQ(10, gsize);
  ASSERT_EQ(5, vd.globalColSize());
  // ASSERT_EQ(5, vd.localColSize());
  // ASSERT_EQ(0, vd.colOffset());
}

TEST(TestDistribution, MatrixGlobalLocalSizeConstructor)
{
  auto* pm = Environment::parallelMng();
  auto np = pm->commSize();
  auto rk = pm->commRank();
  auto global_size = 2 * np;
  Alien::MatrixDistribution vd(global_size, 5, 2, pm);
  ASSERT_EQ(vd.isParallel(), Environment::parallelMng()->commSize() > 1);
  ASSERT_EQ(vd.parallelMng(), Environment::parallelMng());
  ASSERT_EQ(global_size, vd.globalRowSize());
  ASSERT_EQ(2, vd.localRowSize());
  ASSERT_EQ(2 * rk, vd.rowOffset());
  ASSERT_EQ(5, vd.globalColSize());
  // ASSERT_EQ(5, vd.localColSize());
}

TEST(TestDistribution, MatrixGlobalLocalSize2Constructor)
{
  auto* pm = Environment::parallelMng();
  auto np = pm->commSize();
  auto rk = pm->commRank();
  auto row_global_size = 3 * np;
  auto col_global_size = 2 * np;
  Alien::MatrixDistribution vd(row_global_size, col_global_size, 3, 2, pm);
  ASSERT_EQ(vd.isParallel(), pm->commSize() > 1);
  ASSERT_EQ(vd.parallelMng(), pm);
  ASSERT_EQ(row_global_size, vd.globalRowSize());
  ASSERT_EQ(3, vd.localRowSize());
  ASSERT_EQ(3 * rk, vd.rowOffset());
  ASSERT_EQ(col_global_size, vd.globalColSize());
  ASSERT_EQ(2, vd.localColSize());
  ASSERT_EQ(2 * rk, vd.colOffset());
}
