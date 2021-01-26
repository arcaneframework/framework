#include <gtest/gtest.h>

#include <alien/ref/AlienRefSemantic.h>

namespace Environment {
extern Arccore::MessagePassing::IMessagePassingMng* parallelMng();
}

TEST(TestVBlockMatrix, DefaultConstructor)
{
  Alien::VBlockMatrix m;
  ASSERT_EQ(m.rowSpace().size(), 0);
  ASSERT_EQ(m.colSpace().size(), 0);
  ASSERT_THROW(m.vblock(), Alien::FatalErrorException);
  ASSERT_THROW(m.rowBlock(), Alien::FatalErrorException);
  ASSERT_THROW(m.colBlock(), Alien::FatalErrorException);
}

TEST(TestVBlockMatrix, OneSpaceConstructor)
{
  Alien::Integer size = 10;
  Alien::Space s(size, "MySpace");
  Alien::MatrixDistribution mdist(s, s, Environment::parallelMng());
  Alien::VBlock::ValuePerBlock v;
  for (Alien::Integer i = 0; i < size; ++i)
    v[i] = i;
  Alien::VBlock block(std::move(v));
  Alien::VBlockMatrix m(block, mdist);
  ASSERT_EQ(m.rowSpace().size(), size);
  ASSERT_EQ(m.colSpace().size(), size);
  ASSERT_EQ(m.distribution().globalRowSize(), size);
  ASSERT_EQ(m.distribution().globalColSize(), size);
  ASSERT_NO_THROW(m.vblock());
  ASSERT_NO_THROW(m.rowBlock());
  ASSERT_NO_THROW(m.colBlock());
  for (Alien::Integer i = 0; i < size; ++i) {
    ASSERT_EQ(m.vblock().size(i), i);
    ASSERT_EQ(m.rowBlock().size(i), i);
    ASSERT_EQ(m.colBlock().size(i), i);
  }
  ASSERT_EQ(m.vblock().maxBlockSize(), size - 1);
  ASSERT_EQ(m.rowBlock().maxBlockSize(), size - 1);
  ASSERT_EQ(m.colBlock().maxBlockSize(), size - 1);
}

TEST(TestVBlockMatrix, TwoSpacesConstructor)
{
  Alien::Integer rowSize = 10;
  Alien::Integer colSize = 5;
  Alien::Space rowS(rowSize, "MySpace");
  Alien::Space colS(colSize, "MySpace2");
  Alien::MatrixDistribution mdist(rowS, colS, Environment::parallelMng());
  Alien::VBlock::ValuePerBlock vRow, vCol;
  for (Alien::Integer i = 0; i < rowSize; ++i)
    vRow[i] = i;
  for (Alien::Integer i = 0; i < colSize; ++i)
    vCol[i] = rowSize + i;
  Alien::VBlock vBlockRow(std::move(vRow));
  Alien::VBlock vBlockCol(std::move(vCol));
  Alien::VBlockMatrix m(vBlockRow, vBlockCol, mdist);
  ASSERT_EQ(m.rowSpace().size(), rowSize);
  ASSERT_EQ(m.colSpace().size(), colSize);
  ASSERT_EQ(m.distribution().globalRowSize(), rowSize);
  ASSERT_EQ(m.distribution().globalColSize(), colSize);
  ASSERT_NO_THROW(m.vblock());
  ASSERT_NO_THROW(m.rowBlock());
  ASSERT_NO_THROW(m.colBlock());
  for (Alien::Integer i = 0; i < rowSize; ++i) {
    ASSERT_EQ(m.vblock().size(i), i);
    ASSERT_EQ(m.rowBlock().size(i), i);
  }
  ASSERT_EQ(m.vblock().maxBlockSize(), rowSize - 1);
  ASSERT_EQ(m.rowBlock().maxBlockSize(), rowSize - 1);

  for (Alien::Integer i = 0; i < colSize; ++i) {
    ASSERT_EQ(m.colBlock().size(i), i + rowSize);
  }
  ASSERT_EQ(m.colBlock().maxBlockSize(), rowSize + colSize - 1);
}

TEST(TestVBlockMatrix, OneSizeConstructor)
{
  Alien::Integer size = 10;
  Alien::MatrixDistribution mdist(size, size, Environment::parallelMng());
  Alien::VBlock::ValuePerBlock v;
  for (Alien::Integer i = 0; i < size; ++i)
    v[i] = i;
  Alien::VBlock block(std::move(v));
  Alien::VBlockMatrix m(block, mdist);
  ASSERT_EQ(m.rowSpace().size(), size);
  ASSERT_EQ(m.colSpace().size(), size);
  ASSERT_EQ(m.distribution().globalRowSize(), size);
  ASSERT_EQ(m.distribution().globalColSize(), size);
  ASSERT_NO_THROW(m.vblock());
  ASSERT_NO_THROW(m.rowBlock());
  ASSERT_NO_THROW(m.colBlock());
  for (Alien::Integer i = 0; i < size; ++i) {
    ASSERT_EQ(m.vblock().size(i), i);
    ASSERT_EQ(m.rowBlock().size(i), i);
    ASSERT_EQ(m.colBlock().size(i), i);
  }
  ASSERT_EQ(m.vblock().maxBlockSize(), size - 1);
  ASSERT_EQ(m.rowBlock().maxBlockSize(), size - 1);
  ASSERT_EQ(m.colBlock().maxBlockSize(), size - 1);
}

TEST(TestVBlockMatrix, TwoSizesConstructor)
{
  Alien::Integer rowSize = 10;
  Alien::Integer colSize = 5;
  Alien::MatrixDistribution mdist(rowSize, colSize, Environment::parallelMng());
  Alien::VBlock::ValuePerBlock vRow, vCol;
  for (Alien::Integer i = 0; i < rowSize; ++i)
    vRow[i] = i;
  for (Alien::Integer i = 0; i < colSize; ++i)
    vCol[i] = rowSize + i;
  Alien::VBlock vBlockRow(std::move(vRow));
  Alien::VBlock vBlockCol(std::move(vCol));
  Alien::VBlockMatrix m(vBlockRow, vBlockCol, mdist);
  ASSERT_EQ(m.rowSpace().size(), rowSize);
  ASSERT_EQ(m.colSpace().size(), colSize);
  ASSERT_EQ(m.distribution().globalRowSize(), rowSize);
  ASSERT_EQ(m.distribution().globalColSize(), colSize);
  ASSERT_NO_THROW(m.vblock());
  ASSERT_NO_THROW(m.rowBlock());
  ASSERT_NO_THROW(m.colBlock());
  for (Alien::Integer i = 0; i < rowSize; ++i) {
    ASSERT_EQ(m.vblock().size(i), i);
    ASSERT_EQ(m.rowBlock().size(i), i);
  }
  ASSERT_EQ(m.vblock().maxBlockSize(), rowSize - 1);
  ASSERT_EQ(m.rowBlock().maxBlockSize(), rowSize - 1);

  for (Alien::Integer i = 0; i < colSize; ++i) {
    ASSERT_EQ(m.colBlock().size(i), i + rowSize);
  }
  ASSERT_EQ(m.colBlock().maxBlockSize(), rowSize + colSize - 1);
}

TEST(TestVBlockMatrix, RValueConstructor)
{
  Alien::Integer size = 10;
  Alien::MatrixDistribution mdist(size, size, Environment::parallelMng());
  Alien::VBlock::ValuePerBlock v;
  for (Alien::Integer i = 0; i < size; ++i)
    v[i] = i;
  Alien::VBlock block(std::move(v));
  Alien::VBlockMatrix m(block, mdist);
  Alien::VBlockMatrix m2(std::move(m));
  ASSERT_EQ(m2.rowSpace().size(), size);
  ASSERT_EQ(m2.colSpace().size(), size);
  ASSERT_EQ(m2.distribution().globalRowSize(), size);
  ASSERT_EQ(m2.distribution().globalColSize(), size);
  ASSERT_NO_THROW(m2.vblock());
  ASSERT_NO_THROW(m2.rowBlock());
  ASSERT_NO_THROW(m2.colBlock());
  for (Alien::Integer i = 0; i < size; ++i) {
    ASSERT_EQ(m2.vblock().size(i), i);
    ASSERT_EQ(m2.rowBlock().size(i), i);
    ASSERT_EQ(m2.colBlock().size(i), i);
  }
  ASSERT_EQ(m2.vblock().maxBlockSize(), size - 1);
  ASSERT_EQ(m2.rowBlock().maxBlockSize(), size - 1);
  ASSERT_EQ(m2.colBlock().maxBlockSize(), size - 1);
}

TEST(TestVBlockMatrix, RValueAssignment)
{
  Alien::Integer size = 10;
  Alien::MatrixDistribution mdist(size, size, Environment::parallelMng());
  Alien::VBlock::ValuePerBlock v;
  for (Alien::Integer i = 0; i < size; ++i)
    v[i] = i;
  Alien::VBlock block(std::move(v));
  Alien::VBlockMatrix m(block, mdist);
  Alien::VBlockMatrix m2;
  m2 = std::move(m);
  ASSERT_EQ(m2.rowSpace().size(), size);
  ASSERT_EQ(m2.colSpace().size(), size);
  ASSERT_EQ(m2.distribution().globalRowSize(), size);
  ASSERT_EQ(m2.distribution().globalColSize(), size);
  ASSERT_NO_THROW(m2.vblock());
  ASSERT_NO_THROW(m2.rowBlock());
  ASSERT_NO_THROW(m2.colBlock());
  for (Alien::Integer i = 0; i < size; ++i) {
    ASSERT_EQ(m2.vblock().size(i), i);
    ASSERT_EQ(m2.rowBlock().size(i), i);
    ASSERT_EQ(m2.colBlock().size(i), i);
  }
  ASSERT_EQ(m2.vblock().maxBlockSize(), size - 1);
  ASSERT_EQ(m2.rowBlock().maxBlockSize(), size - 1);
  ASSERT_EQ(m2.colBlock().maxBlockSize(), size - 1);
}

TEST(TestVBlockMatrix, InitMethod)
{
  Alien::Integer size = 10;
  Alien::MatrixDistribution mdist(size, size, Environment::parallelMng());
  Alien::VBlock::ValuePerBlock v;
  for (Alien::Integer i = 0; i < size; ++i)
    v[i] = i;
  Alien::VBlock block(std::move(v));
  Alien::VBlockMatrix m;
  m.init(block, mdist);
  ASSERT_EQ(m.rowSpace().size(), size);
  ASSERT_EQ(m.colSpace().size(), size);
  ASSERT_EQ(m.distribution().globalRowSize(), size);
  ASSERT_EQ(m.distribution().globalColSize(), size);
  ASSERT_NO_THROW(m.vblock());
  ASSERT_NO_THROW(m.rowBlock());
  ASSERT_NO_THROW(m.colBlock());
  for (Alien::Integer i = 0; i < size; ++i) {
    ASSERT_EQ(m.vblock().size(i), i);
    ASSERT_EQ(m.rowBlock().size(i), i);
    ASSERT_EQ(m.colBlock().size(i), i);
  }
  ASSERT_EQ(m.vblock().maxBlockSize(), size - 1);
  ASSERT_EQ(m.rowBlock().maxBlockSize(), size - 1);
  ASSERT_EQ(m.colBlock().maxBlockSize(), size - 1);
}
