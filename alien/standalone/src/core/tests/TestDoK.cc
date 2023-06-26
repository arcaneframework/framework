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

#include <Environment.h>
#include <iostream>
#include <utility>

#include <alien/distribution/MatrixDistribution.h>
#include <alien/kernels/dok/DoKMatrixT.h>
#include <alien/kernels/dok/DoKVector.h>

// For DoKReverseIndexer test.
#include <alien/kernels/dok/DoKReverseIndexer.h>

// For SimpleCSR convert tests
#include <alien/kernels/dok/converters/from_simple_csr_matrix.h>
#include <alien/kernels/dok/converters/to_simple_csr_matrix.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/dok/converters/from_simple_csr_vector.h>
#include <alien/kernels/dok/converters/to_simple_csr_vector.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/kernels/dok/DoKBackEnd.h>

namespace
{
Arccore::Int32 g_row[] = { 0, 0, 1, 1, 1, 2, 2 };
Arccore::Int32 g_col[] = { 0, 1, 0, 1, 2, 0, 2 };
Arccore::Real g_value[] = { 3, -2, -1, 4, -0.5, -2, 5 };
} // namespace
using namespace Alien;
/*
 * Auxiliary class to build toy matrices.
 */
class TestDoKBuilder
{
 public:
  TestDoKBuilder(int num_rows = 3, bool square = false)
  : m_num_rows(num_rows)
  , m_num_cols(num_rows)
  , m_mdist(m_num_rows, m_num_cols, AlienTest::Environment::parallelMng())
  , m_row_space(new Alien::Space(m_mdist.globalRowSize(), "RowSpace"))
  , m_col_space(new Alien::Space(m_mdist.globalColSize(), "ColSpace"))
  , m_multimat()
  {
    if (square)
      m_col_space = m_row_space;
    m_multimat.reset(
    new Alien::MultiMatrixImpl(m_row_space, m_col_space, m_mdist.clone()));
  }

  template <class Matrix>
  Matrix* createEmptyMatrix() const
  {
    return new Matrix(m_multimat.get());
  }

  void fill(Alien::DoKMatrix& mat) const
  {
    int lenght = sizeof(g_row) / sizeof(Arccore::Int32);
    for (int i = 0; i < lenght; i++) {
      if (g_row[i] >= row_begin() && g_row[i] < row_end())
        mat.setNNZ(g_row[i], g_col[i], g_value[i]);
    }
  }

  Arccore::Int32 row_begin() const { return m_mdist.rowOffset(); }

  Arccore::Int32 row_end() const { return m_mdist.rowOffset() + m_mdist.localRowSize(); }

 private:
  Arccore::Int32 m_num_rows;
  Arccore::Int32 m_num_cols;
  Alien::MatrixDistribution m_mdist;
  std::shared_ptr<Alien::Space> m_row_space;
  std::shared_ptr<Alien::Space> m_col_space;
  std::unique_ptr<Alien::MultiMatrixImpl> m_multimat;
};

TEST(TestDoKMatrix, DoKReverseIndexer)
{
  Alien::DoKReverseIndexer r_index;

  std::cout << std::is_same<int, Arccore::Int32>::value << std::endl;
  std::cout << std::is_same<long long, Arccore::Int64>::value << std::endl;
  // std::cout << std::is_same<long long, Alien::DoKReverseIndexer::Index>::value <<
  // std::endl;

  std::pair<int, int> i1(0, 0);
  long long o(0);
  r_index.record(o, i1);

  Alien::DoKReverseIndexer::Index i2(1, 0);
  r_index.record(1, i2);
  Alien::DoKReverseIndexer::Index i3(0, 1);
  r_index.record(2, i3);
  Alien::DoKReverseIndexer::Index i4(1, 1);
  r_index.record(3, i4);

  ASSERT_EQ(i1, r_index[0]);
  ASSERT_EQ(i2, r_index[1]);
  ASSERT_EQ(i3, r_index[2]);
  ASSERT_EQ(i4, r_index[3]);
}

TEST(TestDoKMatrix, Constructor)
{
  TestDoKBuilder builder;
  std::unique_ptr<Alien::DoKMatrix> mat(builder.createEmptyMatrix<Alien::DoKMatrix>());

  builder.fill(*mat);
  mat->compact();
  mat->assemble();

  auto& data = mat->data();
  auto r_index = data.getReverseIndexer();

  auto start = builder.row_begin();
  auto stop = builder.row_end();
  for (int i = 0; i < r_index->size(); i++) {
    auto cur = (*r_index)[i];
    ASSERT_GE(cur.value().first, start);
    ASSERT_LT(cur.value().first, stop);
  }
}

TEST(TestDoKMatrix, ConvertFromCSR)
{
  Arccore::Int32 rows = 5;
  Arccore::Int32 cols = 5;
  auto pm = AlienTest::Environment::parallelMng();
  Alien::MatrixDistribution mdist(rows, cols, pm);
  Alien::Space row_space(mdist.globalRowSize(), "Space");
  Alien::Space col_space(mdist.globalColSize(), "Space");

  std::unique_ptr<Alien::MultiMatrixImpl> multimat(
  new Alien::MultiMatrixImpl(row_space.clone(), col_space.clone(), mdist.clone()));

  auto& csr_mat(multimat->get<Alien::BackEnd::tag::simplecsr>());
  auto& dok_mat(multimat->get<Alien::BackEnd::tag::DoK>(true));

  Alien::SimpleCSRtoDoKMatrixConverter converter;
  converter.convert(&csr_mat, &dok_mat);

  dok_mat.assemble();
}

TEST(TestDoKMatrix, ConvertToCSR)
{
  TestDoKBuilder builder(3, true);
  std::unique_ptr<Alien::DoKMatrix> dok_mat(
  builder.createEmptyMatrix<Alien::DoKMatrix>());

  builder.fill(*dok_mat);
  dok_mat->compact();

  dok_mat->setNNZ(0, 2, -5.);

  dok_mat->assemble();

  typedef Alien::SimpleCSRMatrix<Real> SimpleCSR;
  std::unique_ptr<SimpleCSR> csr_mat(builder.createEmptyMatrix<SimpleCSR>());
  Alien::DoKtoSimpleCSRMatrixConverter converter;
  converter.convert(dok_mat.get(), csr_mat.get());
}

TEST(TestDoKVector, Build)
{
  auto space = std::make_shared<Space>(Space(10));
  auto vd = std::make_shared<VectorDistribution>(VectorDistribution(*space, AlienTest::Environment::parallelMng()));
  DoKVector v(new Alien::MultiVectorImpl(space, vd));

  for (int i = 0; i < space->size(); ++i) {
    v.contribute(i, i + 1.0);
  }
  v.assemble();
}

TEST(TestDoKVector, ConvertToCSR)
{
  auto space = std::make_shared<Space>(Space(10));
  auto vd = std::make_shared<VectorDistribution>(VectorDistribution(*space, AlienTest::Environment::parallelMng()));
  Alien::MultiVectorImpl multi(space, vd);
  DoKVector v(&multi);

  if (!vd->parallelMng()->commRank()) {
    for (int i = 0; i < space->size(); ++i) {
      v.contribute(i, i + 1.0);
    }
  }
  v.assemble();

  typedef Alien::SimpleCSRVector<Real> SimpleCSRVect;
  SimpleCSRVect csr_vect(&multi);
  Alien::DoKToSimpleCSRVectorConverter converter;
  converter.convert(&v, &csr_vect);

  for (int i = 0; i < vd->localSize(); ++i) {
    ASSERT_EQ(csr_vect[i], vd->offset() + i + 1.0);
  }
}
