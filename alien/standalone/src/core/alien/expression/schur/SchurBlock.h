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

/*!
 * \file ShurBlock.h
 * \brief ShurBlock.h
 */

#pragma once

#ifdef ALIEN_USE_EIGEN3
#include <Eigen/Core>
#include <Eigen/LU>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SchurBlock1D
{
 public:
  typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> vector;
  typedef Eigen::Map<vector> eigen_vector;

  SchurBlock1D(ArrayView<Real> block, Integer size)
  : m_block(block)
  , m_size_1(std::min(size, m_block.size()))
  , m_size_2(m_block.size() - m_size_1)
  , m_block_1(m_block.data(), m_size_1)
  , m_block_2(m_block.data() + m_size_1, m_size_2)
  {
  }

  ~SchurBlock1D() {}

  ConstArrayView<Real> block() const { return m_block; }

  eigen_vector& block_1() { return m_block_1; }
  const eigen_vector& block_1() const { return m_block_1; }

  eigen_vector& block_2() { return m_block_2; }
  const eigen_vector& block_2() const { return m_block_2; }

 private:
  ArrayView<Real> m_block;

  Integer m_size_1;
  Integer m_size_2;

  eigen_vector m_block_1;
  eigen_vector m_block_2;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SchurBlock2D
{
 public:
  typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix;
  typedef Eigen::OuterStride<> outer_stride;
  typedef Eigen::Map<matrix, 0, outer_stride> eigen_matrix;

  SchurBlock2D(Array2View<Real> block, Integer size)
  : m_block(block)
  , m_sizex_1(std::min(size, m_block.dim1Size()))
  , m_sizex_2(m_block.dim1Size() - m_sizex_1)
  , m_sizey_1(std::min(size, m_block.dim2Size()))
  , m_sizey_2(m_block.dim2Size() - m_sizey_1)
  , m_block_11(m_block.unguardedBasePointer(),
               m_sizex_1, m_sizey_1,
               outer_stride(m_block.dim2Size()))
  , m_block_12(m_block.unguardedBasePointer() + m_sizex_1,
               m_sizex_1, m_sizey_2,
               outer_stride(m_block.dim2Size()))
  , m_block_21(m_block.unguardedBasePointer() + (m_sizex_1 * m_block.dim2Size()),
               m_sizex_2, m_sizey_1,
               outer_stride(m_block.dim2Size()))
  , m_block_22(m_block.unguardedBasePointer() + (m_sizex_1 * (m_block.dim2Size() + 1)),
               m_sizex_2, m_sizey_2,
               outer_stride(m_block.dim2Size()))
  {
  }

  ~SchurBlock2D() {}

  ConstArray2View<Real> block() const { return m_block; }

  eigen_matrix& block_11() { return m_block_11; }
  const eigen_matrix& block_11() const { return m_block_11; }

  eigen_matrix& block_12() { return m_block_12; }
  const eigen_matrix& block_12() const { return m_block_12; }

  eigen_matrix& block_21() { return m_block_21; }
  const eigen_matrix& block_21() const { return m_block_21; }

  eigen_matrix& block_22() { return m_block_22; }
  const eigen_matrix& block_22() const { return m_block_22; }

 private:
  Array2View<Real> m_block;

  Integer m_sizex_1;
  Integer m_sizex_2;
  Integer m_sizey_1;
  Integer m_sizey_2;

  eigen_matrix m_block_11;
  eigen_matrix m_block_12;
  eigen_matrix m_block_21;
  eigen_matrix m_block_22;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SchurAlgo
{
 public:
  SchurAlgo() {}

  static void compute(SchurBlock2D& A, SchurBlock1D& b)
  {
    ALIEN_ASSERT((A.block().dim1Size() == A.block().dim2Size()), ("Schur complement must be applied on square block"));
    ALIEN_ASSERT((A.block().dim1Size() == b.block().size()), ("Blocks size are not equals"));
    ;

    typedef SchurBlock2D::eigen_matrix matrix;
    typedef SchurBlock1D::eigen_vector vector;

    matrix& matrix_11 = A.block_11();
    matrix& matrix_12 = A.block_12();
    matrix& matrix_21 = A.block_21();
    matrix& matrix_22 = A.block_22();

    vector& vector_1 = b.block_1();
    vector& vector_2 = b.block_2();

    /*
      A22 => A22^-1
      A21 => A22^-1 * A 21
      A11 => A11 - A12 * A22^-1 A21 
      
      b2 => A22^-1 * b2
      b1 => b1 - A12 * A22^-1 * b2
    */

    matrix_22 = matrix_22.inverse();
    matrix_21 = matrix_22 * matrix_21;
    matrix_11 -= matrix_12 * matrix_21;

    vector_2 = matrix_22 * vector_2;
    vector_1 -= matrix_12 * vector_2;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
