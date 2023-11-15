/// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

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

class ALIEN_EXPORT SchurBlock1D
{
 public:
#ifdef ALIEN_USE_EIGEN3
  typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> vector;
  typedef Eigen::Map<vector> eigen_vector;
#endif
  SchurBlock1D(ArrayView<Real> block, Integer size)
  : m_block(block)
  , m_size_1(std::min(size, m_block.size()))
  , m_size_2(m_block.size() - m_size_1)
#ifdef ALIEN_USE_EIGEN3
  , m_block_1(m_block.data(), m_size_1)
  , m_block_2(m_block.data() + m_size_1, m_size_2)
#endif
  {
  }

  virtual ~SchurBlock1D() {}

  ConstArrayView<Real> block() const { return m_block; }

#ifdef ALIEN_USE_EIGEN3
  eigen_vector& block_1() { return m_block_1; }
  const eigen_vector& block_1() const { return m_block_1; }

  eigen_vector& block_2() { return m_block_2; }
  const eigen_vector& block_2() const { return m_block_2; }
#endif
 private:
  ArrayView<Real> m_block;

  Integer m_size_1;
  Integer m_size_2;

#ifdef ALIEN_USE_EIGEN3
  eigen_vector m_block_1;
  eigen_vector m_block_2;
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_EXPORT SchurBlock2D
{
 public:
#ifdef ALIEN_USE_EIGEN3
  typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix;
  typedef Eigen::OuterStride<> outer_stride;
  typedef Eigen::Map<matrix, 0, outer_stride> eigen_matrix;
#endif
  SchurBlock2D(Array2View<Real> block, Integer size)
  : m_block(block)
  , m_sizex_1(std::min(size, m_block.dim1Size()))
  , m_sizex_2(m_block.dim1Size() - m_sizex_1)
  , m_sizey_1(std::min(size, m_block.dim2Size()))
  , m_sizey_2(m_block.dim2Size() - m_sizey_1)
#ifdef ALIEN_USE_EIGEN3
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
#endif
  {
  }

  virtual ~SchurBlock2D() {}

  ConstArray2View<Real> block() const { return m_block; }

#ifdef ALIEN_USE_EIGEN3
  eigen_matrix& block_11() { return m_block_11; }
  const eigen_matrix& block_11() const { return m_block_11; }

  eigen_matrix& block_12() { return m_block_12; }
  const eigen_matrix& block_12() const { return m_block_12; }

  eigen_matrix& block_21() { return m_block_21; }
  const eigen_matrix& block_21() const { return m_block_21; }

  eigen_matrix& block_22() { return m_block_22; }
  const eigen_matrix& block_22() const { return m_block_22; }
#endif
 private:
  Array2View<Real> m_block;

  Integer m_sizex_1;
  Integer m_sizex_2;
  Integer m_sizey_1;
  Integer m_sizey_2;

#ifdef ALIEN_USE_EIGEN3
  eigen_matrix m_block_11;
  eigen_matrix m_block_12;
  eigen_matrix m_block_21;
  eigen_matrix m_block_22;
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_EXPORT SchurAlgo
{
 public:
  SchurAlgo() {}

  static void compute(SchurBlock2D& A, SchurBlock1D& b)
  {
    ALIEN_ASSERT((A.block().dim1Size() == A.block().dim2Size()), ("Schur complement must be applied on square block"));
    ALIEN_ASSERT((A.block().dim1Size() == b.block().size()), ("Blocks size are not equals"));
    ;

#ifdef ALIEN_USE_EIGEN3
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
#endif
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
