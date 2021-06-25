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

#include "CompositeMatrix.h"
#include "CompositeMultiMatrixImpl.h"
#include "CompositeSpace.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace CompositeKernel
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  Matrix::Matrix(const Alien::MultiMatrixImpl* multi_impl)
  : IMatrixImpl(multi_impl, AlgebraTraits<BackEnd::tag::composite>::name())
  , m_nb_composite(0)
  {
    alien_fatal([&] {
      cout() << "CompositeMatrix(const Alien::MultiMatrixImpl*) : Not implemented";
    });
  }

  /*---------------------------------------------------------------------------*/

  Matrix::Matrix(MultiMatrixImpl* multi_impl)
  : IMatrixImpl(multi_impl, AlgebraTraits<BackEnd::tag::composite>::name())
  , m_nb_composite(0)
  , m_row_space(multi_impl->rowSpace())
  , m_col_space(multi_impl->colSpace())
  {
    alien_debug([&] { cout() << "Construct CompositeMatrix " << this; });
  }

  /*---------------------------------------------------------------------------*/

  void Matrix::clear()
  {
    alien_debug([&] { cout() << "Clear CompositeMatrix" << this; });

    for (auto& m : m_matrices) {
      if (m->impl())
        m->impl()->clear();
    }
  }

  /*---------------------------------------------------------------------------*/

  void Matrix::free()
  {
    alien_debug([&] { cout() << "Free CompositeMatrix" << this; });

    for (auto& m : m_matrices) {
      if (m->impl())
        m->impl()->free();
    }
  }

  /*---------------------------------------------------------------------------*/

  void Matrix::resize(Integer nc)
  {
    alien_debug([&] {
      cout() << "Resize CompositeMatrix" << this;
      cout() << " - old size = " << m_nb_composite;
      cout() << " - new size = " << nc;
    });

    m_row_space.resizeSubSpace(nc);
    m_col_space.resizeSubSpace(nc);

    m_nb_composite = nc;

    m_matrices.resize(nc * nc);
  }

  /*---------------------------------------------------------------------------*/

  Integer Matrix::size() const { return m_nb_composite; }

  /*---------------------------------------------------------------------------*/

  MatrixElement Matrix::element(Integer i, Integer j)
  {
    ALIEN_ASSERT(i < size(), "Bound error");
    ALIEN_ASSERT(j < size(), "Bound error");
    return MatrixElement(
    m_matrices[i + j * m_nb_composite], m_row_space[i], m_col_space[j], *this);
  }

  /*---------------------------------------------------------------------------*/

  IMatrix& Matrix::operator()(Integer i, Integer j)
  {
    ALIEN_ASSERT(i < size(), "Bound error");
    ALIEN_ASSERT(j < size(), "Bound error");
    return *m_matrices[i + j * m_nb_composite];
  }

  /*---------------------------------------------------------------------------*/

  const IMatrix& Matrix::operator()(Integer i, Integer j) const
  {
    ALIEN_ASSERT(i < size(), "Bound error");
    ALIEN_ASSERT(j < size(), "Bound error");
    return *m_matrices[i + j * m_nb_composite];
  }

  /*---------------------------------------------------------------------------*/

  void Matrix::setComposite(Integer i, Integer j, IMatrix* m)
  {
    ALIEN_ASSERT(i < size(), "Bound error");
    ALIEN_ASSERT(j < size(), "Bound error");
    m_matrices[i + j * m_nb_composite].reset(m);
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace CompositeKernel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
