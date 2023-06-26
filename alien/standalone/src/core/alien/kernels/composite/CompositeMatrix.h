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

#pragma once

#include <vector>

#include <alien/core/impl/IMatrixImpl.h>
#include <alien/data/IMatrix.h>
#include <alien/data/Space.h>
#include <alien/kernels/composite/CompositeBackEnd.h>
#include <alien/kernels/composite/CompositeMatrixElement.h>
#include <alien/kernels/composite/CompositeSpace.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace CompositeKernel
{

  class MultiMatrixImpl;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  class ALIEN_EXPORT Matrix : public IMatrixImpl
  {
   public:
    //! Constructor from a composite
    Matrix(MultiMatrixImpl* multi_impl);

    //! Constructor, converting data
    //! @throw Not implemented yet
    Matrix(const Alien::MultiMatrixImpl* multi_impl);

    void clear();

    void free();

    void resize(Integer nc);

    Integer size() const;

    MatrixElement element(Integer i, Integer j);

    IMatrix& operator()(Integer i, Integer j);

    const IMatrix& operator()(Integer i, Integer j) const;

    void setComposite(Integer i, Integer j, IMatrix* m);

   private:
    Integer m_nb_composite;

    std::vector<std::shared_ptr<IMatrix>> m_matrices;

    CompositeKernel::Space m_row_space;
    CompositeKernel::Space m_col_space;
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace CompositeKernel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
