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
/*
 * MCILU0Preconditioner.h
 *
 *  Created on: Sep 20, 2010
 *      Author: gratienj
 */

#pragma once

namespace Alien
{
template <typename AlgebraT>
class DiagPreconditioner
{
 public:
  // clang-format off
    typedef AlgebraT                        AlgebraType ;
    typedef typename AlgebraType::Matrix    MatrixType;
    typedef typename AlgebraType::Vector    VectorType;
    typedef typename MatrixType::ValueType  ValueType;
  // clang-format on

  DiagPreconditioner(MatrixType const& matrix)
  : m_matrix(matrix)
  {}

  virtual ~DiagPreconditioner(){};

  //! operator preparation
  void init()
  {
  }

  void update()
  {
    // update value from m_matrix
  }

  template <typename AlgebraType>
  void solve(AlgebraType& algebra,
             VectorType const& x,
             VectorType& y) const
  {
    algebra.copy(x, y);
  }

 private:
  MatrixType const& m_matrix;
};

} // namespace Alien
