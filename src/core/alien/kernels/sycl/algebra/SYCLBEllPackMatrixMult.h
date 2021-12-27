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

#include <alien/kernels/sycl/SYCLPrecomp.h>
#include <alien/kernels/sycl/data/SYCLBEllPackMatrix.h>
#include <alien/kernels/sycl/data/SYCLVector.h>

namespace Alien::SYCLInternal
{

/*@! Classe amie de SYCLMatrix pour externaliser plus rapidement (mais moins
 * proprement)
 * le produit matrice vecteur */
template <typename ValueT>
class SYCLBEllPackMatrixMultT
{
 public:
  //! Template parameter
  // clang-format off
  typedef ValueT                        ValueType;
  typedef SYCLBEllPackMatrix<ValueType> MatrixType;
  typedef SYCLVector<ValueType>         VectorType;
  // clang-format on

 public:
  //! Constructeur de la classe
  SYCLBEllPackMatrixMultT(const MatrixType& matrix);

  //! Destructeur de la classe
  virtual ~SYCLBEllPackMatrixMultT() {}

 public:
  //! Matrix vector product
  void mult(const VectorType& x, VectorType& y) const;
  void mult(const UniqueArray<Real>& x, UniqueArray<Real>& y) const;

  void addLMult(Real alpha, const VectorType& x, VectorType& y) const;
  void addUMult(Real alpha, const VectorType& x, VectorType& y) const;

  void computeInvDiag(VectorType& y) const;
  void multInvDiag(VectorType& y) const;

 private:
  void _parallelMult(const VectorType& x, VectorType& y) const;
  void _parallelMult(const UniqueArray<Real>& x, UniqueArray<Real>& y) const;

  void _seqMult(const VectorType& x, VectorType& y) const;
  void _seqMult(const UniqueArray<Real>& x, UniqueArray<Real>& y) const;

 private:
  const MatrixType& m_matrix_impl;
};

} // namespace Alien::SYCLInternal

#include "SYCLBEllPackMatrixMultT.h"
