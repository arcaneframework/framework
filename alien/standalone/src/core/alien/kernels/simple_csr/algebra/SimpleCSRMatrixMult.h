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



*/

#pragma once

#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRPrecomp.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>

namespace Alien::SimpleCSRInternal
{

/*@! Classe amie de SimpleCSRMatrix pour externaliser plus rapidement (mais moins
 * proprement)
 * le produit matrice vecteur */
template <typename ValueT>
class SimpleCSRMatrixMultT
{
 public:
  //! Template parameter
  typedef ValueT ValueType;
  typedef SimpleCSRMatrix<ValueType> MatrixType;
  typedef SimpleCSRVector<ValueType> VectorType;

 public:
  //! Constructeur de la classe
  SimpleCSRMatrixMultT(const MatrixType& matrix);

  //! Destructeur de la classe
  virtual ~SimpleCSRMatrixMultT() {}

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

  void _seqAddLMult(Real alpha, const VectorType& x, VectorType& y) const;
  void _seqAddUMult(Real alpha, const VectorType& x, VectorType& y) const;

  void _parallelMultBlock(const VectorType& x, VectorType& y) const;

  void _seqMultBlock(const VectorType& x, VectorType& y) const;

  void _parallelMultVariableBlock(const VectorType& x, VectorType& y) const;

  void _seqMultVariableBlock(const VectorType& x, VectorType& y) const;

 private:
  const MatrixType& m_matrix_impl;
};

} // namespace Alien::SimpleCSRInternal

#include "SimpleCSRMatrixMultT.h"
