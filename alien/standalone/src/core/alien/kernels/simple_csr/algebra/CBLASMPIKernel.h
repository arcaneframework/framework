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

#include <cmath>
#include <arccore/message_passing/Messages.h>

#include <alien/utils/Precomp.h>
#include <alien/kernels/simple_csr/algebra/alien_cblas.h>

namespace Alien
{

class CBLASMPIKernel
{
 public:
  // static const tag::eType  type       = tag::CPU;
  static const bool is_hybrid = false;
  static const bool is_mpi = true;

  template <typename Distribution, typename VectorT>
  static void copy(
  Distribution const& dist ALIEN_UNUSED_PARAM, const VectorT& x, VectorT& y)
  {
    typedef typename VectorT::ValueType ValueType;
    cblas::copy(
    x.scalarizedLocalSize(), (ValueType*)x.getDataPtr(), 1, y.getDataPtr(), 1);
  }

  template <typename Distribution, typename VectorT>
  static void axpy(Distribution const& dist ALIEN_UNUSED_PARAM,
                   typename VectorT::ValueType alpha, const VectorT& x, VectorT& y)
  {
    cblas::axpy(x.scalarizedLocalSize(), alpha, x.getDataPtr(), 1, y.getDataPtr(), 1);
  }

  template <typename Distribution, typename VectorT>
  static void scal(Distribution const& dist ALIEN_UNUSED_PARAM,
                   typename VectorT::ValueType alpha, VectorT& x)
  {
    cblas::scal(x.scalarizedLocalSize(), alpha, x.getDataPtr(), 1);
  }

  template <typename Distribution, typename VectorT>
  static void pointwiseMult(Distribution const& dist,
                            VectorT const& x,
                            VectorT const& y,
                            VectorT& z)
  {
    auto local_size = x.scalarizedLocalSize();
    auto x_ptr = x.getDataPtr();
    auto y_ptr = y.getDataPtr();
    auto z_ptr = z.getDataPtr();
    for (std::size_t i = 0; i < local_size; ++i) {
      z_ptr[i] = x_ptr[i] * y_ptr[i];
    }
  }

  template <typename Distribution, typename VectorT>
  static void assign(Distribution const& dist,
                     typename VectorT::ValueType alpha,
                     VectorT& y)
  {
    auto local_size = y.scalarizedLocalSize();
    auto y_ptr = y.getDataPtr();
    for (std::size_t i = 0; i < local_size; ++i) {
      y_ptr[i] = alpha;
    }
  }

  template <typename Distribution, typename VectorT>
  static typename VectorT::ValueType dot(
  Distribution const& dist, const VectorT& x, const VectorT& y)
  {
    typedef typename VectorT::ValueType ValueType;
    ValueType value = cblas::dot(x.scalarizedLocalSize(), (ValueType*)x.getDataPtr(), 1,
                                 (ValueType*)y.getDataPtr(), 1);
    if (dist.isParallel()) {
      return Arccore::MessagePassing::mpAllReduce(
      dist.parallelMng(), Arccore::MessagePassing::ReduceSum, value);
    }
    return value;
  }

  template <typename Distribution, typename VectorT>
  static typename VectorT::ValueType nrm1(Distribution const& dist, const VectorT& x)
  {
    typedef typename VectorT::ValueType ValueType;
    typename VectorT::ValueType value = cblas::nrm1(x.scalarizedLocalSize(),
                                                    (ValueType*)x.getDataPtr(), 1);
    if (dist.isParallel()) {
      value = Arccore::MessagePassing::mpAllReduce(
      dist.parallelMng(), Arccore::MessagePassing::ReduceSum, value);
    }
    return value;
  }

  template <typename Distribution, typename VectorT>
  static typename VectorT::ValueType nrm2(Distribution const& dist, const VectorT& x)
  {
    typedef typename VectorT::ValueType ValueType;
    typename VectorT::ValueType value = cblas::dot(x.scalarizedLocalSize(),
                                                   (ValueType*)x.getDataPtr(), 1, (ValueType*)x.getDataPtr(), 1);
    if (dist.isParallel()) {
      value = Arccore::MessagePassing::mpAllReduce(
      dist.parallelMng(), Arccore::MessagePassing::ReduceSum, value);
    }
    return std::sqrt(value);
  }

  template <typename Distribution, typename MatrixT>
  static typename MatrixT::ValueType matrix_nrm2(Distribution const& dist, const MatrixT& x)
  {
    typedef typename MatrixT::ValueType ValueType;
    typename MatrixT::ValueType value = cblas::dot(x.getProfile().getNnz(),
                                                   (ValueType*)x.data(), 1, (ValueType*)x.data(), 1);
    if (dist.isParallel()) {
      value = Arccore::MessagePassing::mpAllReduce(
      dist.parallelMng(), Arccore::MessagePassing::ReduceSum, value);
    }
    return std::sqrt(value);
  }
};

} // namespace Alien
