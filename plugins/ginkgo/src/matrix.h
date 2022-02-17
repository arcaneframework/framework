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

#include <alien/core/impl/IMatrixImpl.h>

#include <alien/ginkgo/backend.h>
#include <alien/ginkgo/machine_backend.h>

#include <ginkgo/core/matrix/csr.hpp>

#include <memory>

namespace Alien::Ginkgo
{
class Matrix : public IMatrixImpl
, public gko::matrix::Csr<double, int>
{
 public:
  explicit Matrix(const MultiMatrixImpl* multi_impl);

  ~Matrix() override;

 public:
  void setRowValues(int rows,
                    Arccore::ConstArrayView<int> cols,
                    Arccore::ConstArrayView<double>);

  void assemble();

  /* Return a raw pointer */
  gko::matrix::Csr<double, int> const* internal() const
  {
    return this;
  }

  gko::matrix::Csr<double, int>* internal()
  {
    return this;
  }

 private:
  gko::matrix_assembly_data<double, int> data;
};

} // namespace Alien::Ginkgo
