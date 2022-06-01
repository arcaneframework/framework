/*
 * Copyright 2022 IFPEN-CEA
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
#include <alien/trilinos/backend.h>
#include <alien/trilinos/trilinos_config.h>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Core.hpp>
#include <Teuchos_ParameterXMLFileReader.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_DefaultMpiComm.hpp>

namespace Alien::Trilinos
{
class Matrix : public IMatrixImpl
{

 public:
  explicit Matrix(const MultiMatrixImpl* multi_impl);

  ~Matrix() final = default;

  void setProfile(int numLocalRows, int numGlobalRows, const Arccore::UniqueArray<int>& rowSizes);

  void setRowValues(int rows,
                    Arccore::ConstArrayView<int> cols,
                    Arccore::ConstArrayView<double> values);

  void assemble();

  Teuchos::RCP<crs_matrix_type> const& internal() const { return mtx; }
  Teuchos::RCP<crs_matrix_type>& internal() { return mtx; }
  Teuchos::RCP<const Teuchos::Comm<int>> getComm() const { return t_comm; };

 private:
  Teuchos::RCP<crs_matrix_type> mtx;
  Teuchos::RCP<const Teuchos::Comm<int>> t_comm;
};

} // namespace Alien::Trilinos
