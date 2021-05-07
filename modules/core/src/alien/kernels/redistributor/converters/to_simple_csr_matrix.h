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

#include <alien/core/backend/IMatrixConverter.h>
#include <alien/core/backend/MatrixConverterRegisterer.h>
#include <alien/utils/Precomp.h>

namespace Alien
{

template <typename T>
class SimpleCSRMatrix;
class RedistributorMatrix;

class RedistributorToSimpleCSRMatrixConverter : public IMatrixConverter
{
 public:
  typedef RedistributorMatrix SrcMatrix;
  typedef SimpleCSRMatrix<Real> TgtMatrix;

 public:
  RedistributorToSimpleCSRMatrixConverter();
  virtual ~RedistributorToSimpleCSRMatrixConverter();

 public:
  BackEndId sourceBackend() const override;
  BackEndId targetBackend() const override;

  void convert(const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const override;
};

} // namespace Alien
