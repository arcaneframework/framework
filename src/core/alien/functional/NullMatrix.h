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

#include <alien/data/IMatrix.h>
#include <alien/data/Space.h>
#include <alien/distribution/MatrixDistribution.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NullMatrix final : public IMatrix
{
 public:
  NullMatrix();

  ~NullMatrix() {}

  NullMatrix(const NullMatrix&) = delete;
  NullMatrix(NullMatrix&&) = delete;
  NullMatrix& operator=(NullMatrix&& vector) = delete;
  NullMatrix& operator=(const NullMatrix&) = delete;

 public:
  // Pour les expressions
  void visit(ICopyOnWriteMatrix&) const;

  const MatrixDistribution& distribution() const;

  const Space& rowSpace() const;
  const Space& colSpace() const;

 public:
  MultiMatrixImpl* impl();

  const MultiMatrixImpl* impl() const;

 private:
  Space m_space;
  MatrixDistribution m_distribution;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
