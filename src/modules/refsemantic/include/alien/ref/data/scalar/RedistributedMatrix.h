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
#include <alien/distribution/MatrixDistribution.h>
#include <alien/kernels/redistributor/Redistributor.h>
#include <alien/ref/AlienRefSemanticPrecomp.h>
#include <alien/utils/ICopyOnWriteObject.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_REFSEMANTIC_EXPORT RedistributedMatrix final : public IMatrix
{
 public:
  RedistributedMatrix() = delete;
  RedistributedMatrix(RedistributedMatrix&& matrix) = delete;
  RedistributedMatrix(IMatrix& matrix, Redistributor& redist);

  ~RedistributedMatrix();

  RedistributedMatrix& operator=(RedistributedMatrix&& matrix) = delete;

  RedistributedMatrix(const RedistributedMatrix& matrix) = delete;
  RedistributedMatrix& operator=(const RedistributedMatrix& matrix) = delete;

 public:
  // Pour les expressions
  void visit(ICopyOnWriteMatrix&) const;

  const MatrixDistribution& distribution() const;

  const ISpace& rowSpace() const;
  const ISpace& colSpace() const;

  void setUserFeature(String feature);

  bool hasUserFeature(String feature) const;

 public:
  MultiMatrixImpl* impl();

  const MultiMatrixImpl* impl() const;

 private:
  std::shared_ptr<MultiMatrixImpl> m_impl;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
