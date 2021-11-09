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

#include <memory>

#include <alien/ref/AlienRefSemanticPrecomp.h>

#include <alien/core/block/Block.h>
#include <alien/data/IMatrix.h>
#include <alien/data/Space.h>
#include <alien/distribution/MatrixDistribution.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_REFSEMANTIC_EXPORT BlockMatrix final : public IMatrix
{
 public:
  BlockMatrix();

  BlockMatrix(const Block& block, const MatrixDistribution& dist);

  BlockMatrix(Integer nrows, Integer ncols, Integer nrows_local, const Block& block,
              IMessagePassingMng* parallel_mng);

  BlockMatrix(
  Integer nrows, Integer ncols, const Block& block, IMessagePassingMng* parallel_mng);

  BlockMatrix(BlockMatrix&& m);

  ~BlockMatrix() {}

  BlockMatrix& operator=(BlockMatrix&& m);

  BlockMatrix(const BlockMatrix&) = delete;
  BlockMatrix& operator=(const BlockMatrix&) = delete;

 public:
  void init(const Block& block, const MatrixDistribution& dist);

  void free();

  void clear();

  // Pour les expressions
  void visit(ICopyOnWriteMatrix&) const;

  const MatrixDistribution& distribution() const;

  const ISpace& rowSpace() const;

  const ISpace& colSpace() const;

  void setUserFeature(String feature);

  bool hasUserFeature(String feature) const;

  const Block& block() const;

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
