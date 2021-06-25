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

#include <alien/core/block/Block.h>
#include <alien/data/IVector.h>
#include <alien/data/Space.h>
#include <alien/distribution/VectorDistribution.h>
#include <alien/ref/AlienRefSemanticPrecomp.h>
#include <cstdlib>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_REFSEMANTIC_EXPORT BlockVector final : public IVector
{
 public:
  BlockVector();

  BlockVector(const Block& block, const VectorDistribution& dist);

  BlockVector(Integer nrows, Integer nrows_local, const Block& block,
              IMessagePassingMng* parallel_mng);

  BlockVector(Integer nrows, const Block& block, IMessagePassingMng* parallel_mng);

  BlockVector(BlockVector&& vector);

  ~BlockVector() override = default;

  BlockVector& operator=(BlockVector&& vector);

  BlockVector(const BlockVector&) = delete;
  BlockVector& operator=(const BlockVector&) = delete;

 public:
  void init(const Block& block, const VectorDistribution& dist);

  void free();

  void clear();

  // Pour les expressions
  void visit(ICopyOnWriteVector&) const override;

  [[nodiscard]] const VectorDistribution& distribution() const;

  [[nodiscard]] const ISpace& space() const override;

  void setUserFeature(String feature);

  [[nodiscard]] bool hasUserFeature(Arccore::String feature) const;

  [[nodiscard]] const Block& block() const;

 public:
  MultiVectorImpl* impl() override;

  [[nodiscard]] const MultiVectorImpl* impl() const override;

 private:
  std::shared_ptr<MultiVectorImpl> m_impl;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
