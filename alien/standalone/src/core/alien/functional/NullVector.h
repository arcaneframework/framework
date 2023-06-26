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

#include <alien/data/IVector.h>
#include <alien/data/Space.h>
#include <alien/distribution/VectorDistribution.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NullVector final : public IVector
{
 public:
  NullVector();

  ~NullVector() {}

  NullVector(const NullVector&) = delete;
  NullVector(NullVector&&) = delete;
  NullVector& operator=(NullVector&& vector) = delete;
  NullVector& operator=(const NullVector&) = delete;

 public:
  const Space& space() const;

  const VectorDistribution& distribution() const;

 private:
  // Pour les expressions
  void visit(ICopyOnWriteVector&) const;

  MultiVectorImpl* impl();

  const MultiVectorImpl* impl() const;

 private:
  Space m_space;
  VectorDistribution m_distribution;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
