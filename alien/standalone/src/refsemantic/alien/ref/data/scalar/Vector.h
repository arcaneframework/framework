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

#include <alien/distribution/VectorDistribution.h>
#include <alien/ref/AlienRefSemanticPrecomp.h>

#include <alien/data/IVector.h>
#include <alien/data/Space.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_REFSEMANTIC_EXPORT Vector final : public IVector
{
 public:
  Vector();

  Vector(const VectorDistribution& dist);

  Vector(Integer nrows, Integer nrows_local, IMessagePassingMng* parallel_mng);

  Vector(Arccore::Integer nrows, IMessagePassingMng* parallel_mng);

  Vector(Vector&& vector);

  ~Vector() {}

  Vector& operator=(Vector&& vector);

  Vector(const Vector&) = delete;
  Vector& operator=(const Vector&) = delete;

  template <typename E>
  Vector& operator=(const E&);

 public:
  // Pour les expressions
  void visit(ICopyOnWriteVector&) const;

  const VectorDistribution& distribution() const;

  const ISpace& space() const;

  void setUserFeature(String feature);

  bool hasUserFeature(String feature) const;

 public:
  MultiVectorImpl* impl();

  const MultiVectorImpl* impl() const;

 private:
  std::shared_ptr<MultiVectorImpl> m_impl;
#ifdef DEBUG
  std::string m_name = "UnamedVector";

 public:
  void setName(std::string name) { m_name = name; }
  std::string const& name() const { return m_name; }
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
