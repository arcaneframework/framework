/*
 * Copyright 2021 IFPEN-CEA
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
 *  SPDX-License-Identifier: Apache-2.0
 */

#include "BaseDoKDirectMatrixBuilder.h"

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/data/IMatrix.h>

#include "DoKBackEnd.h"
#include "DoKMatrixT.h"

namespace Alien
{

Common::BaseDoKDirectMatrixBuilder::BaseDoKDirectMatrixBuilder(Alien::IMatrix& self)
: m_matrix(self)
{
  m_matrix.impl()->lock();
  m_impl = &m_matrix.impl()->get<BackEnd::tag::DoK>(true);
}

Common::BaseDoKDirectMatrixBuilder::~BaseDoKDirectMatrixBuilder()
{
  assemble();
}

std::optional<Real> Common::BaseDoKDirectMatrixBuilder::contribute(Arccore::Integer row, Arccore::Integer col, Arccore::Real value)
{
  return std::optional<Real>(m_impl->addNNZ(row, col, value));
}

std::optional<Real> Common::BaseDoKDirectMatrixBuilder::setNNZ(Arccore::Integer row, Arccore::Integer col, Arccore::Real value)
{
  m_impl->setNNZ(row, col, value);
  return std::optional<Real>(value);
}

bool Common::BaseDoKDirectMatrixBuilder::assemble()
{
  m_impl->assemble();
  m_matrix.impl()->unlock();
  return true;
}
} // namespace Alien
