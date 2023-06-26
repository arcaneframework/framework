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

#ifndef ALIEN_DOKDIRECTMATRIXBUILDER_H
#define ALIEN_DOKDIRECTMATRIXBUILDER_H

#include <optional>

#include <alien/move/AlienMoveSemanticExport.h>
#include <alien/utils/Precomp.h>

#include <alien/move/data/MatrixData.h>

#include <alien/kernels/dok/BaseDoKDirectMatrixBuilder.h>

namespace Alien::Move
{
class ALIEN_MOVESEMANTIC_EXPORT DoKDirectMatrixBuilder
{
 public:
  explicit DoKDirectMatrixBuilder(MatrixData&& self)
  : m_data(std::move(self))
  {
    m_builder = std::make_unique<Alien::Common::BaseDoKDirectMatrixBuilder>(Alien::Common::BaseDoKDirectMatrixBuilder(m_data));
  }
  virtual ~DoKDirectMatrixBuilder() = default;

  DoKDirectMatrixBuilder(const DoKDirectMatrixBuilder&) = delete;
  DoKDirectMatrixBuilder& operator=(const DoKDirectMatrixBuilder&) = delete;
  DoKDirectMatrixBuilder(DoKDirectMatrixBuilder&&) = delete;
  DoKDirectMatrixBuilder& operator=(DoKDirectMatrixBuilder&&) = delete;

  std::optional<Arccore::Real> contribute(Arccore::Integer row, Arccore::Integer col, Arccore::Real value)
  {
    if (!m_builder) {
      return std::nullopt;
    }

    return m_builder->contribute(row, col, value);
  }

  MatrixData&& release()
  {
    m_builder->assemble();
    m_builder.reset(nullptr);
    return std::move(m_data);
  }

 private:
  MatrixData m_data;
  std::unique_ptr<Alien::Common::BaseDoKDirectMatrixBuilder> m_builder;
};
} // namespace Alien::Move
#endif //ALIEN_DOKDIRECTMATRIXBUILDER_H
