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

#include <map>

#include "SystemInfo.h"
#include <alien/ref/AlienRefSemanticPrecomp.h>
#include <arccore/base/ArccoreGlobal.h>
#include <arccore/message_passing/IMessagePassingMng.h>

namespace Alien
{
class Matrix;
class Vector;
struct Exporter;

class ALIEN_REFSEMANTIC_EXPORT MatrixMarketSystemWriter
{
 public:
  MatrixMarketSystemWriter(std::string const& filename, Arccore::MessagePassing::IMessagePassingMng* parallel_mng = nullptr);
  virtual ~MatrixMarketSystemWriter();

  void dump(Matrix const& A, std::string const& description);
  void dump(Vector const& rhs, std::string const& description);

 private:
  std::string m_filename;
  Arccore::Integer m_rank, m_nproc;
  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng = nullptr;
};

} // namespace Alien
