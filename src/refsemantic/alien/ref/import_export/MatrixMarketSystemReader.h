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

#include <alien/ref/AlienRefSemanticPrecomp.h>
#include <arccore/base/ArccoreGlobal.h>

#include <string>

namespace Alien
{
class Matrix;
class Vector;

class ALIEN_REFSEMANTIC_EXPORT MatrixMarketSystemReader
{
 public:
  MatrixMarketSystemReader() = delete;
  MatrixMarketSystemReader(MatrixMarketSystemReader const&) = delete;
  MatrixMarketSystemReader& operator=(MatrixMarketSystemReader const&) = delete;

  explicit MatrixMarketSystemReader(std::string const& filename);
  virtual ~MatrixMarketSystemReader();

  void read(Matrix& A);
  void read(Vector& rhs);

 private:
  std::string m_filename;
};

} // namespace Alien
