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

#include <arccore/base/ArccoreGlobal.h>
#include <string>

#include <alien/import_export/Reader.h>

namespace Alien
{

class ALIEN_EXPORT MatrixMarketSystemReader
{
 public:
  MatrixMarketSystemReader() = delete;
  MatrixMarketSystemReader(MatrixMarketSystemReader const&) = delete;
  MatrixMarketSystemReader& operator=(MatrixMarketSystemReader const&) = delete;

  explicit MatrixMarketSystemReader(std::string const& filename);
  ~MatrixMarketSystemReader();

  template <typename MatrixT>
  void readMatrix(MatrixT& A);

  template <typename VectorT>
  void readVector(VectorT& rhs);

 private:
  std::string m_filename;
};

template <typename MatrixT>
void MatrixMarketSystemReader::readMatrix(MatrixT& A)
{
  std::fstream file_stream(m_filename, std::ios::in);

  if (!file_stream.good()) {
    throw FatalErrorException(__PRETTY_FUNCTION__);
  }

  FStreamReader reader(&file_stream);
  loadMMMatrixFromReader<MatrixT, FStreamReader>(A, reader);
}

template <typename VectorT>
void MatrixMarketSystemReader::readVector(VectorT& rhs)
{
  std::fstream file_stream(m_filename, std::ios::in);

  if (!file_stream.good()) {
    throw FatalErrorException(__PRETTY_FUNCTION__);
  }

  FStreamReader reader(&file_stream);
  loadMMRhsFromReader<VectorT, FStreamReader>(rhs, reader);
}

} // namespace Alien
