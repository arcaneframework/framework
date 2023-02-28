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

#include <alien/ref/import_export/MatrixMarketSystemReader.h>
#include <alien/ref/import_export/Reader.h>

namespace Alien
{

MatrixMarketSystemReader::MatrixMarketSystemReader(std::string const& filename)
: m_filename(filename)
{
}

MatrixMarketSystemReader::~MatrixMarketSystemReader() = default;

void MatrixMarketSystemReader::read(Matrix& A)
{
  std::fstream file_stream(m_filename, std::ios::in);

  if (!file_stream.good()) {
    throw FatalErrorException(__PRETTY_FUNCTION__);
  }

  FStreamReader reader(&file_stream);
  loadMMMatrixFromReader<FStreamReader>(A, reader);
}

void MatrixMarketSystemReader::read(Vector& rhs)
{
  std::fstream file_stream(m_filename, std::ios::in);

  if (!file_stream.good()) {
    throw FatalErrorException(__PRETTY_FUNCTION__);
  }

  FStreamReader reader(&file_stream);
  loadMMRhsFromReader<FStreamReader>(rhs, reader);
}

} /* namespace Alien */
