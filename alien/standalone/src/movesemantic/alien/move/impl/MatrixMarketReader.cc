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

#include <fstream>
#include <algorithm>
#include <cctype>

#include <arccore/base/FatalErrorException.h>
#include <arccore/message_passing/IMessagePassingMng.h>

#include <alien/data/Space.h>
#include <alien/distribution/MatrixDistribution.h>
#include <alien/distribution/VectorDistribution.h>

#include <alien/move/data/MatrixData.h>
#include <alien/move/handlers/scalar/DoKDirectMatrixBuilder.h>
#include <alien/move/data/VectorData.h>
#include <alien/kernels/dok/DoKVector.h>
#include <alien/kernels/dok/DoKBackEnd.h>

namespace Alien::Move
{

namespace
{
  std::pair<size_t, size_t> partition(size_t full_size, const Arccore::MessagePassing::IMessagePassingMng* pm)
  {
    auto my_rank = pm->commRank();
    auto comm_size = pm->commSize();
    size_t line_slice = full_size / comm_size;
    auto start = line_slice * my_rank;
    auto stop = line_slice * (my_rank + 1);
    if (my_rank == comm_size - 1) {
      stop = full_size;
    }
    return std::make_pair(start, stop);
  }

  void tolower(std::string& str)
  {
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  }

  class MatrixDescription
  {
   public:
    MatrixDescription() = default;

    int n_rows{ 0 };
    int n_cols{ 0 };
    size_t n_nnz{ 0 };
    bool symmetric{ true };
  };

  std::optional<MatrixDescription> readBanner(std::istream& fstream)
  {
    std::string line;

    MatrixDescription out;

    auto try_header = true;

    while (std::getline(fstream, line)) {
      std::stringstream ss;
      ss << line;

      if (try_header) {
        std::string matrix;
        std::string _; // junk
        std::string format; // (coordinate, array)
        std::string scalar; // (pattern, real, complex, integer)
        std::string symmetry; // (general, symmetric, skew-symmetric, hermitian)

        // get matrix kind
        ss >> matrix; // skip '%%MatrixMarket

        if ("%%MatrixMarket" != matrix) {
          continue;
        }

        ss >> _; // skip matrix
        ss >> format;
        ss >> scalar;
        ss >> symmetry;

        tolower(format);
        tolower(scalar);
        tolower(symmetry);

        if ("coordinate" != format) {
          return std::nullopt;
        }

        if ("real" != scalar) {
          return std::nullopt;
        }

        if ("general" == symmetry) {
          out.symmetric = false;
        }
        else {
          out.symmetric = true;
        }
        try_header = false;
      }
      else if ('%' == line[0]) {
        // skip comment
        continue;
      }
      else {
        //first line is matrix size, then done with banner
        ss >> out.n_rows;
        ss >> out.n_cols;
        ss >> out.n_nnz;
        break;
      }
    }
    return std::make_optional(out);
  }

  bool readValues(std::istream& fstream, DoKDirectMatrixBuilder& builder, bool symmetric, size_t start, size_t stop)
  {
    for (size_t i = 0; i < start; ++i) {
      fstream.ignore(4096, '\n');
    }

    std::string line;
    size_t count = start;
    while (count < stop && std::getline(fstream, line)) {
      if ('%' == line[0]) {
        continue;
      }

      count++;
      int row = 0;
      int col = 0;
      double value = 0.0;
      std::stringstream ss;
      ss << line;
      ss >> row >> col >> value;
      builder.contribute(row - 1, col - 1, value);
      if (symmetric && row != col) {
        builder.contribute(col - 1, row - 1, value);
      }
    }
    return true;
  }

  MatrixData createMatrixData(MatrixDescription desc, Arccore::MessagePassing::IMessagePassingMng* pm)
  {
    Alien::Space row_space(desc.n_rows, "RowSpace");
    Alien::Space col_space(desc.n_cols, "ColSpace");
    Alien::MatrixDistribution dist(
    row_space, col_space, pm);
    return Alien::Move::MatrixData(dist);
  }

} // namespace

MatrixData ALIEN_MOVESEMANTIC_EXPORT
readFromMatrixMarket(Arccore::MessagePassing::IMessagePassingMng* pm, const std::string& filename)
{
  std::ifstream stream;
  std::array<char, 1024 * 1024> buf; // Buffer for reading
  stream.rdbuf()->pubsetbuf(buf.data(), buf.size());
  stream.open(filename);
  if (!stream) {
    throw Arccore::FatalErrorException("readFromMatrixMarket", "Unable to matrix read file");
  }
  auto desc = readBanner(stream);
  if (!desc) {
    throw Arccore::FatalErrorException("readFromMatrixMarket", "Invalid header");
  }
  DoKDirectMatrixBuilder builder(createMatrixData(desc.value(), pm));

  auto [nnz_start, nnz_stop] = partition(desc.value().n_nnz, pm);
  readValues(stream, builder, desc->symmetric, nnz_start, nnz_stop);
  return builder.release();
}

VectorData ALIEN_MOVESEMANTIC_EXPORT
readFromMatrixMarket(const VectorDistribution& distribution, const std::string& filename)
{
  VectorData out(distribution);
  auto& v = out.impl()->template get<BackEnd::tag::DoK>(true);

  std::ifstream stream;
  std::array<char, 1024 * 1024> buf; // Buffer for reading
  stream.rdbuf()->pubsetbuf(buf.data(), buf.size());
  stream.open(filename);
  if (!stream) {
    throw Arccore::FatalErrorException("readFromMatrixMarket", "Unable to read vector file");
  }
  std::string line;
  size_t rows = 0;

  auto try_header = true;
  while (std::getline(stream, line)) {
    // get matrix kind
    std::stringstream ss;
    ss << line;

    if (try_header) {
      std::string matrix;
      std::string _; // junk
      std::string format; // (coordinate, array)
      std::string scalar; // (pattern, real, complex, integer)

      ss >> matrix; // skip '%%MatrixMarket

      if ("%%MatrixMarket" != matrix)
        continue;

      ss >> _; // skip matrix
      ss >> format;
      ss >> scalar;
      ss >> _;

      tolower(format);
      tolower(scalar);

      if ("array" != format || "real" != scalar) {
        throw Arccore::FatalErrorException("mtx vector must be in 'array' and 'real' formats.");
      }
      try_header = false;
    }
    if ('%' == line[0])
      continue;

    int vectors = 0;
    ss >> rows;
    ss >> vectors;
    if (vectors != 1) {
      throw Arccore::FatalErrorException("mtx vector reader does not support multiple vectors.");
    }
    if (distribution.globalSize() != rows) {
      throw Arccore::FatalErrorException("mtx vector is not of correct size");
    }
    break;
  }

  auto [row_start, row_stop] = partition(rows, distribution.parallelMng());

  Arccore::Int32 row = 0;
  for (row = 0; row < row_start; ++row) {
    stream.ignore(4096, '\n');
  }

  while (row < row_stop && std::getline(stream, line)) {
    double value;
    std::stringstream ss;
    ss << line;
    ss >> value;

    v.contribute(row, value);
    row++;
  }
  v.assemble();

  return out;
}

} // namespace Alien::Move
