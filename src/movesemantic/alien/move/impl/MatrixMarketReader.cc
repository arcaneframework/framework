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
  void tolower(std::string& str)
  {
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  }

  class MatrixDescription
  {
   public:
    MatrixDescription() = default;

    explicit MatrixDescription(Arccore::Span<Arccore::Integer> src)
    {
      if (src.size() != 4) {
        throw Arccore::FatalErrorException("Matrix Descriptor", "Cannot deserialize array");
      }
      n_rows = src[0];
      n_cols = src[1];
      n_nnz = src[2];
      symmetric = (src[3] == 0);
    }

    Arccore::UniqueArray<Arccore::Integer> to_array() const
    {
      Arccore::UniqueArray<Arccore::Integer> array(4);
      array[0] = n_rows;
      array[1] = n_cols;
      array[2] = n_nnz;
      array[3] = symmetric ? 0 : 1;
      return array;
    }

    static constexpr size_t serializedSize()
    {
      return 4;
    }

    int n_rows{ 0 };
    int n_cols{ 0 };
    int n_nnz{ 0 };
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

  bool readValues(std::istream& fstream, DoKDirectMatrixBuilder& builder, bool symmetric)
  {
    std::string line;
    while (std::getline(fstream, line)) {

      if ('%' == line[0]) {
        continue;
      }

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
  if (pm->commRank() == 0) { // Only rank 0 read the file
    std::ifstream stream;
    std::array<char, 1024 * 1024> buf; // Buffer for reading
    stream.rdbuf()->pubsetbuf(buf.data(), buf.size());
    stream.open(filename);
    if (!stream) {
      throw Arccore::FatalErrorException("readFromMatrixMarket", "Unable to read file");
    }
    auto desc = readBanner(stream);
    if (!desc) {
      throw Arccore::FatalErrorException("readFromMatrixMarket", "Invalid header");
    }
    auto ser_desc = desc.value().to_array();
    Arccore::MessagePassing::mpBroadcast(pm, ser_desc, 0);
    DoKDirectMatrixBuilder builder(createMatrixData(desc.value(), pm));
    readValues(stream, builder, desc->symmetric);
    return builder.release();
  }
  else {
    // Receive description description from rank 0
    Arccore::UniqueArray<Arccore::Integer> ser_desc(MatrixDescription::serializedSize());
    Arccore::MessagePassing::mpBroadcast(pm, ser_desc, 0);
    MatrixDescription desc(ser_desc);
    DoKDirectMatrixBuilder builder(createMatrixData(desc, pm));
    return builder.release();
  }
}

VectorData ALIEN_MOVESEMANTIC_EXPORT
readFromMatrixMarket(const VectorDistribution& distribution, const std::string& filename)
{
  VectorData out(distribution);
  auto& v = out.impl()->template get<BackEnd::tag::DoK>(true);

  if (distribution.parallelMng()->commRank() == 0) { // Only rank 0 read the file
    std::ifstream stream;
    std::array<char, 1024 * 1024> buf; // Buffer for reading
    stream.rdbuf()->pubsetbuf(buf.data(), buf.size());
    stream.open(filename);
    if (!stream) {
      throw Arccore::FatalErrorException("readFromMatrixMarket", "Unable to read file");
    }
    std::string line;
    Arccore::Int32 row = 0;

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

      int rows = 0;
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
    while (std::getline(stream, line)) {
      double value;
      std::stringstream ss;
      ss << line;
      ss >> value;

      v.contribute(row, value);
      row++;
    }
  }
  v.assemble();

  return out;
}

} // namespace Alien::Move
