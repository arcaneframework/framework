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

#include <string>
#include <vector>

#include <arccore/base/ArccoreGlobal.h>

#include <alien/ref/AlienRefExport.h>
#include <alien/ref/AlienRefSemanticPrecomp.h>

namespace Arccore::MessagePassing
{
class IMessagePassingMng;
}

namespace Alien
{

class Matrix;
class Vector;

class BlockMatrix;
class BlockVector;

struct Importer;

class ALIEN_REFSEMANTIC_EXPORT SystemReader
{
 public:
  SystemReader(std::string const& filename, std::string format = "ascii",
               Arccore::MessagePassing::IMessagePassingMng* parallel_mng = nullptr);
  virtual ~SystemReader();

  void read(Matrix& A);
  /*
      void read(Vector & x) ;
  */
  void read(BlockMatrix& A);
  /*
      void read(BlockVector & x) ;
  */
 private:
  template <typename FileNodeT>
  void _readMatrixInfo(Importer& importer, FileNodeT& info_node, int& nrows, int& ncols,
                       int& nnz, int& blk_size, int& blk_size2);

  template <typename FileNodeT>
  void _readCSRProfile(Importer& importer, FileNodeT& parent_node, int& nrows, int& nnz,
                       std::vector<int>& kcol, std::vector<int>& cols);

  template <typename FileNodeT>
  void _readMatrixValues(Importer& importer, FileNodeT& parent_node, int& size,
                         int& blk_size, int& blk_size2, std::vector<double>& values);

  std::string m_filename;
  std::string m_format;
  int m_prec;
  Arccore::Integer m_rank, m_nproc;
  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng;
};

} // namespace Alien
