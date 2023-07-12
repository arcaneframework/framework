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
class BlockMatrix;
class BlockVector;
struct Exporter;

class ALIEN_REFSEMANTIC_EXPORT SystemWriter
{
 public:
  SystemWriter(std::string const& filename, std::string format = "ascii",
               Arccore::MessagePassing::IMessagePassingMng* parallel_mng = nullptr);
  virtual ~SystemWriter();

  void dump(Matrix const& A);
  void dump(Matrix const& A, Vector const& rhs);
  void dump(Matrix const& A, Vector const& rhs, Vector const& sol,
            const SolutionInfo& sol_info);
  void dump(BlockMatrix const& A, BlockVector const& rhs);
  void dump(BlockMatrix const& A, BlockVector const& rhs, BlockVector const& sol,
            const SolutionInfo& sol_info);

 private:
  template <typename FileNodeT>
  void _writeMatrixInfo(Exporter& exporter, FileNodeT& parent_node, int nrows, int ncols,
                        int nnz, int blk_size, int blk_size2);

  template <typename FileNodeT>
  void _writeCSRProfile(Exporter& exporter, FileNodeT& parent_node, int nrows, int nnz,
                        int const* kcol, int const* cols);

  template <typename FileNodeT>
  void _writeMatrixValues(Exporter& exporter, FileNodeT& parent_node, int nnz,
                          int blk_size, int blk_size2, double const* values);

  template <typename FileNodeT>
  void _writeVector(Exporter& exporter, FileNodeT& vector_node, int nrows, int blk_size,
                    double const* values);

  template <typename FileNodeT>
  void _writeSolutionInfo(
  Exporter& exporter, FileNodeT& parent_node, const SolutionInfo& sol_info);

  template <typename FileNodeT>
  void _beginDump(Exporter*& exporter, FileNodeT& base_node);

  template <typename FileNodeT>
  void _endDump(Exporter* exporter, FileNodeT& base_node);

  std::string m_filename;
  std::string m_format;
  int m_prec;
  Arccore::Integer m_rank, m_nproc;
  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng;

  std::map<SolutionInfo::eConvCrit, std::string> m_conv_crit_to_str;
};

} // namespace Alien
