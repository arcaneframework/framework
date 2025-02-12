// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include <arccore/base/ArccoreGlobal.h>
#include <arccore/message_passing/MessagePassingGlobal.h>

#include <alien/ref/AlienRefExport.h>
#include <alien/ref/AlienRefSemanticPrecomp.h>

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
