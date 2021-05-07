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

#include <alien/utils/Precomp.h>

#include <alien/ref/data/block/VBlockMatrix.h>

/*---------------------------------------------------------------------------*/

namespace Alien
{

class IBlockBuilder;
template <typename Scalar>
class SimpleCSRMatrix;

/*---------------------------------------------------------------------------*/

template <typename ValueT = Real>
class StreamVBlockMatrixBuilderT
{
 public:
  class Inserter;
  class BaseInserter;
  class Profiler;
  class Filler;

 public:
  /** Constructeur de la classe */
  StreamVBlockMatrixBuilderT(VBlockMatrix& matrix, bool init_and_start = true);

  /** Destructeur de la classe */
  virtual ~StreamVBlockMatrixBuilderT();

 public:
  void init();

  void start();
  void finalize();
  void end();

  Inserter& getNewInserter();

  Inserter& getInserter(Integer id);

  void allocate();

  void fillZero();

  const VBlock* vblock() const;

 private:
  UniqueArray<Inserter*> m_inserters;

 private:
  void _freeInserters();
  void computeProfile();

 protected:
  IMatrix& m_matrix;
  SimpleCSRMatrix<ValueT>* m_matrix_impl;

  Integer m_local_size;
  Integer m_global_size;
  Integer m_local_offset;

  UniqueArray<Integer> m_ghost_row_size;
  UniqueArray<Integer> m_block_ghost_row_size;
  Integer m_ghost_size;
  Integer m_block_ghost_size;
  UniqueArray<Integer> m_offset;
  Integer m_matrix_size;
  Integer m_block_matrix_size;
  UniqueArray<Integer> m_row_size;
  UniqueArray<Integer> m_block_row_size;
  Integer m_myrank, m_nproc;

  enum ColOrdering
  {
    eUndef,
    eOwnAndGhost,
    eFull
  };
  ColOrdering m_col_ordering;

  IMessagePassingMng* m_parallel_mng;
  ITraceMng* m_trace;

  enum State
  {
    eNone,
    eInit,
    ePrepared,
    eStart
  };
  State m_state;
};

/*---------------------------------------------------------------------------*/

typedef StreamVBlockMatrixBuilderT<double> StreamVBlockMatrixBuilder;

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/

#include "StreamVBlockMatrixBuilderInserter.h"
#include "StreamVBlockMatrixBuilderInserterT.h"
#include "StreamVBlockMatrixBuilderT.h"
