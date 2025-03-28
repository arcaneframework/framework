// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <alien/data/IMatrix.h>

#include <alien/ref/data/block/BlockMatrix.h>
#include <alien/ref/data/scalar/Matrix.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Scalar>
class SimpleCSRMatrix;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ValueT = Real>
class StreamMatrixBuilderT
{
 public:
  class Inserter;
  class BaseInserter;
  class Profiler;
  class Filler;

  typedef Alien::Matrix MatrixType;
  using Matrix = Alien::Matrix;
  using BlockMatrix = Alien::BlockMatrix;

  enum eColOrdering
  {
    eUndef,
    eOwnAndGhost,
    eFull
  };

 public:
  /** Constructeur de la classe */
  StreamMatrixBuilderT(Matrix& matrix, bool init_and_start = true);
  StreamMatrixBuilderT(BlockMatrix& matrix, bool init_and_start = true);
  StreamMatrixBuilderT(IMatrix& matrix, bool init_and_start = true);

  /** Destructeur de la classe */
  virtual ~StreamMatrixBuilderT();

  void setOrderRowColsOpt(bool value) { m_order_row_cols_opt = value; }

  void setTraceMng(ITraceMng* trace_mng)
  {
    m_trace = trace_mng;
    if (m_matrix_impl)
      m_matrix_impl->setTraceMng(m_trace);
  }

 public:
  void init();

  void start();

  Inserter& getNewInserter();

  Inserter& getInserter(Integer id);

  void allocate(); // allocate includes fillZero ?

  void fillZero();

  void finalize();

  void end();

 private:
  UniqueArray<Inserter*> m_inserters;

 private:
  void _freeInserters();
  void computeProfile();

 protected:
  IMatrix& m_matrix;
  SimpleCSRMatrix<ValueT>* m_matrix_impl;

  Integer m_local_size = 0;
  Integer m_global_size = 0;
  Integer m_local_offset = 0;

  UniqueArray<Integer> m_ghost_row_size;
  Integer m_ghost_size = 0;
  UniqueArray<Integer> m_offset;
  Integer m_matrix_size = 0;
  UniqueArray<Integer> m_row_size;
  Integer m_myrank = 0;
  Integer m_nproc = 1;

  UniqueArray<Integer> m_ordered_idx;
  eColOrdering m_col_ordering;
  bool m_order_row_cols_opt = false;

  IMessagePassingMng* m_parallel_mng = nullptr;
  ITraceMng* m_trace = nullptr;

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
/*---------------------------------------------------------------------------*/

typedef StreamMatrixBuilderT<double> StreamMatrixBuilder;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "StreamMatrixBuilderInserter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "StreamMatrixBuilderInserterT.h"
#include "StreamMatrixBuilderT.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
