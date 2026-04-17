// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleMatrixPartitioner.h                                   (C) 2000-2026 */
/*                                                                           */
/* Simple matrix partitioner merging consecutive domains together.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_SIMPLEMATRIXPARTITIONER_H
#define ARCCORE_ALINA_SIMPLEMATRIXPARTITIONER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * This file is based on the work on AMGCL library (version march 2026)
 * which can be found at https://github.com/ddemidov/amgcl.
 *
 * Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>
 * SPDX-License-Identifier: MIT
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <memory>

#include "arccore/alina/BackendInterface.h"
#include "arccore/alina/MessagePassingUtils.h"
#include "arccore/alina/DistributedMatrix.h"
#include "arccore/alina/MatrixPartitionUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Simple matrix partitioner merging consecutive domains together.
 */
template <class Backend>
struct SimpleMatrixPartitioner
{
  typedef typename Backend::value_type value_type;
  typedef DistributedMatrix<Backend> matrix;

  struct params
  {
    bool enable = false;
    int min_per_proc = 10000;
    int shrink_ratio = 8;

    params() = default;

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, enable)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, min_per_proc)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, shrink_ratio)
    {
      p.check_params({ "enable", "min_per_proc", "shrink_ratio" });
    }

    void get(PropertyTree& p, const std::string& path = "") const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, enable);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, min_per_proc);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, shrink_ratio);
    }

  } prm;

  explicit SimpleMatrixPartitioner(const params& prm = params())
  : prm(prm)
  {}

  bool is_needed(const matrix& A) const
  {
    if (!prm.enable)
      return false;

    mpi_communicator comm = A.comm();
    ptrdiff_t n = A.loc_rows();
    std::vector<ptrdiff_t> row_dom = comm.exclusive_sum(n);

    int non_empty = 0;
    ptrdiff_t min_n = std::numeric_limits<ptrdiff_t>::max();
    for (int i = 0; i < comm.size; ++i) {
      ptrdiff_t m = row_dom[i + 1] - row_dom[i];
      if (m) {
        min_n = std::min(min_n, m);
        ++non_empty;
      }
    }

    return (non_empty > 1) && (min_n <= prm.min_per_proc);
  }

  std::shared_ptr<matrix> operator()(const matrix& A, unsigned /*block_size*/ = 1) const
  {
    mpi_communicator comm = A.comm();
    ptrdiff_t nrows = A.loc_rows();

    std::vector<ptrdiff_t> row_dom = comm.exclusive_sum(nrows);
    std::vector<ptrdiff_t> col_dom(comm.size + 1);

    for (int i = 0; i <= comm.size; ++i)
      col_dom[i] = row_dom[std::min<int>(i * prm.shrink_ratio, comm.size)];

    int old_domains = 0;
    int new_domains = 0;

    for (int i = 0; i < comm.size; ++i) {
      if (row_dom[i + 1] > row_dom[i])
        ++old_domains;
      if (col_dom[i + 1] > col_dom[i])
        ++new_domains;
    }

    if (comm.rank == 0)
      std::cout << "Partitioning[Simple] " << old_domains << " -> " << new_domains << std::endl;

    ptrdiff_t row_beg = row_dom[comm.rank];
    ptrdiff_t col_beg = col_dom[comm.rank];
    ptrdiff_t col_end = col_dom[comm.rank + 1];

    std::vector<ptrdiff_t> perm(nrows);
    for (ptrdiff_t i = 0; i < nrows; ++i) {
      perm[i] = i + row_beg;
    }

    return mpi_graph_perm_matrix<Backend>(comm, col_beg, col_end, perm);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
