// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParmetisMatrixPartitioner.h                                 (C) 2000-2026 */
/*                                                                           */
/* Matrix partitioning using ParMetis.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_PARMETISMATRIXPARTITIONER_H
#define ARCCORE_ALINA_PARMETISMATRIXPARTITIONER_H
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
#include "arccore/alina/ValueTypeInterface.h"
#include "arccore/alina/MessagePassingUtils.h"
#include "arccore/alina/DistributedMatrix.h"
#include "arccore/alina/MatrixPartitionUtils.h"

#include <parmetis.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend>
struct ParmetisMatrixPartitioner
{
  typedef typename Backend::value_type value_type;
  typedef DistributedMatrix<Backend> matrix;
  using col_type = Backend::col_type;
  using ptr_type = Backend::ptr_type;

  struct params
  {
    bool shrink;
    int min_per_proc;
    int shrink_ratio;

    params()
    : shrink(false)
    , min_per_proc(10000)
    , shrink_ratio(8)
    {}

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, shrink)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, min_per_proc)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, shrink_ratio)
    {
      p.check_params({ "shrink", "min_per_proc", "shrink_ratio" });
    }

    void get(PropertyTree& p, const std::string& path = "") const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, shrink);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, min_per_proc);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, shrink_ratio);
    }

  } prm;

  explicit ParmetisMatrixPartitioner(const params& prm = params())
  : prm(prm)
  {}

  bool is_needed(const matrix& A) const
  {
    if (!prm.shrink)
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

  std::shared_ptr<matrix> operator()(const matrix& A, unsigned block_size = 1) const
  {
    mpi_communicator comm = A.comm();
    idx_t n = A.loc_rows();
    ptrdiff_t row_beg = A.loc_col_shift();

    // Partition the graph.
    int active = (n > 0);
    int active_ranks = comm.reduceSum(active);
    int shrink = prm.shrink ? prm.shrink_ratio : 1;

    idx_t npart = std::max(1, active_ranks / shrink);

    if (comm.rank == 0)
      std::cout << "Partitioning[ParMETIS] " << active_ranks << " -> " << npart << std::endl;

    std::vector<ptrdiff_t> perm(n);
    ptrdiff_t col_beg, col_end;

    if (npart == 1) {
      col_beg = (comm.rank == 0) ? 0 : A.glob_rows();
      col_end = A.glob_rows();

      for (ptrdiff_t i = 0; i < n; ++i) {
        perm[i] = row_beg + i;
      }
    }
    else {
      if (block_size == 1) {
        std::tie(col_beg, col_end) = partition(A, npart, perm);
      }
      else {
        typedef typename math::scalar_of<value_type>::type scalar;
        using sbackend = BuiltinBackend<scalar,col_type,ptr_type>;
        ptrdiff_t np = n / block_size;

        DistributedMatrix<sbackend> A_pw(A.comm(),
                                         pointwise_matrix(*A.local(), block_size),
                                         pointwise_matrix(*A.remote(), block_size));

        std::vector<ptrdiff_t> perm_pw(np);

        std::tie(col_beg, col_end) = partition(A_pw, npart, perm_pw);

        col_beg *= block_size;
        col_end *= block_size;

        for (ptrdiff_t ip = 0; ip < np; ++ip) {
          ptrdiff_t i = ip * block_size;
          ptrdiff_t j = perm_pw[ip] * block_size;

          for (unsigned k = 0; k < block_size; ++k)
            perm[i + k] = j + k;
        }
      }
    }

    return mpi_graph_perm_matrix<Backend>(comm, col_beg, col_end, perm);
  }

  template <class B>
  std::tuple<ptrdiff_t, ptrdiff_t>
  partition(const DistributedMatrix<B>& A, idx_t npart, std::vector<ptrdiff_t>& perm) const
  {
    mpi_communicator comm = A.comm();
    idx_t n = A.loc_rows();
    int active = (n > 0);

    std::vector<idx_t> ptr;
    std::vector<idx_t> col;

    mpi_symm_graph(A, ptr, col);

    idx_t wgtflag = 0;
    idx_t numflag = 0;
    idx_t options = 0;
    idx_t edgecut = 0;
    idx_t ncon = 1;

    std::vector<real_t> tpwgts(npart, 1.0 / npart);
    std::vector<real_t> ubvec(ncon, 1.05);
    std::vector<idx_t> part(n);
    if (!n)
      part.reserve(1); // So that part.data() is not NULL

    MPI_Comm scomm;
    MPI_Comm_split(comm, active ? 0 : MPI_UNDEFINED, comm.rank, &scomm);

    if (active) {
      mpi_communicator sc(scomm);
      std::vector<idx_t> vtxdist = sc.exclusive_sum(n);

      sc.check(
      METIS_OK == ParMETIS_V3_PartKway(&vtxdist[0], &ptr[0], &col[0], NULL, NULL, &wgtflag, &numflag, &ncon, &npart, &tpwgts[0], &ubvec[0], &options, &edgecut, &part[0], &scomm),
      "Error in ParMETIS");

      MPI_Comm_free(&scomm);
    }

    return mpi_graph_perm_index(comm, npart, part, perm);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
