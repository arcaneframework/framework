// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatrixPartitionUtils.h                                      (C) 2000-2026 */
/*                                                                           */
/* Utils for matrix repartitioning.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_MATRIXPARTITIONUTILS_H
#define ARCCORE_ALINA_MATRIXPARTITIONUTILS_H
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

#include <vector>
#include <algorithm>
#include <numeric>

#include <tuple>

#include "arccore/alina/BackendInterface.h"
#include "arccore/alina/DistributedMatrix.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend, class Ptr, class Col> void
mpi_symm_graph(const DistributedMatrix<Backend>& A,
               std::vector<Ptr>& ptr, std::vector<Col>& col)
{
  using build_matrix = Backend::matrix;

  ARCCORE_ALINA_TIC("symm graph");

  build_matrix& A_loc = *A.local();
  build_matrix& A_rem = *A.remote();

  ptrdiff_t n = A_loc.nbRow();
  ptrdiff_t row_beg = A.loc_col_shift();

  auto T = transpose(A);

  build_matrix& T_loc = *T->local();
  build_matrix& T_rem = *T->remote();

  // Build symmetric graph
  ptr.resize(n + 1, 0);

  arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
    for (ptrdiff_t i = begin; i < (begin + size); ++i) {
      using Alina::detail::sort_row;

      ptrdiff_t A_loc_beg = A_loc.ptr[i];
      ptrdiff_t A_loc_end = A_loc.ptr[i + 1];

      ptrdiff_t A_rem_beg = A_rem.ptr[i];
      ptrdiff_t A_rem_end = A_rem.ptr[i + 1];

      ptrdiff_t T_loc_beg = T_loc.ptr[i];
      ptrdiff_t T_loc_end = T_loc.ptr[i + 1];

      ptrdiff_t T_rem_beg = T_rem.ptr[i];
      ptrdiff_t T_rem_end = T_rem.ptr[i + 1];

      sort_row(A_loc.col + A_loc_beg, A_loc.val + A_loc_beg, A_loc_end - A_loc_beg);
      sort_row(A_rem.col + A_rem_beg, A_rem.val + A_rem_beg, A_rem_end - A_rem_beg);

      sort_row(T_loc.col + T_loc_beg, T_loc.val + T_loc_beg, T_loc_end - T_loc_beg);
      sort_row(T_rem.col + T_rem_beg, T_rem.val + T_rem_beg, T_rem_end - T_rem_beg);

      Ptr row_width = 0;

      for (ptrdiff_t ja = A_loc_beg, jt = T_loc_beg; ja < A_loc_end || jt < T_loc_end;) {
        ptrdiff_t c;
        if (ja == A_loc_end) {
          c = T_loc.col[jt];
          ++jt;
        }
        else if (jt == T_loc_end) {
          c = A_loc.col[ja];
          ++ja;
        }
        else {
          ptrdiff_t ca = A_loc.col[ja];
          ptrdiff_t ct = T_loc.col[jt];
          if (ca < ct) {
            c = ca;
            ++ja;
          }
          else if (ca == ct) {
            c = ca;
            ++ja;
            ++jt;
          }
          else {
            c = ct;
            ++jt;
          }
        }

        if (c != i)
          ++row_width;
      }

      for (ptrdiff_t ja = A_rem_beg, jt = T_rem_beg; ja < A_rem_end || jt < T_rem_end;) {
        if (ja == A_rem_end) {
          ++jt;
        }
        else if (jt == T_rem_end) {
          ++ja;
        }
        else {
          ptrdiff_t ca = A_rem.col[ja];
          ptrdiff_t ct = T_rem.col[jt];
          if (ca < ct) {
            ++ja;
          }
          else if (ca == ct) {
            ++ja;
            ++jt;
          }
          else {
            ++jt;
          }
        }

        ++row_width;
      }

      ptr[i + 1] = row_width;
    }
  });

  std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());

  col.resize(ptr.back());
  if (col.empty())
    col.reserve(1); // So that col.data() is not NULL

  arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
    for (ptrdiff_t i = begin; i < (begin + size); ++i) {
      ptrdiff_t A_loc_beg = A_loc.ptr[i];
      ptrdiff_t A_loc_end = A_loc.ptr[i + 1];

      ptrdiff_t A_rem_beg = A_rem.ptr[i];
      ptrdiff_t A_rem_end = A_rem.ptr[i + 1];

      ptrdiff_t T_loc_beg = T_loc.ptr[i];
      ptrdiff_t T_loc_end = T_loc.ptr[i + 1];

      ptrdiff_t T_rem_beg = T_rem.ptr[i];
      ptrdiff_t T_rem_end = T_rem.ptr[i + 1];

      Ptr head = ptr[i];

      for (ptrdiff_t ja = A_loc_beg, jt = T_loc_beg; ja < A_loc_end || jt < T_loc_end;) {
        ptrdiff_t c;
        if (ja == A_loc_end) {
          c = T_loc.col[jt];
          ++jt;
        }
        else if (jt == T_loc_end) {
          c = A_loc.col[ja];
          ++ja;
        }
        else {
          ptrdiff_t ca = A_loc.col[ja];
          ptrdiff_t ct = T_loc.col[jt];

          if (ca < ct) {
            c = ca;
            ++ja;
          }
          else if (ca == ct) {
            c = ca;
            ++ja;
            ++jt;
          }
          else {
            c = ct;
            ++jt;
          }
        }
        if (c != i)
          col[head++] = c + row_beg;
      }

      for (ptrdiff_t ja = A_rem_beg, jt = T_rem_beg; ja < A_rem_end || jt < T_rem_end;) {
        if (ja == A_rem_end) {
          col[head] = T_rem.col[jt];
          ++jt;
        }
        else if (jt == T_rem_end) {
          col[head] = A_rem.col[ja];
          ++ja;
        }
        else {
          ptrdiff_t ca = A_rem.col[ja];
          ptrdiff_t ct = T_rem.col[jt];

          if (ca < ct) {
            col[head] = ca;
            ++ja;
          }
          else if (ca == ct) {
            col[head] = ca;
            ++ja;
            ++jt;
          }
          else {
            col[head] = ct;
            ++jt;
          }
        }
        ++head;
      }
    }
  });

  ARCCORE_ALINA_TOC("symm graph");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Idx> std::tuple<ptrdiff_t, ptrdiff_t>
mpi_graph_perm_index(mpi_communicator comm, int npart, const std::vector<Idx>& part,
                     std::vector<ptrdiff_t>& perm)
{
  ARCCORE_ALINA_TIC("perm index");
  ptrdiff_t n = part.size();
  perm.resize(n);

  std::vector<ptrdiff_t> loc_part_cnt(npart, 0);
  std::vector<ptrdiff_t> loc_part_beg(npart, 0);
  std::vector<ptrdiff_t> glo_part_cnt(npart);
  std::vector<ptrdiff_t> glo_part_beg(npart + 1);

  for (Idx p : part)
    ++loc_part_cnt[p];

  MPI_Exscan(&loc_part_cnt[0], &loc_part_beg[0], npart, mpi_datatype<ptrdiff_t>(), MPI_SUM, comm);
  MPI_Allreduce(&loc_part_cnt[0], &glo_part_cnt[0], npart, mpi_datatype<ptrdiff_t>(), MPI_SUM, comm);

  glo_part_beg[0] = 0;
  std::partial_sum(glo_part_cnt.begin(), glo_part_cnt.end(), glo_part_beg.begin() + 1);

  std::vector<ptrdiff_t> cnt(npart, 0);
  for (ptrdiff_t i = 0; i < n; ++i) {
    Idx p = part[i];
    perm[i] = glo_part_beg[p] + loc_part_beg[p] + cnt[p]++;
  }

  ARCCORE_ALINA_TOC("perm index");
  return std::make_tuple(
  glo_part_beg[std::min(npart, comm.rank)],
  glo_part_beg[std::min(npart, comm.rank + 1)]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend, class Idx>
std::shared_ptr<DistributedMatrix<Backend>>
mpi_graph_perm_matrix(mpi_communicator comm, ptrdiff_t col_beg, ptrdiff_t col_end,
                      const std::vector<Idx>& perm)
{
  typedef typename Backend::value_type value_type;
  using build_matrix = Backend::matrix;

  ARCCORE_ALINA_TIC("perm matrix");

  ptrdiff_t n = perm.size();
  ptrdiff_t ncols = col_end - col_beg;

  auto i_loc = std::make_shared<build_matrix>();
  auto i_rem = std::make_shared<build_matrix>();

  build_matrix& I_loc = *i_loc;
  build_matrix& I_rem = *i_rem;

  I_loc.set_size(n, ncols, false);
  I_rem.set_size(n, 0, false);

  I_loc.ptr[0] = 0;
  I_rem.ptr[0] = 0;

  arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
    for (ptrdiff_t i = begin; i < (begin + size); ++i) {
      ptrdiff_t j = perm[i];

      if (col_beg <= j && j < col_end) {
        I_loc.ptr[i + 1] = 1;
        I_rem.ptr[i + 1] = 0;
      }
      else {
        I_loc.ptr[i + 1] = 0;
        I_rem.ptr[i + 1] = 1;
      }
    }
  });

  I_loc.set_nonzeros(I_loc.scan_row_sizes());
  I_rem.set_nonzeros(I_rem.scan_row_sizes());

  arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
    for (ptrdiff_t i = begin; i < (begin + size); ++i) {
      ptrdiff_t j = perm[i];

      if (col_beg <= j && j < col_end) {
        ptrdiff_t k = I_loc.ptr[i];
        I_loc.col[k] = j - col_beg;
        I_loc.val[k] = math::identity<value_type>();
      }
      else {
        ptrdiff_t k = I_rem.ptr[i];
        I_rem.col[k] = j;
        I_rem.val[k] = math::identity<value_type>();
      }
    }
  });

  ARCCORE_ALINA_TOC("perm matrix");
  return std::make_shared<DistributedMatrix<Backend>>(comm, i_loc, i_rem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
