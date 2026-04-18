// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DistributedCoarsening.h                                     (C) 2000-2026 */
/*                                                                           */
/* Distributed coarsening algorithms.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_MPI_DISTRIBUTEDCOARSENING_H
#define ARCCORE_ALINA_MPI_DISTRIBUTEDCOARSENING_H
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

#include <tuple>
#include <memory>
#include <numeric>
#include <cassert>

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/Coarsening.h"
#include "arccore/alina/MessagePassingUtils.h"
#include "arccore/alina/DistributedMatrix.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Distributed PMIS aggregation.
 */
template <class Backend>
struct DistributedPMISAggregation
{
  typedef typename Backend::value_type value_type;
  typedef typename math::scalar_of<value_type>::type scalar_type;
  typedef DistributedMatrix<Backend> matrix;
  typedef CommunicationPattern<Backend> CommPattern;
  using build_matrix = Backend::matrix;
  using col_type = Backend::col_type;
  using ptr_type = Backend::ptr_type;
  using bool_backend = BuiltinBackend<char,col_type,ptr_type>;
  using bool_matrix = bool_backend::matrix;

  struct params
  {
    /// Near nullspace parameters.
    nullspace_params nullspace;

    // Strong connectivity threshold
    double eps_strong = 0.08;

    // Block size for non-scalar problems.
    Int32 block_size = 1;

    params() = default;

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, nullspace)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, eps_strong)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, block_size)
    {
      p.check_params({ "nullspace", "eps_strong", "block_size" });
    }

    void get(PropertyTree& p, const std::string& path) const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, nullspace);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, eps_strong);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, block_size);
    }

  } & prm;

  std::shared_ptr<DistributedMatrix<bool_backend>> conn;
  std::shared_ptr<matrix> p_tent;

  DistributedPMISAggregation(const matrix& A, params& prm)
  : prm(prm)
  {
    ptrdiff_t n = A.loc_rows();
    std::vector<ptrdiff_t> state(n);
    std::vector<int> owner(n);

    if (prm.block_size == 1) {
      conn = conn_strength(A, prm.eps_strong);

      ptrdiff_t naggr = aggregates(*conn, state, owner);
      p_tent = tentative_prolongation(A.comm(), n, naggr, state, owner);
    }
    else {
      typedef typename math::scalar_of<value_type>::type scalar;
      using sbackend = BuiltinBackend<scalar, col_type, ptr_type>;

      ptrdiff_t np = n / prm.block_size;

      assert(np * prm.block_size == n && "Matrix size should be divisible by block_size");

      DistributedMatrix<sbackend> A_pw(A.comm(),
                                       pointwise_matrix(*A.local(), prm.block_size),
                                       pointwise_matrix(*A.remote(), prm.block_size));

      auto conn_pw = conn_strength(A_pw, prm.eps_strong);

      std::vector<ptrdiff_t> state_pw(np);
      std::vector<int> owner_pw(np);

      ptrdiff_t naggr = aggregates(*conn_pw, state_pw, owner_pw);

      conn = std::make_shared<DistributedMatrix<bool_backend>>(A.comm(),
                                                               expand_conn(*A.local(), *A_pw.local(), *conn_pw->local(), prm.block_size),
                                                               expand_conn(*A.remote(), *A_pw.remote(), *conn_pw->remote(), prm.block_size));

      arccoreParallelFor(0, np, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
        for (ptrdiff_t ip = begin; ip < (begin + size); ++ip) {
          ptrdiff_t i = ip * prm.block_size;
          ptrdiff_t s = state_pw[ip];
          int o = owner_pw[ip];

          for (unsigned k = 0; k < prm.block_size; ++k) {
            state[i + k] = (s < 0) ? s : (s * prm.block_size + k);
            owner[i + k] = o;
          }
        }
      });

      p_tent = tentative_prolongation(A.comm(), n, naggr * prm.block_size, state, owner);
    }
  }

  std::shared_ptr<DistributedMatrix<bool_backend>>
  squared_interface(const DistributedMatrix<bool_backend>& A)
  {
    const CommunicationPattern<bool_backend>& C = A.cpat();

    bool_matrix& A_loc = *A.local();
    bool_matrix& A_rem = *A.remote();

    ptrdiff_t A_rows = A.loc_rows();

    ptrdiff_t A_beg = A.loc_col_shift();
    ptrdiff_t A_end = A_beg + A_rows;

    auto a_nbr = remote_rows(C, A, false);
    bool_matrix& A_nbr = *a_nbr;

    // Build mapping from global to local column numbers in the remote part of
    // the square matrix.
    std::vector<ptrdiff_t> rem_cols(A_rem.nbNonZero() + A_nbr.nbNonZero());

    std::copy(A_nbr.col.data(), A_nbr.col.data() + A_nbr.nbNonZero(),
              std::copy(A_rem.col.data(), A_rem.col.data() + A_rem.nbNonZero(), rem_cols.begin()));

    std::sort(rem_cols.begin(), rem_cols.end());
    rem_cols.erase(std::unique(rem_cols.begin(), rem_cols.end()), rem_cols.end());

    ptrdiff_t n_rem_cols = 0;
    std::unordered_map<ptrdiff_t, int> rem_idx(2 * rem_cols.size());
    for (ptrdiff_t c : rem_cols) {
      if (c >= A_beg && c < A_end)
        continue;
      rem_idx[c] = n_rem_cols++;
    }

    // Build the product.
    auto s_loc = std::make_shared<bool_matrix>();
    auto s_rem = std::make_shared<bool_matrix>();

    bool_matrix& S_loc = *s_loc;
    bool_matrix& S_rem = *s_rem;

    S_loc.set_size(A_rows, A_rows, false);
    S_rem.set_size(A_rows, 0, false);

    S_loc.ptr[0] = 0;
    S_rem.ptr[0] = 0;

    ARCCORE_ALINA_TIC("analyze");
    arccoreParallelFor(0, A_rows, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      std::vector<ptrdiff_t> loc_marker(A_rows, -1);
      std::vector<ptrdiff_t> rem_marker(n_rem_cols, -1);

      for (ptrdiff_t ia = begin; ia < (begin + size); ++ia) {
        ptrdiff_t loc_cols = 0;
        ptrdiff_t rem_cols = 0;

        for (ptrdiff_t ja = A_rem.ptr[ia], ea = A_rem.ptr[ia + 1]; ja < ea; ++ja) {
          ptrdiff_t ca = C.local_index(A_rem.col[ja]);

          for (ptrdiff_t jb = A_nbr.ptr[ca], eb = A_nbr.ptr[ca + 1]; jb < eb; ++jb) {
            ptrdiff_t cb = A_nbr.col[jb];

            if (cb >= A_beg && cb < A_end) {
              cb -= A_beg;

              if (loc_marker[cb] != ia) {
                loc_marker[cb] = ia;
                ++loc_cols;
              }
            }
            else {
              cb = rem_idx[cb];

              if (rem_marker[cb] != ia) {
                rem_marker[cb] = ia;
                ++rem_cols;
              }
            }
          }
        }

        for (ptrdiff_t ja = A_loc.ptr[ia], ea = A_loc.ptr[ia + 1]; ja < ea; ++ja) {
          ptrdiff_t ca = A_loc.col[ja];

          for (ptrdiff_t jb = A_rem.ptr[ca], eb = A_rem.ptr[ca + 1]; jb < eb; ++jb) {
            ptrdiff_t cb = rem_idx[A_rem.col[jb]];

            if (rem_marker[cb] != ia) {
              rem_marker[cb] = ia;
              ++rem_cols;
            }
          }
        }

        if (rem_cols) {
          for (ptrdiff_t ja = A_loc.ptr[ia], ea = A_loc.ptr[ia + 1]; ja < ea; ++ja) {
            ptrdiff_t ca = A_loc.col[ja];

            for (ptrdiff_t jb = A_loc.ptr[ca], eb = A_loc.ptr[ca + 1]; jb < eb; ++jb) {
              ptrdiff_t cb = A_loc.col[jb];

              if (loc_marker[cb] != ia) {
                loc_marker[cb] = ia;
                ++loc_cols;
              }
            }
          }
        }

        S_rem.ptr[ia + 1] = rem_cols;
        S_loc.ptr[ia + 1] = rem_cols ? loc_cols : 0;
      }
    });
    ARCCORE_ALINA_TOC("analyze");

    S_loc.set_nonzeros(S_loc.scan_row_sizes(), false);
    S_rem.set_nonzeros(S_rem.scan_row_sizes(), false);

    ARCCORE_ALINA_TIC("compute");
    arccoreParallelFor(0, A_rows, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      std::vector<ptrdiff_t> loc_marker(A_rows, -1);
      std::vector<ptrdiff_t> rem_marker(n_rem_cols, -1);

      for (ptrdiff_t ia = begin; ia < (begin + size); ++ia) {
        ptrdiff_t loc_beg = S_loc.ptr[ia];
        ptrdiff_t rem_beg = S_rem.ptr[ia];
        ptrdiff_t loc_end = loc_beg;
        ptrdiff_t rem_end = rem_beg;

        if (rem_beg == S_rem.ptr[ia + 1])
          continue;

        for (ptrdiff_t ja = A_loc.ptr[ia], ea = A_loc.ptr[ia + 1]; ja < ea; ++ja) {
          ptrdiff_t ca = A_loc.col[ja];

          for (ptrdiff_t jb = A_loc.ptr[ca], eb = A_loc.ptr[ca + 1]; jb < eb; ++jb) {
            ptrdiff_t cb = A_loc.col[jb];

            if (loc_marker[cb] < loc_beg) {
              loc_marker[cb] = loc_end;
              S_loc.col[loc_end] = cb;
              ++loc_end;
            }
          }

          for (ptrdiff_t jb = A_rem.ptr[ca], eb = A_rem.ptr[ca + 1]; jb < eb; ++jb) {
            ptrdiff_t gb = A_rem.col[jb];
            ptrdiff_t cb = rem_idx[gb];

            if (rem_marker[cb] < rem_beg) {
              rem_marker[cb] = rem_end;
              S_rem.col[rem_end] = gb;
              ++rem_end;
            }
          }
        }

        for (ptrdiff_t ja = A_rem.ptr[ia], ea = A_rem.ptr[ia + 1]; ja < ea; ++ja) {
          ptrdiff_t ca = C.local_index(A_rem.col[ja]);

          for (ptrdiff_t jb = A_nbr.ptr[ca], eb = A_nbr.ptr[ca + 1]; jb < eb; ++jb) {
            ptrdiff_t gb = A_nbr.col[jb];

            if (gb >= A_beg && gb < A_end) {
              ptrdiff_t cb = gb - A_beg;

              if (loc_marker[cb] < loc_beg) {
                loc_marker[cb] = loc_end;
                S_loc.col[loc_end] = cb;
                ++loc_end;
              }
            }
            else {
              ptrdiff_t cb = rem_idx[gb];

              if (rem_marker[cb] < rem_beg) {
                rem_marker[cb] = rem_end;
                S_rem.col[rem_end] = gb;
                ++rem_end;
              }
            }
          }
        }
      }
    });
    ARCCORE_ALINA_TOC("compute");

    return std::make_shared<DistributedMatrix<bool_backend>>(A.comm(), s_loc, s_rem);
  }

  template <class B>
  std::shared_ptr<DistributedMatrix<bool_backend>>
  conn_strength(const DistributedMatrix<B>& A, scalar_type eps_strong)
  {
    typedef typename B::value_type val_type;
    typedef CSRMatrix<val_type> B_matrix;

    ARCCORE_ALINA_TIC("conn_strength");
    ptrdiff_t n = A.loc_rows();

    const B_matrix& A_loc = *A.local();
    const B_matrix& A_rem = *A.remote();
    const CommunicationPattern<B>& C = A.cpat();

    scalar_type eps_squared = eps_strong * eps_strong;

    auto d = diagonal(A_loc);
    numa_vector<val_type>& D = *d;

    std::vector<val_type> D_loc(C.send.count());
    std::vector<val_type> D_rem(C.recv.count());

    for (size_t i = 0, nv = C.send.count(); i < nv; ++i)
      D_loc[i] = D[C.send.col[i]];

    C.exchange(&D_loc[0], &D_rem[0]);

    auto s_loc = std::make_shared<bool_matrix>();
    auto s_rem = std::make_shared<bool_matrix>();

    bool_matrix& S_loc = *s_loc;
    bool_matrix& S_rem = *s_rem;

    S_loc.set_size(n, n, true);
    S_rem.set_size(n, 0, true);

    S_loc.val.resize(A_loc.nbNonZero());
    S_rem.val.resize(A_rem.nbNonZero());

    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      for (ptrdiff_t i = begin; i < (begin + size); ++i) {
        val_type eps_dia_i = eps_squared * D[i];

        for (ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i + 1]; j < e; ++j) {
          ptrdiff_t c = A_loc.col[j];
          val_type v = A_loc.val[j];

          if ((S_loc.val[j] = (c == i || (eps_dia_i * D[c] < v * v))))
            ++S_loc.ptr[i + 1];
        }

        for (ptrdiff_t j = A_rem.ptr[i], e = A_rem.ptr[i + 1]; j < e; ++j) {
          ptrdiff_t c = C.local_index(A_rem.col[j]);
          val_type v = A_rem.val[j];

          if ((S_rem.val[j] = (eps_dia_i * D_rem[c] < v * v)))
            ++S_rem.ptr[i + 1];
        }
      }
    });

    S_loc.setNbNonZero(S_loc.scan_row_sizes());
    S_rem.setNbNonZero(S_rem.scan_row_sizes());

    S_loc.col.resize(S_loc.nbNonZero());
    S_rem.col.resize(S_rem.nbNonZero());

    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      for (ptrdiff_t i = begin; i < (begin + size); ++i) {
        ptrdiff_t loc_head = S_loc.ptr[i];
        ptrdiff_t rem_head = S_rem.ptr[i];

        for (ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i + 1]; j < e; ++j)
          if (S_loc.val[j])
            S_loc.col[loc_head++] = A_loc.col[j];

        for (ptrdiff_t j = A_rem.ptr[i], e = A_rem.ptr[i + 1]; j < e; ++j)
          if (S_rem.val[j])
            S_rem.col[rem_head++] = A_rem.col[j];
      }
    });
    ARCCORE_ALINA_TOC("conn_strength");

    return std::make_shared<DistributedMatrix<bool_backend>>(A.comm(), s_loc, s_rem);
  }

  ptrdiff_t aggregates(const DistributedMatrix<bool_backend>& A,
                       std::vector<ptrdiff_t>& loc_state,
                       std::vector<int>& loc_owner)
  {
    ARCCORE_ALINA_TIC("PMIS");
    static const int tag_exc_cnt = 4001;
    static const int tag_exc_pts = 4002;

    const bool_matrix& A_loc = *A.local();
    const bool_matrix& A_rem = *A.remote();

    ptrdiff_t n = A_loc.nbRow();

    mpi_communicator comm = A.comm();

    // 1. Get symbolic square of the connectivity matrix.
    ARCCORE_ALINA_TIC("symbolic square");
    auto S = squared_interface(A);
    const bool_matrix& S_loc = *S->local();
    const bool_matrix& S_rem = *S->remote();
    const CommunicationPattern<bool_backend>& Sp = S->cpat();
    ARCCORE_ALINA_TOC("symbolic square");

    // 2. Apply PMIS algorithm to the symbolic square.
    ptrdiff_t n_undone = 0;
    std::vector<ptrdiff_t> rem_state(Sp.recv.count(), DistributedPMISAggregation::undone);
    std::vector<int> rem_owner(Sp.recv.count(), -1);
    std::vector<ptrdiff_t> send_state(Sp.send.count());
    std::vector<int> send_owner(Sp.send.count());

    // Remove lonely nodes.
#pragma omp parallel for reduction(+ : n_undone)
    for (ptrdiff_t i = 0; i < n; ++i) {
      ptrdiff_t wl = A_loc.ptr[i + 1] - A_loc.ptr[i];
      ptrdiff_t wr = S_rem.ptr[i + 1] - S_rem.ptr[i];

      if (wl + wr == 1) {
        loc_state[i] = DistributedPMISAggregation::deleted;
        ++n_undone;
      }
      else {
        loc_state[i] = DistributedPMISAggregation::undone;
      }

      loc_owner[i] = -1;
    }

    n_undone = n - n_undone;

    // Exchange state
    for (ptrdiff_t i = 0, m = Sp.send.count(); i < m; ++i)
      send_state[i] = loc_state[Sp.send.col[i]];
    Sp.exchange(&send_state[0], &rem_state[0]);

    std::vector<std::vector<ptrdiff_t>> send_pts(Sp.recv.nbr.size());
    std::vector<ptrdiff_t> recv_pts;

    UniqueArray<MessagePassing::Request> send_cnt_req(Sp.recv.nbr.size());
    UniqueArray<MessagePassing::Request> send_pts_req(Sp.recv.nbr.size());

    ptrdiff_t naggr = 0;

    std::vector<ptrdiff_t> nbr;

    while (true) {
      for (size_t i = 0; i < Sp.recv.nbr.size(); ++i)
        send_pts[i].clear();

      if (n_undone) {
        for (ptrdiff_t i = 0; i < n; ++i) {
          if (loc_state[i] != DistributedPMISAggregation::undone)
            continue;

          if (S_rem.ptr[i + 1] > S_rem.ptr[i]) {
            // Boundary points
            bool selectable = true;
            for (ptrdiff_t j = S_rem.ptr[i], e = S_rem.ptr[i + 1]; j < e; ++j) {
              int d, c;
              std::tie(d, c) = Sp.remote_info(S_rem.col[j]);

              if (rem_state[c] == DistributedPMISAggregation::undone && Sp.recv.nbr[d] > comm.rank) {
                selectable = false;
                break;
              }
            }

            if (!selectable)
              continue;

            ptrdiff_t id = naggr++;
            loc_owner[i] = comm.rank;
            loc_state[i] = id;
            --n_undone;

            // A gives immediate neighbors
            for (ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i + 1]; j < e; ++j) {
              ptrdiff_t c = A_loc.col[j];
              if (c != i) {
                if (loc_state[c] == DistributedPMISAggregation::undone)
                  --n_undone;
                loc_owner[c] = comm.rank;
                loc_state[c] = id;
              }
            }

            for (ptrdiff_t j = A_rem.ptr[i], e = A_rem.ptr[i + 1]; j < e; ++j) {
              ptrdiff_t c = A_rem.col[j];
              int d, k;
              std::tie(d, k) = Sp.remote_info(c);

              rem_state[k] = id;

              send_pts[d].push_back(c);
              send_pts[d].push_back(id);
            }

            // S gives removed neighbors
            for (ptrdiff_t j = S_loc.ptr[i], e = S_loc.ptr[i + 1]; j < e; ++j) {
              ptrdiff_t c = S_loc.col[j];
              if (c != i && loc_state[c] == DistributedPMISAggregation::undone) {
                loc_owner[c] = comm.rank;
                loc_state[c] = id;
                --n_undone;
              }
            }

            for (ptrdiff_t j = S_rem.ptr[i], e = S_rem.ptr[i + 1]; j < e; ++j) {
              ptrdiff_t c = S_rem.col[j];
              int d, k;
              std::tie(d, k) = Sp.remote_info(c);

              if (rem_state[k] == DistributedPMISAggregation::undone) {
                rem_state[k] = id;
                send_pts[d].push_back(c);
                send_pts[d].push_back(id);
              }
            }
          }
          else {
            // Inner points
            ptrdiff_t id = naggr++;
            loc_owner[i] = comm.rank;
            loc_state[i] = id;
            --n_undone;

            nbr.clear();

            for (ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i + 1]; j < e; ++j) {
              ptrdiff_t c = A_loc.col[j];

              if (c != i && loc_state[c] != DistributedPMISAggregation::deleted) {
                if (loc_state[c] == DistributedPMISAggregation::undone)
                  --n_undone;
                loc_owner[c] = comm.rank;
                loc_state[c] = id;
                nbr.push_back(c);
              }
            }

            for (ptrdiff_t k : nbr) {
              for (ptrdiff_t j = A_loc.ptr[k], e = A_loc.ptr[k + 1]; j < e; ++j) {
                ptrdiff_t c = A_loc.col[j];
                if (c != k && loc_state[c] == DistributedPMISAggregation::undone) {
                  loc_owner[c] = comm.rank;
                  loc_state[c] = id;
                  --n_undone;
                }
              }
            }
          }
        }
      }

      for (size_t i = 0; i < Sp.recv.nbr.size(); ++i) {
        int npts = send_pts[i].size();
        send_cnt_req[i] = comm.doISend(&npts, 1, Sp.recv.nbr[i], tag_exc_cnt);

        if (!npts)
          continue;
        send_pts_req[i] = comm.doISend(&send_pts[i][0], npts, Sp.recv.nbr[i], tag_exc_pts);
      }

      for (size_t i = 0; i < Sp.send.nbr.size(); ++i) {
        int npts;
        comm.doReceive(&npts, 1, Sp.send.nbr[i], tag_exc_cnt);

        if (!npts)
          continue;
        recv_pts.resize(npts);
        comm.doReceive(&recv_pts[0], npts, Sp.send.nbr[i], tag_exc_pts);

        for (int k = 0; k < npts; k += 2) {
          ptrdiff_t c = recv_pts[k] - Sp.loc_col_shift();
          ptrdiff_t id = recv_pts[k + 1];

          if (loc_state[c] == DistributedPMISAggregation::undone)
            --n_undone;

          loc_owner[c] = Sp.send.nbr[i];
          loc_state[c] = id;
        }
      }

      for (size_t i = 0; i < Sp.recv.nbr.size(); ++i) {
        int npts = send_pts[i].size();
        comm.wait(send_cnt_req[i]);
        if (npts == 0)
          continue;
        comm.wait(send_pts_req[i]);
      }

      for (ptrdiff_t i = 0, m = Sp.send.count(); i < m; ++i)
        send_state[i] = loc_state[Sp.send.col[i]];
      Sp.exchange(&send_state[0], &rem_state[0]);

      if (0 == comm.reduceSum(n_undone))
        break;
    }

    // Some of the aggregates could potentially vanish during expansion
    // step (*) above. We need to exclude those and renumber the rest.
    ARCCORE_ALINA_TIC("drop empty aggregates");
    for (ptrdiff_t i = 0, m = Sp.send.count(); i < m; ++i)
      send_owner[i] = loc_owner[Sp.send.col[i]];
    Sp.exchange(&send_owner[0], &rem_owner[0]);

    std::vector<ptrdiff_t> new_id(naggr + 1, 0);
    for (ptrdiff_t i = 0; i < n; ++i) {
      if (loc_owner[i] == comm.rank && loc_state[i] >= 0)
        new_id[loc_state[i] + 1] = 1;
    }

    for (size_t i = 0; i < Sp.recv.count(); ++i) {
      if (rem_owner[i] == comm.rank && rem_state[i] >= 0)
        new_id[rem_state[i] + 1] = 1;
    }

    std::partial_sum(new_id.begin(), new_id.end(), new_id.begin());

    if (comm.reduceSum(naggr - new_id.back()) > 0) {
      naggr = new_id.back();

      for (ptrdiff_t i = 0; i < n; ++i) {
        if (loc_owner[i] == comm.rank && loc_state[i] >= 0) {
          loc_state[i] = new_id[loc_state[i]];
        }
      }

      for (size_t i = 0; i < Sp.recv.nbr.size(); ++i) {
        send_pts[i].clear();
      }

      for (auto p = Sp.remote_begin(); p != Sp.remote_end(); ++p) {
        ptrdiff_t c = p->first;

        int d, k;
        std::tie(d, k) = p->second;

        if (rem_owner[k] == comm.rank && rem_state[k] >= 0) {
          send_pts[d].push_back(c);
          send_pts[d].push_back(new_id[rem_state[k]]);
        }
      }

      for (size_t i = 0; i < Sp.recv.nbr.size(); ++i) {
        int npts = send_pts[i].size();
        send_cnt_req[i] = comm.doISend(&npts, 1, Sp.recv.nbr[i], tag_exc_cnt);

        if (!npts)
          continue;
        send_pts_req[i] = comm.doISend(&send_pts[i][0], npts, Sp.recv.nbr[i], tag_exc_pts);
      }

      for (size_t i = 0; i < Sp.send.nbr.size(); ++i) {
        int npts;
        comm.doReceive(&npts, 1, Sp.send.nbr[i], tag_exc_cnt);

        if (!npts)
          continue;
        recv_pts.resize(npts);
        comm.doReceive(&recv_pts[0], npts, Sp.send.nbr[i], tag_exc_pts);

        for (int k = 0; k < npts; k += 2) {
          ptrdiff_t c = recv_pts[k] - Sp.loc_col_shift();
          ptrdiff_t id = recv_pts[k + 1];

          loc_state[c] = id;
        }
      }

      for (size_t i = 0; i < Sp.recv.nbr.size(); ++i) {
        int npts = send_pts[i].size();
        comm.wait(send_cnt_req[i]);
        if (!npts)
          continue;
        comm.wait(send_pts_req[i]);
      }
    }

    ARCCORE_ALINA_TOC("drop empty aggregates");
    ARCCORE_ALINA_TOC("PMIS");

    return naggr;
  }

  std::shared_ptr<matrix>
  tentative_prolongation(mpi_communicator comm, ptrdiff_t n, ptrdiff_t naggr,
                         std::vector<ptrdiff_t>& state, std::vector<int>& owner)
  {
    auto p_loc = std::make_shared<build_matrix>();
    auto p_rem = std::make_shared<build_matrix>();
    build_matrix& P_loc = *p_loc;
    build_matrix& P_rem = *p_rem;

    ARCCORE_ALINA_TIC("tentative prolongation");

    if (int null_cols = prm.nullspace.cols) {
      ptrdiff_t nba = naggr / prm.block_size;

      std::vector<ptrdiff_t> fdom = comm.exclusive_sum(n);
      std::vector<ptrdiff_t> cdom = comm.exclusive_sum(naggr);

      std::vector<int> scounts(comm.size, 0);
      std::vector<int> rcounts(comm.size);

      // Precompute the shape of the prolongation operator.
      // Each row contains exactly nullspace.cols non-zero entries.
      // Rows that do not belong to any aggregate are empty.
      P_loc.set_size(n, null_cols * nba, true);
      P_rem.set_size(n, 0, true);

      // Also count the number of local DOFs in local aggregates
      ptrdiff_t loc_dofs = 0;

      for (ptrdiff_t i = 0; i < n; ++i) {
        if (state[i] == DistributedPMISAggregation::deleted)
          continue;

        if (owner[i] == comm.rank) {
          P_loc.ptr[i + 1] = null_cols;
          ++loc_dofs;
        }
        else {
          P_rem.ptr[i + 1] = null_cols;
          ++scounts[owner[i]];
        }
      }

      // Setup the exchange
      MPI_Request req;
      MPI_Ialltoall(scounts.data(), 1, MPI_INT,
                    rcounts.data(), 1, MPI_INT,
                    comm, &req);

      P_loc.set_nonzeros(P_loc.scan_row_sizes());
      P_rem.set_nonzeros(P_rem.scan_row_sizes());

      MPI_Wait(&req, MPI_STATUS_IGNORE);

      int snbr = 0;
      int rnbr = 0;
      for (int i = 0; i < comm.size; ++i) {
        if (scounts[i])
          ++snbr;
        if (rcounts[i])
          ++rnbr;
      }

      std::vector<int> send_nbr;
      send_nbr.reserve(snbr);
      std::vector<int> recv_nbr;
      recv_nbr.reserve(rnbr);
      std::vector<int> send_ptr;
      send_ptr.reserve(snbr + 1);
      send_ptr.push_back(0);
      std::vector<int> recv_ptr;
      recv_ptr.reserve(rnbr + 1);
      recv_ptr.push_back(0);

      for (int i = 0; i < comm.size; ++i) {
        if (scounts[i]) {
          send_nbr.push_back(i);
          send_ptr.push_back(send_ptr.back() + scounts[i]);
        }
        if (rcounts[i]) {
          recv_nbr.push_back(i);
          recv_ptr.push_back(recv_ptr.back() + rcounts[i]);
        }
      }

      int send_dofs = send_ptr.back();
      int recv_dofs = recv_ptr.back();

      std::vector<ptrdiff_t> send_agg(send_dofs); // IDs of the aggregates we are sending
      std::vector<ptrdiff_t> send_dof(send_dofs); // DOFs included in the aggregates
      std::vector<double> send_row(send_dofs * null_cols); // Rows of the nullspace matrix corresponding to the DOFs

      std::vector<ptrdiff_t> recv_agg(recv_dofs); // IDs of the aggregates we are receiving
      std::vector<ptrdiff_t> recv_dof(recv_dofs); // DOFs included in the aggregates
      std::vector<double> recv_row(recv_dofs * null_cols); // Rows of the nullspace matrix corresponding to the DOFs

      // Prepare the data to send
      std::vector<ptrdiff_t> send_rank_ptr(comm.size + 1);
      send_rank_ptr[0] = 0;
      std::partial_sum(scounts.begin(), scounts.end(), send_rank_ptr.begin() + 1);
      for (ptrdiff_t i = 0; i < n; ++i) {
        auto s = state[i];
        auto o = owner[i];

        if (s == DistributedPMISAggregation::deleted)
          continue;
        if (o == comm.rank)
          continue;

        auto head = send_rank_ptr[o]++;

        send_agg[head] = s;
        send_dof[head] = i + fdom[comm.rank];
        std::copy_n(&prm.nullspace.B[i * null_cols], null_cols, &send_row[head * null_cols]);
      }

      // Exchange the data
      UniqueArray<MessagePassing::Request> send_req(3 * snbr);
      UniqueArray<MessagePassing::Request> recv_req(3 * rnbr);

      for (int i = 0; i < rnbr; ++i) {
        int n = recv_nbr[i];
        int p = recv_ptr[i];
        int w = recv_ptr[i + 1] - p;

        MessagePassing::Request* req = &recv_req[3 * i];

        req[0] = comm.doIReceive(&recv_agg[p], w, n, tag_exc_agg);
        req[1] = comm.doIReceive(&recv_dof[p], w, n, tag_exc_dof);
        req[2] = comm.doIReceive(&recv_row[null_cols * p], null_cols * w, n, tag_exc_row);
      }

      for (int i = 0; i < snbr; ++i) {
        int n = send_nbr[i];
        int p = send_ptr[i];
        int w = send_ptr[i + 1] - p;

        MessagePassing::Request* req = &send_req[3 * i];

        req[0] = comm.doISend(&send_agg[p], w, n, tag_exc_agg);
        req[1] = comm.doISend(&send_dof[p], w, n, tag_exc_dof);
        req[2] = comm.doISend(&send_row[null_cols * p], null_cols * w, n, tag_exc_row);
      }

      ARCCORE_ALINA_TIC("MPI Wait");
      comm.waitAll(recv_req);
      comm.waitAll(send_req);
      ARCCORE_ALINA_TOC("MPI Wait");

      // Sort the fine-level points by the aggregate number.
      // The order vector contains tuples of (aggr, dof, src, dst),
      // where src points to a row in B, and dst points to a row in P
      std::vector<std::tuple<ptrdiff_t, ptrdiff_t, double*, value_type*>> order;
      order.reserve(loc_dofs + recv_dofs);
      for (ptrdiff_t i = 0; i < n; ++i) {
        auto s = state[i];
        auto o = owner[i];

        if (s == DistributedPMISAggregation::deleted)
          continue;
        if (o != comm.rank)
          continue;

        order.emplace_back(s / prm.block_size, i + fdom[comm.rank],
                           &prm.nullspace.B[i * null_cols], &P_loc.val[P_loc.ptr[i]]);
      }
      for (ptrdiff_t i = 0; i < recv_dofs; ++i) {
        order.emplace_back(recv_agg[i] / prm.block_size, recv_dof[i],
                           &recv_row[i * null_cols], nullptr);
      }
      std::sort(order.begin(), order.end());

      std::vector<ptrdiff_t> aggr_ptr(nba + 1, 0);
      for (size_t i = 0; i < order.size(); ++i)
        ++aggr_ptr[std::get<0>(order[i]) + 1];
      std::partial_sum(aggr_ptr.begin(), aggr_ptr.end(), aggr_ptr.begin());

      // Compute the tentative prolongation operator and null-space vectors
      // for the coarser level.
      std::vector<double> Bnew;
      Bnew.resize(nba * null_cols * null_cols);

      arccoreParallelFor(0, nba, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
        Alina::detail::QRFactorization<double> qr;
        std::vector<double> Bpart;

        for (ptrdiff_t i = begin; i < (begin + size); ++i) {
          auto aggr_beg = aggr_ptr[i];
          auto aggr_end = aggr_ptr[i + 1];
          auto d = aggr_end - aggr_beg;

          Bpart.resize(d * null_cols);

          for (ptrdiff_t j = aggr_beg, r = 0; j < aggr_end; ++j, ++r) {
            auto src = std::get<2>(order[j]);
            for (int c = 0; c < null_cols; ++c)
              Bpart[r + d * c] = src[c];
          }

          qr.factorize(d, null_cols, &Bpart[0], Alina::detail::col_major);

          for (ptrdiff_t r = 0, k = i * null_cols * null_cols; r < null_cols; ++r)
            for (int c = 0; c < null_cols; ++c, ++k)
              Bnew[k] = qr.R(r, c);

          for (ptrdiff_t j = aggr_beg, r = 0; j < aggr_end; ++j, ++r) {
            auto src = std::get<2>(order[j]);
            auto dst = std::get<3>(order[j]);

            if (dst) {
              // TODO: this is just a workaround to make non-scalar value
              // types compile. Most probably this won't actually work.
              for (int c = 0; c < null_cols; ++c)
                dst[c] = qr.Q(r, c) * math::identity<value_type>();
            }
            else {
              for (int c = 0; c < null_cols; ++c)
                src[c] = qr.Q(r, c);
            }
          }
        }
      });

      // Exchange the computed rows of the prolongation operator with the
      // owners.
      for (int i = 0; i < snbr; ++i) {
        int n = send_nbr[i];
        int p = send_ptr[i];
        int w = send_ptr[i + 1] - p;
        send_req[i] = comm.doIReceive(&send_row[null_cols * p], null_cols * w, n, tag_exc_row);
      }

      for (int i = 0; i < rnbr; ++i) {
        int n = recv_nbr[i];
        int p = recv_ptr[i];
        int w = recv_ptr[i + 1] - p;
        recv_req[i] = comm.doISend(&recv_row[null_cols * p], null_cols * w, n, tag_exc_row);
      }

      // Fill column numbers
      arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
        for (ptrdiff_t i = begin; i < (begin + size); ++i) {
          ptrdiff_t s = state[i];
          if (s == DistributedPMISAggregation::deleted)
            continue;

          int d = owner[i];
          if (d == comm.rank) {
            auto col = &P_loc.col[P_loc.ptr[i]];
            for (int j = 0; j < null_cols; ++j) {
              col[j] = null_cols * s / prm.block_size + j;
            }
          }
          else {
            auto col = &P_rem.col[P_rem.ptr[i]];
            for (int j = 0; j < null_cols; ++j) {
              col[j] = null_cols * (s + cdom[d]) / prm.block_size + j;
            }
          }
        }
      });

      ARCCORE_ALINA_TIC("MPI Wait");
      comm.waitAll(send_req);
      comm.waitAll(recv_req);
      ARCCORE_ALINA_TOC("MPI Wait");

      // Use the P rows computed by the neighbors
      for (ptrdiff_t k = 0; k < send_dofs; ++k) {
        auto i = send_dof[k] - fdom[comm.rank];
        auto src = &send_row[k * null_cols];
        auto dst = &P_rem.val[P_rem.ptr[i]];

        for (ptrdiff_t j = 0; j < null_cols; ++j) {
          dst[j] = src[j] * math::identity<value_type>();
        }
      }

      std::swap(prm.nullspace.B, Bnew);
    }
    else {
      std::vector<ptrdiff_t> dom = comm.exclusive_sum(naggr);

      P_loc.set_size(n, naggr, true);
      P_rem.set_size(n, 0, true);

      arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
        for (ptrdiff_t i = begin; i < (begin + size); ++i) {
          if (state[i] == DistributedPMISAggregation::deleted)
            continue;

          if (owner[i] == comm.rank) {
            ++P_loc.ptr[i + 1];
          }
          else {
            ++P_rem.ptr[i + 1];
          }
        }
      });

      P_loc.set_nonzeros(P_loc.scan_row_sizes());
      P_rem.set_nonzeros(P_rem.scan_row_sizes());

      arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
        for (ptrdiff_t i = begin; i < (begin + size); ++i) {
          ptrdiff_t s = state[i];
          if (s == DistributedPMISAggregation::deleted)
            continue;

          int d = owner[i];
          if (d == comm.rank) {
            P_loc.col[P_loc.ptr[i]] = s;
            P_loc.val[P_loc.ptr[i]] = math::identity<value_type>();
          }
          else {
            P_rem.col[P_rem.ptr[i]] = s + dom[d];
            P_rem.val[P_rem.ptr[i]] = math::identity<value_type>();
          }
        }
      });
    }
    ARCCORE_ALINA_TOC("tentative prolongation");

    return std::make_shared<matrix>(comm, p_loc, p_rem);
  }

  template <class pw_matrix>
  std::shared_ptr<bool_matrix>
  expand_conn(const build_matrix& A, const pw_matrix& Ap, const bool_matrix& Cp,
              unsigned block_size) const
  {
    ptrdiff_t np = Cp.nbRow();
    ptrdiff_t n = np * block_size;

    auto c = std::make_shared<bool_matrix>();
    bool_matrix& C = *c;

    C.set_size(n, n, true);
    C.val.resize(A.nbNonZero());

    arccoreParallelFor(0, np, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      std::vector<ptrdiff_t> j(block_size);
      std::vector<ptrdiff_t> e(block_size);

      for (ptrdiff_t ip = begin; ip < (begin + size); ++ip) {
        ptrdiff_t ia = ip * block_size;

        for (unsigned k = 0; k < block_size; ++k) {
          j[k] = A.ptr[ia + k];
          e[k] = A.ptr[ia + k + 1];
        }

        for (ptrdiff_t jp = Ap.ptr[ip], ep = Ap.ptr[ip + 1]; jp < ep; ++jp) {
          ptrdiff_t cp = Ap.col[jp];
          bool sp = Cp.val[jp];

          ptrdiff_t col_end = (cp + 1) * block_size;

          for (unsigned k = 0; k < block_size; ++k) {
            ptrdiff_t beg = j[k];
            ptrdiff_t end = e[k];

            while (beg < end && A.col[beg] < col_end) {
              C.val[beg++] = sp;

              if (sp)
                ++C.ptr[ia + k + 1];
            }

            j[k] = beg;
          }
        }
      }
    });

    C.setNbNonZero(C.scan_row_sizes());
    C.col.resize(C.nbNonZero());

    arccoreParallelFor(0, np, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      std::vector<ptrdiff_t> j(block_size);
      std::vector<ptrdiff_t> e(block_size);
      std::vector<ptrdiff_t> h(block_size);

      for (ptrdiff_t ip = begin; ip < (begin + size); ++ip) {
        ptrdiff_t ia = ip * block_size;

        for (unsigned k = 0; k < block_size; ++k) {
          j[k] = A.ptr[ia + k];
          e[k] = A.ptr[ia + k + 1];
          h[k] = C.ptr[ia + k];
        }

        for (ptrdiff_t jp = Ap.ptr[ip], ep = Ap.ptr[ip + 1]; jp < ep; ++jp) {
          ptrdiff_t cp = Ap.col[jp];
          bool sp = Cp.val[jp];

          ptrdiff_t col_end = (cp + 1) * block_size;

          for (unsigned k = 0; k < block_size; ++k) {
            ptrdiff_t beg = j[k];
            ptrdiff_t end = e[k];
            ptrdiff_t hed = h[k];

            while (beg < end && A.col[beg] < col_end) {
              if (sp)
                C.col[hed++] = A.col[beg];
              ++beg;
            }

            j[k] = beg;
            h[k] = hed;
          }
        }
      }
    });

    return c;
  }

 private:

  static const int undone = -2;
  static const int deleted = -1;

  static const int tag_exc_agg = 4011;
  static const int tag_exc_dof = 4012;
  static const int tag_exc_row = 4013;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Distributed non-smoothed aggregation coarsening scheme.
 */
template <class Backend>
struct DistributedAggregationCoarsening
{
  typedef typename Backend::value_type value_type;
  typedef typename math::scalar_of<value_type>::type scalar_type;
  using build_matrix = Backend::matrix;

  struct params
  {
    // aggregation params
    typedef typename DistributedPMISAggregation<Backend>::params aggr_params;
    aggr_params aggr;

    /*!
     * \brief Over-interpolation factor \f$\alpha\f$.
     *
     * In case of aggregation coarsening, coarse-grid
     * correction of smooth error, and by this the overall convergence, can
     * often be substantially improved by using "over-interpolation", that is,
     * by multiplying the actual correction (corresponding to piecewise
     * constant interpolation) by some factor \f$\alpha > 1\f$. Equivalently,
     * this means that the coarse-level Galerkin operator is re-scaled by
     * \f$1 / \alpha\f$:
     * \f[I_h^HA_hI_H^h \to \frac{1}{\alpha}I_h^HA_hI_H^h.\f]
     *
     * \sa  \cite Stuben1999, Section 9.1 "Re-scaling of the Galerkin operator".
     */
    float over_interp = 1.5f;

    params() = default;

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, aggr)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, over_interp)
    {
      p.check_params({ "aggr", "over_interp" });
    }

    void get(Alina::PropertyTree& p, const std::string& path) const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, aggr);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, over_interp);
    }
  } prm;

  DistributedAggregationCoarsening(const params& prm = params())
  : prm(prm)
  {}

  std::tuple<std::shared_ptr<DistributedMatrix<Backend>>,
             std::shared_ptr<DistributedMatrix<Backend>>>
  transfer_operators(const DistributedMatrix<Backend>& A)
  {
    DistributedPMISAggregation<Backend> aggr(A, prm.aggr);
    return std::make_tuple(aggr.p_tent, transpose(*aggr.p_tent));
  }

  std::shared_ptr<DistributedMatrix<Backend>>
  coarse_operator(const DistributedMatrix<Backend>& A,
                  const DistributedMatrix<Backend>& P,
                  const DistributedMatrix<Backend>& R) const
  {
    return detail::scaled_galerkin(A, P, R, 1 / prm.over_interp);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend>
unsigned block_size(const DistributedAggregationCoarsening<Backend>& c)
{
  return c.prm.aggr.block_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Distributed smoothed aggregation coarsening scheme.
 */
template <class Backend>
struct DistributedSmoothedAggregationCoarsening
{
  typedef typename Backend::value_type value_type;
  typedef typename math::scalar_of<value_type>::type scalar_type;
  using build_matrix = Backend::matrix;
  using col_type = Backend::col_type;
  using ptr_type = Backend::ptr_type;
  using bool_backend = BuiltinBackend<char,col_type,ptr_type>;
  using bool_matrix = bool_backend::matrix;

  struct params
  {
    // aggregation params
    typedef typename DistributedPMISAggregation<Backend>::params aggr_params;
    aggr_params aggr;

    /// Relaxation factor.
    scalar_type relax;

    // Estimate the matrix spectral radius.
    // This usually improves convergence rate and results in faster solves,
    // but costs some time during setup.
    bool estimate_spectral_radius;

    // Number of power iterations to apply for the spectral radius
    // estimation. Use Gershgorin disk theorem when power_iters = 0.
    int power_iters;

    params()
    : relax(1.0f)
    , estimate_spectral_radius(false)
    , power_iters(0)
    {}

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, aggr)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, relax)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, estimate_spectral_radius)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, power_iters)
    {
      p.check_params({ "aggr", "relax", "estimate_spectral_radius", "power_iters" });
    }

    void get(PropertyTree& p, const std::string& path) const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, aggr);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, relax);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, estimate_spectral_radius);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, power_iters);
    }
  } prm;

  DistributedSmoothedAggregationCoarsening(const params& prm = params())
  : prm(prm)
  {}

  std::tuple<std::shared_ptr<DistributedMatrix<Backend>>,
             std::shared_ptr<DistributedMatrix<Backend>>>
  transfer_operators(const DistributedMatrix<Backend>& A)
  {
    typedef DistributedMatrix<Backend> DM;
    using build_matrix = Backend::matrix;

    DistributedPMISAggregation<Backend> aggr(A, prm.aggr);
    prm.aggr.eps_strong *= 0.5;

    mpi_communicator comm = A.comm();
    const build_matrix& A_loc = *A.local();
    const build_matrix& A_rem = *A.remote();

    bool_matrix& S_loc = *aggr.conn->local();
    bool_matrix& S_rem = *aggr.conn->remote();

    ARCCORE_ALINA_TIC("filtered matrix");
    ptrdiff_t n = A.loc_rows();

    scalar_type omega = prm.relax;
    if (prm.estimate_spectral_radius) {
      omega *= static_cast<scalar_type>(4.0 / 3) / spectral_radius<true>(A, prm.power_iters);
    }
    else {
      omega *= static_cast<scalar_type>(2.0 / 3);
    }

    auto af_loc = std::make_shared<build_matrix>();
    auto af_rem = std::make_shared<build_matrix>();

    build_matrix& Af_loc = *af_loc;
    build_matrix& Af_rem = *af_rem;

    numa_vector<value_type> Af_loc_val(S_loc.nbNonZero(), false);
    numa_vector<value_type> Af_rem_val(S_rem.nbNonZero(), false);

    Af_loc.own_data = false;
    Af_loc.setNbRow(S_loc.nbRow());
    Af_loc.ncols = S_loc.ncols;
    Af_loc.setNbNonZero(S_loc.nbNonZero());
    Af_loc.ptr.setPointerZeroCopy(S_loc.ptr.data());
    Af_loc.col.setPointerZeroCopy(S_loc.col.data());
    Af_loc.val.setPointerZeroCopy(Af_loc_val.data());

    Af_rem.own_data = false;
    Af_rem.setNbRow(S_rem.nbRow());
    Af_rem.ncols = S_rem.ncols;
    Af_rem.setNbNonZero(S_rem.nbNonZero());
    Af_rem.ptr.setPointerZeroCopy(S_rem.ptr.data());
    Af_rem.col.setPointerZeroCopy(S_rem.col.data());
    Af_rem.val.setPointerZeroCopy(Af_rem_val.data());

    numa_vector<value_type> Df(n, false);

    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      for (ptrdiff_t i = begin; i < (begin + size); ++i) {

        ptrdiff_t loc_head = Af_loc.ptr[i];
        ptrdiff_t rem_head = Af_rem.ptr[i];

        value_type dia_f = math::zero<value_type>();

        for (ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i + 1]; j < e; ++j)
          if (A_loc.col[j] == i || !S_loc.val[j])
            dia_f += A_loc.val[j];

        for (ptrdiff_t j = A_rem.ptr[i], e = A_rem.ptr[i + 1]; j < e; ++j)
          if (!S_rem.val[j])
            dia_f += A_rem.val[j];

        dia_f = -omega * math::inverse(dia_f);

        for (ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i + 1]; j < e; ++j) {
          if (A_loc.col[j] == i) {
            Af_loc.val[loc_head++] = (1 - omega) * math::identity<value_type>();
          }
          else if (S_loc.val[j]) {
            Af_loc.val[loc_head++] = dia_f * A_loc.val[j];
          }
        }

        for (ptrdiff_t j = A_rem.ptr[i], e = A_rem.ptr[i + 1]; j < e; ++j) {
          if (S_rem.val[j]) {
            Af_rem.val[rem_head++] = dia_f * A_rem.val[j];
          }
        }
      }
    });

    auto Af = std::make_shared<DM>(comm, af_loc, af_rem);
    ARCCORE_ALINA_TOC("filtered matrix");

    // 5. Smooth tentative prolongation with the filtered matrix.
    ARCCORE_ALINA_TIC("smoothing");
    auto P = product(*Af, *aggr.p_tent);
    ARCCORE_ALINA_TOC("smoothing");

    return std::make_tuple(P, transpose(*P));
  }

  std::shared_ptr<DistributedMatrix<Backend>>
  coarse_operator(const DistributedMatrix<Backend>& A,
                  const DistributedMatrix<Backend>& P,
                  const DistributedMatrix<Backend>& R) const
  {
    return detail::galerkin(A, P, R);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend>
unsigned block_size(const DistributedSmoothedAggregationCoarsening<Backend>& c)
{
  return c.prm.aggr.block_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
