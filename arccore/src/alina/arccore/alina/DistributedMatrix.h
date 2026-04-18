// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DistributedMatrix.h                                         (C) 2000-2026 */
/*                                                                           */
/* Distributed Matrix using message passing.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_MPI_DISTRIBUTED_MATRIX_H
#define ARCCORE_ALINA_MPI_DISTRIBUTED_MATRIX_H
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

#include <memory>
#include <unordered_map>
#include <random>

#include <mpi.h>

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/MessagePassingUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Call to handle communication pattern.
 */
template <class Backend>
class CommunicationPattern
{
 public:

  typedef typename Backend::value_type value_type;
  typedef typename math::rhs_of<value_type>::type rhs_type;
  typedef typename math::scalar_of<value_type>::type scalar_type;
  typedef typename Backend::matrix matrix;
  typedef typename Backend::vector vector;
  typedef typename Backend::params backend_params;
  typedef typename Backend::col_type col_type;
  typedef typename Backend::ptr_type ptr_type;

  struct
  {
    std::vector<ptrdiff_t> nbr;
    std::vector<ptr_type> ptr;
    std::vector<col_type> col;

    size_t count() const
    {
      return col.size();
    }

    mutable std::vector<rhs_type> val;
    mutable UniqueArray<MessagePassing::Request> req;
  } send;

  struct
  {
    std::vector<ptrdiff_t> nbr;
    std::vector<ptr_type> ptr;

    size_t count() const
    {
      return val.size();
    }

    mutable std::vector<rhs_type> val;
    mutable UniqueArray<MessagePassing::Request> req;
  } recv;

  std::shared_ptr<vector> x_rem;

  CommunicationPattern(mpi_communicator comm,
                       ptrdiff_t n_loc_cols,
                       size_t n_rem_cols, const col_type* p_rem_cols)
  : comm(comm)
  , loc_cols(n_loc_cols)
  {
    ARCCORE_ALINA_TIC("communication pattern");
    // Get domain boundaries
    std::vector<ptrdiff_t> domain = comm.exclusive_sum(n_loc_cols);
    loc_beg = domain[comm.rank];

    // Renumber remote columns,
    // find out how many remote values we need from each process.
    std::vector<col_type> rem_cols(p_rem_cols, p_rem_cols + n_rem_cols);

    std::sort(rem_cols.begin(), rem_cols.end());
    rem_cols.erase(std::unique(rem_cols.begin(), rem_cols.end()), rem_cols.end());

    ptrdiff_t ncols = rem_cols.size();
    ptrdiff_t rnbr = 0, snbr = 0, send_size = 0;

    {
      std::vector<int> rcounts(comm.size, 0);
      std::vector<int> scounts(comm.size);

      // Build index for column renumbering;
      // count how many domains send us data and how much.
      idx.reserve(2 * ncols);
      for (int i = 0, d = 0, last = -1; i < ncols; ++i) {
        while (rem_cols[i] >= domain[d + 1])
          ++d;

        ++rcounts[d];

        if (last < d) {
          last = d;
          ++rnbr;
        }

        idx.insert(idx.end(), std::make_pair(rem_cols[i], std::make_tuple(rnbr - 1, i)));
      }

      recv.val.resize(ncols);
      recv.req.resize(rnbr);

      recv.nbr.reserve(rnbr);
      recv.ptr.reserve(rnbr + 1);
      recv.ptr.push_back(0);

      for (int d = 0; d < comm.size; ++d) {
        if (rcounts[d]) {
          recv.nbr.push_back(d);
          recv.ptr.push_back(recv.ptr.back() + rcounts[d]);
        }
      }

      MPI_Alltoall(rcounts.data(), 1, MPI_INT, scounts.data(), 1, MPI_INT, comm);

      for (ptrdiff_t d = 0; d < comm.size; ++d) {
        if (scounts[d]) {
          ++snbr;
          send_size += scounts[d];
        }
      }

      send.col.resize(send_size);
      send.val.resize(send_size);
      send.req.resize(snbr);

      send.nbr.reserve(snbr);
      send.ptr.reserve(snbr + 1);
      send.ptr.push_back(0);

      for (ptrdiff_t d = 0; d < comm.size; ++d) {
        if (scounts[d]) {
          send.nbr.push_back(d);
          send.ptr.push_back(send.ptr.back() + scounts[d]);
        }
      }
    }

    // What columns do you need from me?
    for (size_t i = 0; i < send.nbr.size(); ++i)
      send.req[i] = comm.doIReceive(&send.col[send.ptr[i]], send.ptr[i + 1] - send.ptr[i],
                                    send.nbr[i], tag_exc_cols);

    // Here is what I need from you:
    for (size_t i = 0; i < recv.nbr.size(); ++i)
      recv.req[i] = comm.doISend(&rem_cols[recv.ptr[i]], recv.ptr[i + 1] - recv.ptr[i],
                              recv.nbr[i], tag_exc_cols);

    ARCCORE_ALINA_TIC("MPI Wait");
    comm.waitAll(recv.req);
    comm.waitAll(send.req);
    ARCCORE_ALINA_TOC("MPI Wait");

    // Shift columns to send to local numbering:
    for (col_type& c : send.col)
      c -= loc_beg;

    ARCCORE_ALINA_TOC("communication pattern");
  }

  template <class OtherBackend>
  CommunicationPattern(const CommunicationPattern<OtherBackend>& C)
  : comm(C.comm)
  , idx(C.idx)
  , loc_beg(C.loc_beg)
  , loc_cols(C.loc_cols)
  {
    send.nbr = C.send.nbr;
    send.ptr = C.send.ptr;
    send.col = C.send.col;
    send.val.resize(C.send.val.size());
    send.req.resize(C.send.req.size());

    recv.nbr = C.recv.nbr;
    recv.ptr = C.recv.ptr;
    recv.val.resize(C.recv.val.size());
    recv.req.resize(C.recv.req.size());
  }

  void move_to_backend(const backend_params& bprm = backend_params())
  {
    if (!x_rem) {
      x_rem = Backend::create_vector(recv.count(), bprm);
    }

    if (!gather) {
      gather = std::make_shared<Gather>(loc_cols, send.col, bprm);
    }
  }

  int domain(ptrdiff_t col) const
  {
    return std::get<0>(idx.at(col));
  }

  int local_index(ptrdiff_t col) const
  {
    return std::get<1>(idx.at(col));
  }

  std::tuple<int, int> remote_info(ptrdiff_t col) const
  {
    return idx.at(col);
  }

  std::unordered_map<ptrdiff_t, std::tuple<int, int>>::const_iterator
  remote_begin() const
  {
    return idx.cbegin();
  }

  std::unordered_map<ptrdiff_t, std::tuple<int, int>>::const_iterator
  remote_end() const
  {
    return idx.cend();
  }

  size_t renumber(size_t n, col_type* col) const
  {
    for (size_t i = 0; i < n; ++i)
      col[i] = std::get<1>(idx.at(col[i]));
    return recv.count();
  }

  bool needs_remote() const
  {
    return !recv.val.empty();
  }

  template <class Vector>
  void start_exchange(const Vector& x) const
  {
    // Start receiving ghost values from our neighbours.
    for (size_t i = 0; i < recv.nbr.size(); ++i)
      recv.req[i] = comm.doIReceive(&recv.val[recv.ptr[i]], recv.ptr[i + 1] - recv.ptr[i],
                                    recv.nbr[i], tag_exc_vals);

    // Start sending our data to neighbours.
    if (!send.val.empty()) {
      (*gather)(x, send.val);

      for (size_t i = 0; i < send.nbr.size(); ++i)
        send.req[i] = comm.doISend(&send.val[send.ptr[i]], send.ptr[i + 1] - send.ptr[i],
                                   send.nbr[i], tag_exc_vals);
    }
  }

  void finish_exchange() const
  {
    ARCCORE_ALINA_TIC("MPI Wait");
    comm.waitAll(recv.req);
    comm.waitAll(send.req);
    ARCCORE_ALINA_TOC("MPI Wait");

    if (!recv.val.empty())
      backend::copy(recv.val, *x_rem);
  }

  template <typename T>
  void exchange(const T* send_val, T* recv_val) const
  {
    for (size_t i = 0; i < recv.nbr.size(); ++i)
      recv.req[i] = comm.doIReceive(&recv_val[recv.ptr[i]], recv.ptr[i + 1] - recv.ptr[i],
                                 recv.nbr[i], tag_exc_vals);

    for (size_t i = 0; i < send.nbr.size(); ++i)
      send.req[i] = comm.doISend(const_cast<T*>(&send_val[send.ptr[i]]), send.ptr[i + 1] - send.ptr[i],
                              send.nbr[i], tag_exc_vals);

    ARCCORE_ALINA_TIC("MPI Wait");
    comm.waitAll(recv.req);
    comm.waitAll(send.req);
    ARCCORE_ALINA_TOC("MPI Wait");
  }

  mpi_communicator mpi_comm() const
  {
    return comm;
  }

  ptrdiff_t loc_col_shift() const
  {
    return loc_beg;
  }

 private:

  using Gather = Backend::gather;

  static const int tag_set_comm = 1001;
  static const int tag_exc_cols = 1002;
  static const int tag_exc_vals = 1003;

  mpi_communicator comm;

  std::unordered_map<ptrdiff_t, std::tuple<int, int>> idx;
  std::shared_ptr<Gather> gather;
  ptrdiff_t loc_beg;
  ptrdiff_t loc_cols;

  template <class B>
  friend class CommunicationPattern;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Distributed Matrix using message passing.
 */
template <class Backend>
class DistributedMatrix
{
 public:

  typedef typename Backend::value_type value_type;
  typedef typename math::rhs_of<value_type>::type rhs_type;
  typedef typename math::scalar_of<value_type>::type scalar_type;
  typedef typename Backend::params backend_params;
  typedef typename Backend::matrix matrix;
  typedef CommunicationPattern<Backend> CommPattern;
  typedef typename Backend::matrix build_matrix;

  DistributedMatrix(mpi_communicator comm,
                    std::shared_ptr<build_matrix> a_loc,
                    std::shared_ptr<build_matrix> a_rem,
                    std::shared_ptr<CommPattern> c = std::shared_ptr<CommPattern>())
  : a_loc(a_loc)
  , a_rem(a_rem)
  {
    if (c) {
      C = c;
    }
    else {
      C = std::make_shared<CommPattern>(comm, a_loc->ncols, a_rem->nbNonZero(), a_rem->col);
    }

    a_rem->ncols = C->recv.count();

    n_loc_rows = a_loc->nbRow();
    n_loc_cols = a_loc->ncols;
    n_loc_nonzeros = a_loc->nbNonZero() + a_rem->nbNonZero();

    n_glob_rows = comm.reduceSum(n_loc_rows);
    n_glob_cols = comm.reduceSum(n_loc_cols);
    n_glob_nonzeros = comm.reduceSum(n_loc_nonzeros);
  }

  // Copy the distributed_matrix from another backend
  template <class OtherBackend>
  DistributedMatrix(const DistributedMatrix<OtherBackend>& A)
  : a_loc(std::make_shared<build_matrix>(*A.local()))
  , a_rem(std::make_shared<build_matrix>(*A.remote()))
  {
    C = std::make_shared<CommPattern>(A.cpat());

    this->a_rem->ncols = C->recv.count();

    n_loc_rows = A.loc_rows();
    n_loc_cols = A.loc_cols();
    n_loc_nonzeros = A.loc_nonzeros();
    n_glob_rows = A.glob_rows();
    n_glob_cols = A.glob_cols();
    n_glob_nonzeros = A.glob_nonzeros();
  }

  template <class Matrix>
  DistributedMatrix(mpi_communicator comm,
                    const Matrix& A,
                    ptrdiff_t _n_loc_cols = -1)
  : n_loc_rows(backend::nbRow(A))
  , n_loc_cols(_n_loc_cols < 0 ? n_loc_rows : _n_loc_cols)
  , n_loc_nonzeros(backend::nonzeros(A))
  {
    // Get sizes of each domain in comm.
    std::vector<ptrdiff_t> domain = comm.exclusive_sum(n_loc_cols);
    ptrdiff_t loc_beg = domain[comm.rank];
    ptrdiff_t loc_end = domain[comm.rank + 1];

    n_glob_cols = domain.back();
    n_glob_rows = comm.reduceSum(n_loc_rows);
    n_glob_nonzeros = comm.reduceSum(n_loc_nonzeros);

    // Split the matrix into local and remote parts.
    a_loc = std::make_shared<build_matrix>();
    a_rem = std::make_shared<build_matrix>();

    build_matrix& A_loc = *a_loc;
    build_matrix& A_rem = *a_rem;

    A_loc.set_size(n_loc_rows, n_loc_cols, true);
    A_rem.set_size(n_loc_rows, 0, true);

    arccoreParallelFor(0, n_loc_rows, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      for (ptrdiff_t i = begin; i < (begin + size); ++i) {
        for (auto a = backend::row_begin(A, i); a; ++a) {
          ptrdiff_t c = a.col();

          if (loc_beg <= c && c < loc_end)
            ++A_loc.ptr[i + 1];
          else
            ++A_rem.ptr[i + 1];
        }
      }
    });

    A_loc.set_nonzeros(A_loc.scan_row_sizes());
    A_rem.set_nonzeros(A_rem.scan_row_sizes());

    arccoreParallelFor(0, n_loc_rows, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      for (ptrdiff_t i = begin; i < (begin + size); ++i) {
        ptrdiff_t loc_head = A_loc.ptr[i];
        ptrdiff_t rem_head = A_rem.ptr[i];

        for (auto a = backend::row_begin(A, i); a; ++a) {
          ptrdiff_t c = a.col();
          value_type v = a.value();

          if (loc_beg <= c && c < loc_end) {
            A_loc.col[loc_head] = c - loc_beg;
            A_loc.val[loc_head] = v;
            ++loc_head;
          }
          else {
            A_rem.col[rem_head] = c;
            A_rem.val[rem_head] = v;
            ++rem_head;
          }
        }
      }
    });

    C = std::make_shared<CommPattern>(comm, n_loc_cols, a_rem->nbNonZero(), a_rem->col);
    a_rem->ncols = C->recv.count();
  }

  mpi_communicator comm() const
  {
    return C->mpi_comm();
  }

  std::shared_ptr<build_matrix> local() const
  {
    return a_loc;
  }

  std::shared_ptr<build_matrix> remote() const
  {
    return a_rem;
  }

  std::shared_ptr<matrix> local_backend() const
  {
    return A_loc;
  }

  std::shared_ptr<matrix> remote_backend() const
  {
    return A_rem;
  }

  ptrdiff_t loc_rows() const
  {
    return n_loc_rows;
  }

  ptrdiff_t loc_cols() const
  {
    return n_loc_cols;
  }

  ptrdiff_t loc_col_shift() const
  {
    return C->loc_col_shift();
  }

  ptrdiff_t loc_nonzeros() const
  {
    return n_loc_nonzeros;
  }

  ptrdiff_t glob_rows() const
  {
    return n_glob_rows;
  }

  ptrdiff_t glob_cols() const
  {
    return n_glob_cols;
  }

  ptrdiff_t glob_nonzeros() const
  {
    return n_glob_nonzeros;
  }

  const CommunicationPattern<Backend>& cpat() const
  {
    return *C;
  }

  void set_local(std::shared_ptr<matrix> a)
  {
    A_loc = a;
  }

  void move_to_backend(const backend_params& bprm = backend_params(), bool keep_src = false)
  {
    ARCCORE_ALINA_TIC("move to backend");
    if (!A_loc) {
      A_loc = Backend::copy_matrix(a_loc, bprm);
    }

    if (!A_rem && a_rem && a_rem->nbNonZero() > 0) {
      if (keep_src) {
        auto rem_copy = std::make_shared<build_matrix>(*a_rem);
        C->renumber(rem_copy->nbNonZero(), rem_copy->col);
        A_rem = Backend::copy_matrix(rem_copy, bprm);
      }
      else {
        C->renumber(a_rem->nbNonZero(), a_rem->col);
        A_rem = Backend::copy_matrix(a_rem, bprm);
      }
    }

    C->move_to_backend(bprm);

    if (!keep_src) {
      a_loc.reset();
      a_rem.reset();
    }
    ARCCORE_ALINA_TOC("move to backend");
  }

  template <class A, class VecX, class B, class VecY>
  void mul(A alpha, const VecX& x, B beta, VecY& y) const
  {
    const auto one = math::identity<scalar_type>();

    C->start_exchange(x);

    // Compute local part of the product.
    backend::spmv(alpha, *A_loc, x, beta, y);

    // Compute remote part of the product.
    C->finish_exchange();

    if (C->needs_remote())
      backend::spmv(alpha, *A_rem, *C->x_rem, one, y);
  }

  template <class Vec1, class Vec2, class Vec3>
  void residual(const Vec1& f, const Vec2& x, Vec3& r) const
  {
    const auto one = math::identity<scalar_type>();

    C->start_exchange(x);
    backend::residual(f, *A_loc, x, r);

    C->finish_exchange();

    if (C->needs_remote())
      backend::spmv(-one, *A_rem, *C->x_rem, one, r);
  }

 private:

  std::shared_ptr<CommPattern> C;
  std::shared_ptr<matrix> A_loc, A_rem;
  std::shared_ptr<build_matrix> a_loc, a_rem;

  ptrdiff_t n_loc_rows, n_glob_rows;
  ptrdiff_t n_loc_cols, n_glob_cols;
  ptrdiff_t n_loc_nonzeros, n_glob_nonzeros;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend>
std::shared_ptr<DistributedMatrix<Backend>>
transpose(const DistributedMatrix<Backend>& A)
{
  ARCCORE_ALINA_TIC("MPI Transpose");
  typedef typename Backend::value_type value_type;
  typedef CommunicationPattern<Backend> CommPattern;
  typedef typename Backend::matrix build_matrix;
  typedef typename Backend::col_type col_type;

  static const int tag_cnt = 2001;
  static const int tag_col = 2002;
  static const int tag_val = 2003;

  mpi_communicator comm = A.comm();
  const CommPattern& C = A.cpat();

  build_matrix& A_loc = *A.local();
  build_matrix& A_rem = *A.remote();

  ptrdiff_t nrows = A_loc.ncols;
  ptrdiff_t ncols = A_loc.nbRow();

  UniqueArray<MessagePassing::Request> recv_cnt_req(C.send.req.size());
  UniqueArray<MessagePassing::Request> recv_col_req(C.send.req.size());
  UniqueArray<MessagePassing::Request> recv_val_req(C.send.req.size());

  UniqueArray<MessagePassing::Request> send_cnt_req(C.recv.req.size());
  UniqueArray<MessagePassing::Request> send_col_req(C.recv.req.size());
  UniqueArray<MessagePassing::Request> send_val_req(C.recv.req.size());

  // Our transposed remote part becomes remote part of someone else,
  // and the other way around.
  std::shared_ptr<build_matrix> t_ptr;
  {
    std::vector<col_type> tmp_col(A_rem.col.data(), A_rem.col.data() + A_rem.nbNonZero());
    C.renumber(tmp_col.size(), tmp_col.data());

    col_type* a_rem_col = tmp_col.data();
    col_type* a_rem_col_backup = A_rem.col.data();
    A_rem.col.setPointerZeroCopy(a_rem_col);

    //std::swap(a_rem_col, A_rem.col);

    t_ptr = transpose(A_rem);

    A_rem.col.setPointerZeroCopy(a_rem_col_backup);
    //std::swap(a_rem_col, A_rem.col);
  }
  build_matrix& t_rem = *t_ptr;

  // Shift to global numbering:
  std::vector<ptrdiff_t> domain = comm.exclusive_sum(ncols);
  ptrdiff_t loc_beg = domain[comm.rank];
  for (size_t i = 0; i < t_rem.nbNonZero(); ++i)
    t_rem.col[i] += loc_beg;

  // Shift from row pointers to row sizes:
  std::vector<ptrdiff_t> row_size(t_rem.nbRow());
  for (size_t i = 0; i < t_rem.nbRow(); ++i)
    row_size[i] = t_rem.ptr[i + 1] - t_rem.ptr[i];

  // Sizes of transposed remote blocks:
  // 1. Exchange rem_ptr
  std::vector<ptrdiff_t> rem_ptr(C.send.count() + 1);
  rem_ptr[0] = 0;

  for (size_t i = 0; i < C.send.nbr.size(); ++i) {
    ptrdiff_t beg = C.send.ptr[i];
    ptrdiff_t end = C.send.ptr[i + 1];

    recv_cnt_req[i] = comm.doIReceive(&rem_ptr[beg + 1], end - beg, C.send.nbr[i], tag_cnt);
  }

  for (size_t i = 0; i < C.recv.nbr.size(); ++i) {
    ptrdiff_t beg = C.recv.ptr[i];
    ptrdiff_t end = C.recv.ptr[i + 1];

    send_cnt_req[i] = comm.doISend(&row_size[beg], end - beg, C.recv.nbr[i], tag_cnt);
  }

  ARCCORE_ALINA_TIC("MPI Wait");
  comm.waitAll(recv_cnt_req);
  ARCCORE_ALINA_TOC("MPI Wait");
  std::partial_sum(rem_ptr.begin(), rem_ptr.end(), rem_ptr.begin());

  // 2. Start exchange of rem_col, rem_val
  std::vector<col_type> rem_col(rem_ptr.back());
  std::vector<value_type> rem_val(rem_ptr.back());

  for (size_t i = 0; i < C.send.nbr.size(); ++i) {
    ptrdiff_t rbeg = C.send.ptr[i];
    ptrdiff_t rend = C.send.ptr[i + 1];

    ptrdiff_t cbeg = rem_ptr[rbeg];
    ptrdiff_t cend = rem_ptr[rend];

    recv_col_req[i] = comm.doIReceive(&rem_col[cbeg], cend - cbeg, C.send.nbr[i], tag_col);
    recv_val_req[i] = comm.doIReceive(&rem_val[cbeg], cend - cbeg, C.send.nbr[i], tag_val);
  }

  for (size_t i = 0; i < C.recv.nbr.size(); ++i) {
    ptrdiff_t rbeg = C.recv.ptr[i];
    ptrdiff_t rend = C.recv.ptr[i + 1];

    ptrdiff_t cbeg = t_rem.ptr[rbeg];
    ptrdiff_t cend = t_rem.ptr[rend];

    send_col_req[i] = comm.doISend(&t_rem.col[cbeg], cend - cbeg, C.recv.nbr[i], tag_col);
    send_val_req[i] = comm.doISend(&t_rem.val[cbeg], cend - cbeg, C.recv.nbr[i], tag_val);
  }

  // 3. While rem_col and rem_val are in flight,
  //    start constructing our remote part:
  auto T_ptr = std::make_shared<build_matrix>();
  build_matrix& T_rem = *T_ptr;
  T_rem.set_size(nrows, 0, true);

  for (size_t i = 0; i < C.send.count(); ++i)
    T_rem.ptr[1 + C.send.col[i]] += rem_ptr[i + 1] - rem_ptr[i];

  T_rem.scan_row_sizes();
  T_rem.set_nonzeros();

  // 4. Finish rem_col and rem_val exchange, and
  //    finish contruction of our remote part.
  ARCCORE_ALINA_TIC("MPI Wait");
  comm.waitAll(recv_col_req);
  comm.waitAll(recv_val_req);
  ARCCORE_ALINA_TOC("MPI Wait");

  for (size_t i = 0; i < C.send.count(); ++i) {
    ptrdiff_t row = C.send.col[i];
    ptrdiff_t head = T_rem.ptr[row];

    for (ptrdiff_t j = rem_ptr[i]; j < rem_ptr[i + 1]; ++j, ++head) {
      T_rem.col[head] = rem_col[j];
      T_rem.val[head] = rem_val[j];
    }

    T_rem.ptr[row] = head;
  }

  std::rotate(T_rem.ptr.data(), T_rem.ptr.data() + nrows, T_rem.ptr.data() + nrows + 1);
  T_rem.ptr[0] = 0;

  ARCCORE_ALINA_TIC("MPI Wait");
  comm.waitAll(send_cnt_req);
  comm.waitAll(send_col_req);
  comm.waitAll(send_val_req);
  ARCCORE_ALINA_TOC("MPI Wait");

  ARCCORE_ALINA_TOC("MPI Transpose");

  return std::make_shared<DistributedMatrix<Backend>>(comm, transpose(A_loc), T_ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend>
std::shared_ptr<typename Backend::matrix>
remote_rows(const CommunicationPattern<Backend>& C,
            const DistributedMatrix<Backend>& B,
            bool need_values = true)
{
  typedef typename Backend::matrix build_matrix;

  static const int tag_ptr = 3001;
  static const int tag_col = 3002;
  static const int tag_val = 3003;

  ARCCORE_ALINA_TIC("remote_rows");
  mpi_communicator comm = C.mpi_comm();

  build_matrix& B_loc = *B.local();
  build_matrix& B_rem = *B.remote();
  ptrdiff_t B_beg = B.loc_col_shift();

  size_t nrecv = C.recv.nbr.size();
  size_t nsend = C.send.nbr.size();

  // Create blocked matrix to send to each domain
  // that needs data from us:
  UniqueArray<MessagePassing::Request> send_ptr_req(nsend);
  UniqueArray<MessagePassing::Request> send_col_req(nsend);
  UniqueArray<MessagePassing::Request> send_val_req(nsend);

  std::vector<build_matrix> send_rows(nsend);

  for (size_t k = 0; k < nsend; ++k) {
    ptrdiff_t beg = C.send.ptr[k];
    ptrdiff_t end = C.send.ptr[k + 1];

    build_matrix& m = send_rows[k];
    m.set_size(end - beg, 0, false);

    size_t nnz = 0;
    for (ptrdiff_t i = 0, ii = beg; ii < end; ++i, ++ii) {
      ptrdiff_t r = C.send.col[ii];

      ptrdiff_t w = (B_loc.ptr[r + 1] - B_loc.ptr[r]) + (B_rem.ptr[r + 1] - B_rem.ptr[r]);

      m.ptr[i] = w;
      nnz += w;
    }
    m.setNbNonZero(nnz);

    send_ptr_req[k] = comm.doISend(m.ptr.data(), m.nbRow(), C.send.nbr[k], tag_ptr);

    m.set_nonzeros(nnz, need_values);

    for (ptrdiff_t i = 0, ii = beg, head = 0; ii < end; ++i, ++ii) {
      ptrdiff_t r = C.send.col[ii];

      // Contribution of the local part:
      for (ptrdiff_t j = B_loc.ptr[r]; j < B_loc.ptr[r + 1]; ++j) {
        m.col[head] = B_loc.col[j] + B_beg;

        if (need_values)
          m.val[head] = B_loc.val[j];

        ++head;
      }

      // Contribution of the remote part:
      for (ptrdiff_t j = B_rem.ptr[r]; j < B_rem.ptr[r + 1]; ++j) {
        m.col[head] = B_rem.col[j];

        if (need_values)
          m.val[head] = B_rem.val[j];

        ++head;
      }
    }

    send_col_req[k] = comm.doISend(m.col.data(), m.nbNonZero(), C.send.nbr[k], tag_col);
    if (need_values)
      send_val_req[k] = comm.doISend(m.val.data(), m.nbNonZero(), C.send.nbr[k], tag_val);
  }

  // Receive rows of B in block format from our neighbors:
  UniqueArray<MessagePassing::Request> recv_ptr_req(nrecv);
  UniqueArray<MessagePassing::Request> recv_col_req(nrecv);
  UniqueArray<MessagePassing::Request> recv_val_req(nrecv);

  auto B_nbr = std::make_shared<build_matrix>();
  B_nbr->set_size(C.recv.count(), 0, false);
  B_nbr->ptr[0] = 0;

  for (size_t k = 0; k < nrecv; ++k) {
    ptrdiff_t beg = C.recv.ptr[k];
    ptrdiff_t end = C.recv.ptr[k + 1];

    recv_ptr_req[k] = comm.doIReceive(&B_nbr->ptr[beg + 1], end - beg, C.recv.nbr[k], tag_ptr);
  }

  ARCCORE_ALINA_TIC("MPI Wait");
  comm.waitAll(recv_ptr_req);
  ARCCORE_ALINA_TOC("MPI Wait");

  B_nbr->set_nonzeros(B_nbr->scan_row_sizes(), need_values);

  for (size_t k = 0; k < nrecv; ++k) {
    ptrdiff_t rbeg = C.recv.ptr[k];
    ptrdiff_t rend = C.recv.ptr[k + 1];

    ptrdiff_t cbeg = B_nbr->ptr[rbeg];
    ptrdiff_t cend = B_nbr->ptr[rend];

    recv_col_req[k] = comm.doIReceive(&B_nbr->col[cbeg], cend - cbeg, C.recv.nbr[k], tag_col);

    if (need_values)
      recv_val_req[k] = comm.doIReceive(&B_nbr->val[cbeg], cend - cbeg, C.recv.nbr[k], tag_val);
  }

  ARCCORE_ALINA_TIC("MPI Wait");
  comm.waitAll(send_ptr_req);
  comm.waitAll(send_col_req);
  comm.waitAll(recv_col_req);

  if (need_values) {
    comm.waitAll(send_val_req);
    comm.waitAll(recv_val_req);
  }
  ARCCORE_ALINA_TOC("MPI Wait");

  ARCCORE_ALINA_TOC("remote_rows");
  return B_nbr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend>
std::shared_ptr<DistributedMatrix<Backend>>
product(const DistributedMatrix<Backend>& A, const DistributedMatrix<Backend>& B)
{
  typedef typename Backend::value_type value_type;
  using build_matrix = Backend::matrix;
  typedef typename Backend::col_type col_type;
  ARCCORE_ALINA_TIC("product");

  const CommunicationPattern<Backend>& Acp = A.cpat();

  build_matrix& A_loc = *A.local();
  build_matrix& A_rem = *A.remote();
  build_matrix& B_loc = *B.local();
  build_matrix& B_rem = *B.remote();

  ptrdiff_t A_rows = A.loc_rows();
  ptrdiff_t B_cols = B.loc_cols();

  ptrdiff_t B_beg = B.loc_col_shift();
  ptrdiff_t B_end = B_beg + B_cols;

  auto b_nbr = remote_rows(Acp, B);
  build_matrix& B_nbr = *b_nbr;

  // Build mapping from global to local column numbers in the remote part of
  // the product matrix.
  std::vector<col_type> rem_cols(B_rem.nbNonZero() + B_nbr.nbNonZero());

  std::copy(B_nbr.col.data(), B_nbr.col.data() + B_nbr.nbNonZero(),
            std::copy(B_rem.col.data(), B_rem.col.data() + B_rem.nbNonZero(), rem_cols.begin()));

  std::sort(rem_cols.begin(), rem_cols.end());
  rem_cols.erase(std::unique(rem_cols.begin(), rem_cols.end()), rem_cols.end());

  ptrdiff_t n_rem_cols = 0;
  std::unordered_map<ptrdiff_t, int> rem_idx(2 * rem_cols.size());
  for (ptrdiff_t c : rem_cols) {
    if (c >= B_beg && c < B_end)
      continue;
    rem_idx[c] = n_rem_cols++;
  }

  // Build the product.
  auto c_loc = std::make_shared<build_matrix>();
  auto c_rem = std::make_shared<build_matrix>();

  build_matrix& C_loc = *c_loc;
  build_matrix& C_rem = *c_rem;

  C_loc.set_size(A_rows, B_cols, false);
  C_rem.set_size(A_rows, 0, false);

  C_loc.ptr[0] = 0;
  C_rem.ptr[0] = 0;

  ARCCORE_ALINA_TIC("analyze");
  arccoreParallelFor(0, A_rows, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
    std::vector<ptrdiff_t> loc_marker(B_end - B_beg, -1);
    std::vector<ptrdiff_t> rem_marker(n_rem_cols, -1);

    for (ptrdiff_t ia = begin; ia < (begin + size); ++ia) {
      ptrdiff_t loc_cols = 0;
      ptrdiff_t rem_cols = 0;

      for (ptrdiff_t ja = A_loc.ptr[ia], ea = A_loc.ptr[ia + 1]; ja < ea; ++ja) {
        ptrdiff_t ca = A_loc.col[ja];

        for (ptrdiff_t jb = B_loc.ptr[ca], eb = B_loc.ptr[ca + 1]; jb < eb; ++jb) {
          ptrdiff_t cb = B_loc.col[jb];

          if (loc_marker[cb] != ia) {
            loc_marker[cb] = ia;
            ++loc_cols;
          }
        }

        for (ptrdiff_t jb = B_rem.ptr[ca], eb = B_rem.ptr[ca + 1]; jb < eb; ++jb) {
          ptrdiff_t cb = rem_idx[B_rem.col[jb]];

          if (rem_marker[cb] != ia) {
            rem_marker[cb] = ia;
            ++rem_cols;
          }
        }
      }

      for (ptrdiff_t ja = A_rem.ptr[ia], ea = A_rem.ptr[ia + 1]; ja < ea; ++ja) {
        ptrdiff_t ca = Acp.local_index(A_rem.col[ja]);

        for (ptrdiff_t jb = B_nbr.ptr[ca], eb = B_nbr.ptr[ca + 1]; jb < eb; ++jb) {
          ptrdiff_t cb = B_nbr.col[jb];

          if (cb >= B_beg && cb < B_end) {
            cb -= B_beg;

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

      C_loc.ptr[ia + 1] = loc_cols;
      C_rem.ptr[ia + 1] = rem_cols;
    }
  });
  ARCCORE_ALINA_TOC("analyze");

  C_loc.set_nonzeros(C_loc.scan_row_sizes());
  C_rem.set_nonzeros(C_rem.scan_row_sizes());

  ARCCORE_ALINA_TIC("compute");
  arccoreParallelFor(0, A_rows, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
    std::vector<ptrdiff_t> loc_marker(B_end - B_beg, -1);
    std::vector<ptrdiff_t> rem_marker(n_rem_cols, -1);

    for (ptrdiff_t ia = begin; ia < (begin + size); ++ia) {
      ptrdiff_t loc_beg = C_loc.ptr[ia];
      ptrdiff_t rem_beg = C_rem.ptr[ia];
      ptrdiff_t loc_end = loc_beg;
      ptrdiff_t rem_end = rem_beg;

      for (ptrdiff_t ja = A_loc.ptr[ia], ea = A_loc.ptr[ia + 1]; ja < ea; ++ja) {
        ptrdiff_t ca = A_loc.col[ja];
        value_type va = A_loc.val[ja];

        for (ptrdiff_t jb = B_loc.ptr[ca], eb = B_loc.ptr[ca + 1]; jb < eb; ++jb) {
          ptrdiff_t cb = B_loc.col[jb];
          value_type vb = B_loc.val[jb];

          if (loc_marker[cb] < loc_beg) {
            loc_marker[cb] = loc_end;

            C_loc.col[loc_end] = cb;
            C_loc.val[loc_end] = va * vb;

            ++loc_end;
          }
          else {
            C_loc.val[loc_marker[cb]] += va * vb;
          }
        }

        for (ptrdiff_t jb = B_rem.ptr[ca], eb = B_rem.ptr[ca + 1]; jb < eb; ++jb) {
          ptrdiff_t gb = B_rem.col[jb];
          ptrdiff_t cb = rem_idx[gb];
          value_type vb = B_rem.val[jb];

          if (rem_marker[cb] < rem_beg) {
            rem_marker[cb] = rem_end;

            C_rem.col[rem_end] = gb;
            C_rem.val[rem_end] = va * vb;

            ++rem_end;
          }
          else {
            C_rem.val[rem_marker[cb]] += va * vb;
          }
        }
      }

      for (ptrdiff_t ja = A_rem.ptr[ia], ea = A_rem.ptr[ia + 1]; ja < ea; ++ja) {
        ptrdiff_t ca = Acp.local_index(A_rem.col[ja]);
        value_type va = A_rem.val[ja];

        for (ptrdiff_t jb = B_nbr.ptr[ca], eb = B_nbr.ptr[ca + 1]; jb < eb; ++jb) {
          ptrdiff_t gb = B_nbr.col[jb];
          value_type vb = B_nbr.val[jb];

          if (gb >= B_beg && gb < B_end) {
            ptrdiff_t cb = gb - B_beg;

            if (loc_marker[cb] < loc_beg) {
              loc_marker[cb] = loc_end;

              C_loc.col[loc_end] = cb;
              C_loc.val[loc_end] = va * vb;

              ++loc_end;
            }
            else {
              C_loc.val[loc_marker[cb]] += va * vb;
            }
          }
          else {
            ptrdiff_t cb = rem_idx[gb];

            if (rem_marker[cb] < rem_beg) {
              rem_marker[cb] = rem_end;

              C_rem.col[rem_end] = gb;
              C_rem.val[rem_end] = va * vb;

              ++rem_end;
            }
            else {
              C_rem.val[rem_marker[cb]] += va * vb;
            }
          }
        }
      }
    }
  });
  ARCCORE_ALINA_TOC("compute");
  ARCCORE_ALINA_TOC("product");

  return std::make_shared<DistributedMatrix<Backend>>(A.comm(), c_loc, c_rem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend, class T>
void scale(DistributedMatrix<Backend>& A, T s)
{
  using build_matrix = Backend::matrix;

  build_matrix& A_loc = *A.local();
  build_matrix& A_rem = *A.remote();

  ptrdiff_t n = A_loc.nbRow();

  arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
    for (ptrdiff_t i = begin; i < (begin + size); ++i) {
      for (ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i + 1]; j < e; ++j)
        A_loc.val[j] *= s;
      for (ptrdiff_t j = A_rem.ptr[i], e = A_rem.ptr[i + 1]; j < e; ++j)
        A_rem.val[j] *= s;
    }
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend>
void sort_rows(DistributedMatrix<Backend>& A)
{
  sort_rows(*A.local());
  sort_rows(*A.remote());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::backend
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend>
struct rows_impl<DistributedMatrix<Backend>>
{
  static size_t get(const DistributedMatrix<Backend>& A)
  {
    return A.loc_rows();
  }
};

template <class Backend, class Alpha, class Vec1, class Beta, class Vec2>
struct spmv_impl<Alpha, DistributedMatrix<Backend>, Vec1, Beta, Vec2>
{
  static void apply(Alpha alpha,
                    const DistributedMatrix<Backend>& A,
                    const Vec1& x, Beta beta, Vec2& y)
  {
    A.mul(alpha, x, beta, y);
  }
};

template <class Backend, class Vec1, class Vec2, class Vec3>
struct residual_impl<DistributedMatrix<Backend>, Vec1, Vec2, Vec3>
{
  static void apply(const Vec1& rhs,
                    const DistributedMatrix<Backend>& A,
                    const Vec2& x, Vec3& r)
  {
    A.residual(rhs, x, r);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Diagonal of the matrix
template <class Backend>
std::shared_ptr<numa_vector<typename Backend::value_type>>
diagonal(const DistributedMatrix<Backend>& A, bool invert = false)
{
  return diagonal(*A.local(), invert);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Estimate spectral radius of the matrix.
template <bool scale, class Backend>
typename math::scalar_of<typename Backend::value_type>::type
spectral_radius(const DistributedMatrix<Backend>& A, int power_iters = 0)
{
  ARCCORE_ALINA_TIC("spectral radius");
  typedef typename Backend::value_type value_type;
  typedef typename math::rhs_of<value_type>::type rhs_type;
  typedef typename math::scalar_of<value_type>::type scalar_type;
  typedef CSRMatrix<value_type> build_matrix;

  mpi_communicator comm = A.comm();

  const build_matrix& A_loc = *A.local();
  const build_matrix& A_rem = *A.remote();
  const CommunicationPattern<Backend>& C = A.cpat();

  const ptrdiff_t n = A_loc.nbRow();
  scalar_type radius = 0;

  if (power_iters <= 0) {
#pragma omp parallel
    {
      scalar_type emax = 0;
      value_type dia = math::identity<value_type>();

#pragma omp for nowait
      for (ptrdiff_t i = 0; i < n; ++i) {
        scalar_type s = 0;

        for (ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i + 1]; j < e; ++j) {
          ptrdiff_t c = A_loc.col[j];
          value_type v = A_loc.val[j];

          s += math::norm(v);

          if (scale && c == i)
            dia = v;
        }

        for (ptrdiff_t j = A_rem.ptr[i], e = A_rem.ptr[i + 1]; j < e; ++j)
          s += math::norm(A_rem.val[j]);

        if (scale)
          s *= math::norm(math::inverse(dia));

        emax = std::max(emax, s);
      }

#pragma omp critical
      radius = std::max(radius, emax);
    }
  }
  else {
    numa_vector<rhs_type> b0(n, false), b1(n, false);
    numa_vector<ptrdiff_t> rem_col(A_rem.nbNonZero(), false);

    // Fill the initial vector with random values.
    // Also extract the inverted matrix diagonal values.
    scalar_type b0_loc_norm = 0;

#pragma omp parallel
    {
#ifdef _OPENMP
      int tid = omp_get_thread_num();
      int nt = omp_get_max_threads();
#else
      int tid = 0;
      int nt = 1;
#endif
      std::mt19937 rng(comm.size * nt + tid);
      std::uniform_real_distribution<scalar_type> rnd(-1, 1);

      scalar_type t_norm = 0;

#pragma omp for nowait
      for (ptrdiff_t i = 0; i < n; ++i) {
        rhs_type v = math::constant<rhs_type>(rnd(rng));

        b0[i] = v;
        t_norm += math::norm(math::inner_product(v, v));

        for (ptrdiff_t j = A_rem.ptr[i], e = A_rem.ptr[i + 1]; j < e; ++j) {
          rem_col[j] = C.local_index(A_rem.col[j]);
        }
      }

#pragma omp critical
      b0_loc_norm += t_norm;
    }

    scalar_type b0_norm = comm.reduceSum(b0_loc_norm);

    // Normalize b0
    b0_norm = 1 / sqrt(b0_norm);
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; ++i) {
      b0[i] = b0_norm * b0[i];
    }

    std::vector<rhs_type> b0_send(C.send.count());
    std::vector<rhs_type> b0_recv(C.recv.count());

    for (size_t i = 0, m = C.send.count(); i < m; ++i)
      b0_send[i] = b0[C.send.col[i]];
    C.exchange(b0_send.data(), b0_recv.data());

    for (int iter = 0; iter < power_iters;) {
      // b1 = (D * A) * b0
      // b1_norm = ||b1||
      // radius = <b1,b0>
      scalar_type b1_loc_norm = 0;
      scalar_type loc_radius = 0;

#pragma omp parallel
      {
        scalar_type t_norm = 0;
        scalar_type t_radi = 0;
        value_type dia = math::identity<value_type>();

#pragma omp for nowait
        for (ptrdiff_t i = 0; i < n; ++i) {
          rhs_type s = math::zero<rhs_type>();

          for (ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i + 1]; j < e; ++j) {
            ptrdiff_t c = A_loc.col[j];
            value_type v = A_loc.val[j];
            if (scale && c == i)
              dia = v;
            s += v * b0[c];
          }

          for (ptrdiff_t j = A_rem.ptr[i], e = A_rem.ptr[i + 1]; j < e; ++j)
            s += A_rem.val[j] * b0_recv[rem_col[j]];

          if (scale)
            s = math::inverse(dia) * s;

          t_norm += math::norm(math::inner_product(s, s));
          t_radi += math::norm(math::inner_product(s, b0[i]));

          b1[i] = s;
        }

#pragma omp critical
        {
          b1_loc_norm += t_norm;
          loc_radius += t_radi;
        }
      }

      radius = comm.reduceSum(loc_radius);

      if (++iter < power_iters) {
        scalar_type b1_norm;
        b1_norm = comm.reduceSum(b1_loc_norm);

        // b0 = b1 / b1_norm
        b1_norm = 1 / sqrt(b1_norm);
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
          b0[i] = b1_norm * b1[i];
        }

        for (size_t i = 0, m = C.send.count(); i < m; ++i)
          b0_send[i] = b0[C.send.col[i]];
        C.exchange(b0_send.data(), b0_recv.data());
      }
    }
  }
  ARCCORE_ALINA_TOC("spectral radius");

  return radius < 0 ? static_cast<scalar_type>(2) : radius;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
