// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DistributedSchurPressureCorrection.h                        (C) 2000-2026 */
/*                                                                           */
/* Distributed Schur complement pressure correction preconditioner.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_MPI_DISTRIBUTEDSCHURPRESSURECORRECTION_H
#define ARCCORE_ALINA_MPI_DISTRIBUTEDSCHURPRESSURECORRECTION_H
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

#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/DistributedInnerProduct.h"
#include "arccore/alina/DistributedMatrix.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Distributed Schur complement pressure correction preconditioner.
 */
template <class USolver, class PSolver>
class DistributedSchurPressureCorrection
{
  using USolverBackendType = typename USolver::backend_type;
  using PSolverBackendType = typename PSolver::backend_type;

  static_assert(std::is_same<USolverBackendType, PSolverBackendType>::value,
                "Backends for pressure and flow preconditioners should coincide!");

 public:

  using backend_type = detail::common_scalar_backend<USolverBackendType, PSolverBackendType>::type;
  using BackendType = backend_type;

  typedef typename backend_type::value_type value_type;
  typedef typename math::scalar_of<value_type>::type scalar_type;
  typedef typename backend_type::matrix bmatrix;
  typedef typename backend_type::vector vector;
  typedef typename backend_type::params backend_params;

  typedef DistributedMatrix<backend_type> matrix;

  typedef typename BuiltinBackend<value_type>::matrix build_matrix;

  struct params
  {
    typedef typename USolver::params usolver_params;
    typedef typename PSolver::params psolver_params;

    usolver_params usolver;
    psolver_params psolver;

    std::vector<char> pmask;

    // Variant of block preconditioner to use in apply()
    // 1: schur pressure correction:
    //      S p = fp - Kpu Kuu^-1 fu
    //      Kuu u = fu - Kup p
    // 2: Block triangular:
    //      S p = fp
    //      Kuu u = fu - Kup p
    int type = 1;

    // Approximate Kuu^-1 with inverted diagonal of Kuu during
    // construction of matrix-less Schur complement.
    // When false, USolver is used instead.
    bool approx_schur = false;

    // Use 1/sum_j(abs(Kuu_{i,j})) instead of dia(Kuu)^-1
    // as approximation for the Kuu^-1 (as in SIMPLEC algorithm)
    bool simplec_dia = true;

    int verbose = 0;

    params() = default;

    params(const Alina::PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, usolver)
    , ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, psolver)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, type)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, approx_schur)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, simplec_dia)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, verbose)
    {
      size_t n = 0;

      n = p.get("pmask_size", n);

      precondition(n > 0, "Error in schur_complement parameters: pmask_size is not set");

      if (p.count("pmask_pattern")) {
        pmask.resize(n, 0);

        std::string pattern = p.get("pmask_pattern", std::string());
        switch (pattern[0]) {
        case '%': {
          int start = std::atoi(pattern.substr(1).c_str());
          int stride = std::atoi(pattern.substr(3).c_str());
          for (size_t i = start; i < n; i += stride)
            pmask[i] = 1;
        } break;
        case '<': {
          size_t m = std::atoi(pattern.c_str() + 1);
          for (size_t i = 0; i < std::min(m, n); ++i)
            pmask[i] = 1;
        } break;
        case '>': {
          size_t m = std::atoi(pattern.c_str() + 1);
          for (size_t i = m; i < n; ++i)
            pmask[i] = 1;
        } break;
        default:
          Alina::precondition(false, "Unknown pattern in pmask_pattern");
        }
      }
      else if (p.count("pmask")) {
        void* pm = 0;
        pm = p.get("pmask", pm);
        pmask.assign(static_cast<char*>(pm), static_cast<char*>(pm) + n);
      }
      else {
        ARCANE_FATAL("Error in schur_complement parameters:  neither pmask_pattern, nor pmask is set");
      }

      p.check_params({ "usolver", "psolver", "type", "approx_schur", "simplec_dia", "pmask_size", "verbose" },
                     { "pmask", "pmask_pattern" });
    }

    void get(PropertyTree& p, const std::string& path = "") const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, usolver);
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, psolver);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, type);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, approx_schur);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, simplec_dia);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, verbose);
    }
  };

  template <class Matrix>
  DistributedSchurPressureCorrection(mpi_communicator comm,
                                     const Matrix& K,
                                     const params& prm = params(),
                                     const backend_params& bprm = backend_params())
  : prm(prm)
  , comm(comm)
  {
    this->K = std::make_shared<matrix>(comm, K, backend::nbRow(K));
    init(bprm);
  }

  DistributedSchurPressureCorrection(mpi_communicator comm,
                                     std::shared_ptr<matrix> K,
                                     const params& prm = params(),
                                     const backend_params& bprm = backend_params())
  : prm(prm)
  , comm(comm)
  , K(K)
  {
    init(bprm);
  }

  void init(const backend_params& bprm)
  {
    using std::make_shared;
    using std::make_tuple;
    using std::shared_ptr;
    using std::tie;

    auto _K_loc = K->local();
    auto _K_rem = K->remote();

    build_matrix& K_loc = *_K_loc;
    build_matrix& K_rem = *_K_rem;

    ptrdiff_t n = K->loc_rows();

    // Count pressure and flow variables.
    ARCCORE_ALINA_TIC("count pressure/flow vars");
    std::vector<ptrdiff_t> idx(n);
    ptrdiff_t np = 0, nu = 0;

    for (ptrdiff_t i = 0; i < n; ++i)
      idx[i] = (prm.pmask[i] ? np++ : nu++);
    ARCCORE_ALINA_TOC("count pressure/flow vars");

    ARCCORE_ALINA_TIC("setup communication");
    // We know what points each of our neighbors needs from us;
    // and we know if those points are pressure or flow.
    // We can immediately provide them with our renumbering scheme.
    std::vector<ptrdiff_t> pdomain = comm.exclusive_sum(np);
    std::vector<ptrdiff_t> udomain = comm.exclusive_sum(nu);
    ptrdiff_t p_beg = pdomain[comm.rank];
    ptrdiff_t u_beg = udomain[comm.rank];

    const CommPattern& C = this->K->cpat();
    ptrdiff_t nsend = C.send.count(), nrecv = C.recv.count();
    std::vector<char> smask(nsend), rmask(nrecv);
    std::vector<ptrdiff_t> s_idx(nsend), r_idx(nrecv);

    for (ptrdiff_t i = 0; i < nsend; ++i) {
      ptrdiff_t c = C.send.col[i];
      smask[i] = prm.pmask[c];
      s_idx[i] = idx[c] + (smask[i] ? p_beg : u_beg);
    }

    C.exchange(&smask[0], &rmask[0]);
    C.exchange(&s_idx[0], &r_idx[0]);
    ARCCORE_ALINA_TOC("setup communication");

    // Fill the subblocks of the system matrix.
    // K_rem->col may be used as direct indices into rmask and r_idx.
    ARCCORE_ALINA_TIC("schur blocks");
    this->K->move_to_backend(bprm);

    auto Kpp_loc = make_shared<build_matrix>();
    auto Kpp_rem = make_shared<build_matrix>();
    auto Kuu_loc = make_shared<build_matrix>();
    auto Kuu_rem = make_shared<build_matrix>();

    auto Kpu_loc = make_shared<build_matrix>();
    auto Kpu_rem = make_shared<build_matrix>();
    auto Kup_loc = make_shared<build_matrix>();
    auto Kup_rem = make_shared<build_matrix>();

    Kpp_loc->set_size(np, np, true);
    Kpp_rem->set_size(np, 0, true);

    Kuu_loc->set_size(nu, nu, true);
    Kuu_rem->set_size(nu, 0, true);

    Kpu_loc->set_size(np, nu, true);
    Kpu_rem->set_size(np, 0, true);

    Kup_loc->set_size(nu, np, true);
    Kup_rem->set_size(nu, 0, true);

    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      for (ptrdiff_t i = begin; i < (begin + size); ++i) {
        ptrdiff_t ci = idx[i];
        char pi = prm.pmask[i];

        for (auto a = backend::row_begin(K_loc, i); a; ++a) {
          char pj = prm.pmask[a.col()];

          if (pi) {
            if (pj) {
              ++Kpp_loc->ptr[ci + 1];
            }
            else {
              ++Kpu_loc->ptr[ci + 1];
            }
          }
          else {
            if (pj) {
              ++Kup_loc->ptr[ci + 1];
            }
            else {
              ++Kuu_loc->ptr[ci + 1];
            }
          }
        }

        for (auto a = backend::row_begin(K_rem, i); a; ++a) {
          char pj = rmask[a.col()];

          if (pi) {
            if (pj) {
              ++Kpp_rem->ptr[ci + 1];
            }
            else {
              ++Kpu_rem->ptr[ci + 1];
            }
          }
          else {
            if (pj) {
              ++Kup_rem->ptr[ci + 1];
            }
            else {
              ++Kuu_rem->ptr[ci + 1];
            }
          }
        }
      }
    });

    Kpp_loc->set_nonzeros(Kpp_loc->scan_row_sizes());
    Kpp_rem->set_nonzeros(Kpp_rem->scan_row_sizes());

    Kuu_loc->set_nonzeros(Kuu_loc->scan_row_sizes());
    Kuu_rem->set_nonzeros(Kuu_rem->scan_row_sizes());

    Kpu_loc->set_nonzeros(Kpu_loc->scan_row_sizes());
    Kpu_rem->set_nonzeros(Kpu_rem->scan_row_sizes());

    Kup_loc->set_nonzeros(Kup_loc->scan_row_sizes());
    Kup_rem->set_nonzeros(Kup_rem->scan_row_sizes());

    // Fill subblocks of the system matrix.
    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      for (ptrdiff_t i = begin; i < (begin + size); ++i) {
        ptrdiff_t ci = idx[i];
        char pi = prm.pmask[i];

        ptrdiff_t pp_loc_head = 0, pp_rem_head = 0;
        ptrdiff_t uu_loc_head = 0, uu_rem_head = 0;
        ptrdiff_t pu_loc_head = 0, pu_rem_head = 0;
        ptrdiff_t up_loc_head = 0, up_rem_head = 0;

        if (pi) {
          pp_loc_head = Kpp_loc->ptr[ci];
          pp_rem_head = Kpp_rem->ptr[ci];
          pu_loc_head = Kpu_loc->ptr[ci];
          pu_rem_head = Kpu_rem->ptr[ci];
        }
        else {
          uu_loc_head = Kuu_loc->ptr[ci];
          uu_rem_head = Kuu_rem->ptr[ci];
          up_loc_head = Kup_loc->ptr[ci];
          up_rem_head = Kup_rem->ptr[ci];
        }

        for (auto a = backend::row_begin(K_loc, i); a; ++a) {
          ptrdiff_t j = a.col();
          value_type v = a.value();
          char pj = prm.pmask[j];
          ptrdiff_t cj = idx[j];

          if (pi) {
            if (pj) {
              Kpp_loc->col[pp_loc_head] = cj;
              Kpp_loc->val[pp_loc_head] = v;
              ++pp_loc_head;
            }
            else {
              Kpu_loc->col[pu_loc_head] = cj;
              Kpu_loc->val[pu_loc_head] = v;
              ++pu_loc_head;
            }
          }
          else {
            if (pj) {
              Kup_loc->col[up_loc_head] = cj;
              Kup_loc->val[up_loc_head] = v;
              ++up_loc_head;
            }
            else {
              Kuu_loc->col[uu_loc_head] = cj;
              Kuu_loc->val[uu_loc_head] = v;
              ++uu_loc_head;
            }
          }
        }

        for (auto a = backend::row_begin(K_rem, i); a; ++a) {
          ptrdiff_t j = a.col();
          value_type v = a.value();
          char pj = rmask[j];
          ptrdiff_t cj = r_idx[j];

          if (pi) {
            if (pj) {
              Kpp_rem->col[pp_rem_head] = cj;
              Kpp_rem->val[pp_rem_head] = v;
              ++pp_rem_head;
            }
            else {
              Kpu_rem->col[pu_rem_head] = cj;
              Kpu_rem->val[pu_rem_head] = v;
              ++pu_rem_head;
            }
          }
          else {
            if (pj) {
              Kup_rem->col[up_rem_head] = cj;
              Kup_rem->val[up_rem_head] = v;
              ++up_rem_head;
            }
            else {
              Kuu_rem->col[uu_rem_head] = cj;
              Kuu_rem->val[uu_rem_head] = v;
              ++uu_rem_head;
            }
          }
        }
      }
    });

    auto Kpp = std::make_shared<matrix>(comm, Kpp_loc, Kpp_rem);
    auto Kuu = std::make_shared<matrix>(comm, Kuu_loc, Kuu_rem);

    Kpu = make_shared<matrix>(comm, Kpu_loc, Kpu_rem);
    Kup = make_shared<matrix>(comm, Kup_loc, Kup_rem);

    Kpu->move_to_backend(bprm);
    Kup->move_to_backend(bprm);
    ARCCORE_ALINA_TOC("schur blocks");

    ARCCORE_ALINA_TIC("usolver")
    U = make_shared<USolver>(comm, Kuu, prm.usolver, bprm);
    ARCCORE_ALINA_TOC("usolver")
    ARCCORE_ALINA_TIC("psolver")
    P = make_shared<PSolver>(comm, Kpp, prm.psolver, bprm);
    ARCCORE_ALINA_TOC("psolver")

    ARCCORE_ALINA_TIC("other");
    rhs_u = backend_type::create_vector(nu, bprm);
    rhs_p = backend_type::create_vector(np, bprm);

    u = backend_type::create_vector(nu, bprm);
    p = backend_type::create_vector(np, bprm);

    tmp = backend_type::create_vector(nu, bprm);

    if (prm.approx_schur) {
      std::shared_ptr<numa_vector<value_type>> Kuu_dia;
      ARCCORE_ALINA_TIC("Kuu diagonal");
      if (prm.simplec_dia) {
        Kuu_dia = std::make_shared<numa_vector<value_type>>(nu, false);
        arccoreParallelFor(0, nu, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
          for (ptrdiff_t i = begin; i < (begin + size); ++i) {
            value_type s = math::zero<value_type>();
            for (ptrdiff_t j = Kuu_loc->ptr[i], e = Kuu_loc->ptr[i + 1]; j < e; ++j) {
              s += math::norm(Kuu_loc->val[j]);
            }
            for (ptrdiff_t j = Kuu_rem->ptr[i], e = Kuu_rem->ptr[i + 1]; j < e; ++j) {
              s += math::norm(Kuu_rem->val[j]);
            }
            (*Kuu_dia)[i] = math::inverse(s);
          }
        });
      }
      else {
        Kuu_dia = diagonal(*Kuu_loc, /*invert = */ true);
      }

      M = backend_type::copy_vector(Kuu_dia, bprm);
      ARCCORE_ALINA_TOC("Kuu diagonal");
    }

    // Scatter/Gather matrices
    ARCCORE_ALINA_TIC("scatter/gather");
    auto x2u = std::make_shared<build_matrix>();
    auto x2p = std::make_shared<build_matrix>();
    auto u2x = std::make_shared<build_matrix>();
    auto p2x = std::make_shared<build_matrix>();

    x2u->set_size(nu, n, true);
    x2p->set_size(np, n, true);
    u2x->set_size(n, nu, true);
    p2x->set_size(n, np, true);

    {
      ptrdiff_t x2u_head = 0, x2u_idx = 0;
      ptrdiff_t x2p_head = 0, x2p_idx = 0;
      ptrdiff_t u2x_head = 0, u2x_idx = 0;
      ptrdiff_t p2x_head = 0, p2x_idx = 0;

      for (ptrdiff_t i = 0; i < n; ++i) {
        if (prm.pmask[i]) {
          x2p->ptr[++x2p_idx] = ++x2p_head;
          ++p2x_head;
        }
        else {
          x2u->ptr[++x2u_idx] = ++x2u_head;
          ++u2x_head;
        }

        p2x->ptr[++p2x_idx] = p2x_head;
        u2x->ptr[++u2x_idx] = u2x_head;
      }
    }

    x2u->set_nonzeros();
    x2p->set_nonzeros();
    u2x->set_nonzeros();
    p2x->set_nonzeros();

    {
      ptrdiff_t x2u_head = 0;
      ptrdiff_t x2p_head = 0;
      ptrdiff_t u2x_head = 0;
      ptrdiff_t p2x_head = 0;

      for (ptrdiff_t i = 0; i < n; ++i) {
        ptrdiff_t j = idx[i];

        if (prm.pmask[i]) {
          x2p->col[x2p_head] = i;
          x2p->val[x2p_head] = math::identity<value_type>();
          ++x2p_head;

          p2x->col[p2x_head] = j;
          p2x->val[p2x_head] = math::identity<value_type>();
          ++p2x_head;
        }
        else {
          x2u->col[x2u_head] = i;
          x2u->val[x2u_head] = math::identity<value_type>();
          ++x2u_head;

          u2x->col[u2x_head] = j;
          u2x->val[u2x_head] = math::identity<value_type>();
          ++u2x_head;
        }
      }
    }

    this->x2u = backend_type::copy_matrix(x2u, bprm);
    this->x2p = backend_type::copy_matrix(x2p, bprm);
    this->u2x = backend_type::copy_matrix(u2x, bprm);
    this->p2x = backend_type::copy_matrix(p2x, bprm);
    ARCCORE_ALINA_TOC("scatter/gather");
  }

  std::shared_ptr<matrix> system_matrix_ptr() const
  {
    return K;
  }

  const matrix& system_matrix() const
  {
    return *K;
  }

  template <class Vec1, class Vec2>
  void apply(const Vec1& rhs, Vec2&& x) const
  {
    const auto one = math::identity<scalar_type>();
    const auto zero = math::zero<scalar_type>();

    ARCCORE_ALINA_TIC("split variables");
    backend::spmv(one, *x2u, rhs, zero, *rhs_u);
    backend::spmv(one, *x2p, rhs, zero, *rhs_p);
    ARCCORE_ALINA_TOC("split variables");

    if (prm.type == 1) {
      // Ai u = rhs_u
      ARCCORE_ALINA_TIC("solve U");
      backend::clear(*u);
      report("U1", (*U)(*rhs_u, *u));
      ARCCORE_ALINA_TOC("solve U");

      // rhs_p -= Kpu u
      ARCCORE_ALINA_TIC("solve P");
      backend::spmv(-one, *Kpu, *u, one, *rhs_p);

      // S p = rhs_p
      backend::clear(*p);
      report("P", (*P)(*this, *rhs_p, *p));
      ARCCORE_ALINA_TOC("solve P");

      // rhs_u -= Kup p
      ARCCORE_ALINA_TIC("Update U");
      backend::spmv(-one, *Kup, *p, one, *rhs_u);

      // Ai u = rhs_u
      backend::clear(*u);
      report("U2", (*U)(*rhs_u, *u));
      ARCCORE_ALINA_TOC("Update U");
    }
    else if (prm.type == 2) {
      // S p = rhs_p
      ARCCORE_ALINA_TIC("solve P");
      backend::clear(*p);
      report("P", (*P)(*this, *rhs_p, *p));
      ARCCORE_ALINA_TOC("solve P");

      // Ai u = fu - Kup p
      ARCCORE_ALINA_TIC("solve U");
      backend::spmv(-one, *Kup, *p, one, *rhs_u);
      backend::clear(*u);
      report("U", (*U)(*rhs_u, *u));
      ARCCORE_ALINA_TOC("solve U");
    }

    ARCCORE_ALINA_TIC("merge variables");
    backend::spmv(one, *u2x, *u, zero, x);
    backend::spmv(one, *p2x, *p, one, x);
    ARCCORE_ALINA_TOC("merge variables");
  }

  template <class Alpha, class Vec1, class Beta, class Vec2>
  void spmv(Alpha alpha, const Vec1& x, Beta beta, Vec2& y) const
  {
    const auto one = math::identity<scalar_type>();
    const auto zero = math::zero<scalar_type>();

    // y = beta y + alpha S x, where S = Kpp - Kpu Kuu^-1 Kup
    ARCCORE_ALINA_TIC("matrix-free spmv");
    backend::spmv(alpha, P->system_matrix(), x, beta, y);

    backend::spmv(one, *Kup, x, zero, *tmp);

    if (prm.approx_schur) {
      backend::vmul(one, *M, *tmp, zero, *u);
    }
    else {
      backend::clear(*u);
      (*U)(*tmp, *u);
    }
    backend::spmv(-alpha, *Kpu, *u, one, y);
    ARCCORE_ALINA_TOC("matrix-free spmv");
  }

 public:

  params prm;

 private:

  typedef CommunicationPattern<backend_type> CommPattern;
  mpi_communicator comm;

  std::shared_ptr<bmatrix> x2p, x2u, p2x, u2x;
  std::shared_ptr<matrix> K, Kpu, Kup;
  std::shared_ptr<vector> rhs_u, rhs_p, u, p, tmp;
  std::shared_ptr<typename backend_type::matrix_diagonal> M;

  std::shared_ptr<USolver> U;
  std::shared_ptr<PSolver> P;

#ifdef ARCCORE_ALINA_DEBUG
  void report(const std::string& name, const SolverResult& sr) const
  {
    if (comm.rank == 0 && prm.report >= 1) {
      std::cout << name << " (" << sr.nbIteration() << ", " << sr.residual() << ")\n";
    }
  }
#else
  void report(const std::string&, const SolverResult&) const
  {
  }
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::backend
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class US, class PS, class Alpha, class Beta, class Vec1, class Vec2>
struct spmv_impl<Alpha, DistributedSchurPressureCorrection<US, PS>, Vec1, Beta, Vec2>
{
  static void apply(Alpha alpha, const DistributedSchurPressureCorrection<US, PS>& A, const Vec1& x, Beta beta, Vec2& y)
  {
    A.spmv(alpha, x, beta, y);
  }
};

template <class US, class PS, class Vec1, class Vec2, class Vec3>
struct residual_impl<DistributedSchurPressureCorrection<US, PS>, Vec1, Vec2, Vec3>
{
  static void apply(const Vec1& rhs, const DistributedSchurPressureCorrection<US, PS>& A, const Vec2& x, Vec3& r)
  {
    backend::copy(rhs, r);
    A.spmv(-1, x, 1, r);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina::backend

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
