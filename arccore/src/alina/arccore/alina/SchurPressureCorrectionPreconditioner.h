// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SchurPressureCorrectionPreconditioner.h                     (C) 2000-2026 */
/*                                                                           */
/* Schur-complement pressure correction preconditioning scheme.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_PRECONDITIONERSCHURPRESSURECORRECTION_H
#define ARCCORE_ALINA_PRECONDITIONERSCHURPRESSURECORRECTION_H
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
/*
 * This algorithm is based on the following articles.
 *
 * [1] Elman, Howard, et al. "A taxonomy and comparison of parallel block
 *     multi-level preconditioners for the incompressible Navier–Stokes
 *     equations." Journal of Computational Physics 227.3 (2008): 1790-1808.
 * [2] Gmeiner, Björn, et al. "A quantitative performance analysis for Stokes
 *     solvers at the extreme scale." arXiv preprint arXiv:1511.02134 (2015).
 * [3] Vincent, C., and R. Boyer. "A preconditioned conjugate gradient
 *     Uzawa‐type method for the solution of the Stokes problem by mixed Q1–P0
 *     stabilized finite elements." International journal for numerical methods
 *     in fluids 14.3 (1992): 289-298.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/IO.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::preconditioner
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Schur-complement pressure correction preconditioner
template <class USolver, class PSolver>
class SchurPressureCorrectionPreconditioner
{
  static_assert(backend::backends_compatible<typename USolver::backend_type,
                                             typename PSolver::backend_type>::value,
                "Backends for pressure and flow preconditioners should coincide!");

 public:

  typedef typename detail::common_scalar_backend<typename USolver::backend_type,
                                                 typename PSolver::backend_type>::type backend_type;

  typedef typename backend_type::value_type value_type;
  typedef typename backend_type::col_type col_type;
  typedef typename backend_type::ptr_type ptr_type;
  typedef typename math::scalar_of<value_type>::type scalar_type;
  typedef typename backend_type::matrix matrix;
  typedef typename backend_type::vector vector;
  typedef typename backend_type::params backend_params;

  typedef typename BuiltinBackend<value_type, col_type, ptr_type>::matrix build_matrix;

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
    int type;

    // Approximate Kuu^-1 with inverted diagonal of Kuu during
    // construction of matrix-less Schur complement.
    // When false, USolver is used instead.
    bool approx_schur;

    // Adjust preconditioner matrix for the Schur complement system.
    // That is, use
    //   Kpp                                 when adjust_p == 0,
    //   Kpp - dia(Kpu * dia(Kuu)^-1 * Kup)  when adjust_p == 1,
    //   Kpp - Kpu * dia(Kuu)^-1 * Kup       when adjust_p == 2
    int adjust_p;

    // Use 1/sum_j(abs(Kuu_{i,j})) instead of dia(Kuu)^-1
    // as approximation for the Kuu^-1 (as in SIMPLEC algorithm)
    bool simplec_dia;

    int verbose;

    params()
    : type(1)
    , approx_schur(false)
    , adjust_p(1)
    , simplec_dia(true)
    , verbose(0)
    {}

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, usolver)
    , ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, psolver)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, type)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, approx_schur)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, adjust_p)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, simplec_dia)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, verbose)
    {
      size_t n = 0;

      n = p.get("pmask_size", n);

      precondition(n > 0,
                   "Error in schur_complement parameters: "
                   "pmask_size is not set");

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
          precondition(false, "Unknown pattern in pmask_pattern");
        }
      }
      else if (p.count("pmask")) {
        void* pm = 0;
        pm = p.get("pmask", pm);
        pmask.assign(static_cast<char*>(pm), static_cast<char*>(pm) + n);
      }
      else {
        precondition(false,
                     "Error in schur_complement parameters: "
                     "neither pmask_pattern, nor pmask is set");
      }

      p.check_params({ "usolver", "psolver", "type", "approx_schur", "adjust_p", "simplec_dia", "pmask_size", "verbose" },
                   { "pmask", "pmask_pattern" });
    }

    void get(PropertyTree& p, const std::string& path = "") const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, usolver);
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, psolver);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, type);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, approx_schur);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, adjust_p);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, simplec_dia);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, verbose);
    }
  };

  template <class Matrix>
  SchurPressureCorrectionPreconditioner(const Matrix& K,
                                        const params& prm = params(),
                                        const backend_params& bprm = backend_params())
  : prm(prm)
  , n(backend::nbRow(K))
  , np(0)
  , nu(0)
  {
    init(std::make_shared<build_matrix>(K), bprm);
  }

  SchurPressureCorrectionPreconditioner(std::shared_ptr<build_matrix> K,
                                        const params& prm = params(),
                                        const backend_params& bprm = backend_params())
  : prm(prm)
  , n(backend::nbRow(*K))
  , np(0)
  , nu(0)
  {
    init(K, bprm);
  }

  template <class Vec1, class Vec2>
  void apply(const Vec1& rhs, Vec2&& x) const
  {
    const auto one = math::identity<scalar_type>();
    const auto zero = math::zero<scalar_type>();

    backend::spmv(one, *x2u, rhs, zero, *rhs_u);
    backend::spmv(one, *x2p, rhs, zero, *rhs_p);

    if (prm.type == 1) {
      // Kuu u = rhs_u
      backend::clear(*u);
      report("U1", (*U)(*rhs_u, *u));

      // rhs_p -= Kpu u
      backend::spmv(-one, *Kpu, *u, one, *rhs_p);

      // S p = rhs_p
      backend::clear(*p);
      report("P1", (*P)(*this, *rhs_p, *p));

      // rhs_u -= Kup p
      backend::spmv(-one, *Kup, *p, one, *rhs_u);

      // Kuu u = rhs_u
      backend::clear(*u);
      report("U2", (*U)(*rhs_u, *u));
    }
    else if (prm.type == 2) {
      // S p = fp
      backend::clear(*p);
      report("P", (*P)(*this, *rhs_p, *p));

      // Kuu u = fu - Kup p
      backend::spmv(-one, *Kup, *p, one, *rhs_u);
      backend::clear(*u);
      report("U", (*U)(*rhs_u, *u));
    }

    backend::spmv(one, *u2x, *u, zero, x);
    backend::spmv(one, *p2x, *p, one, x);
  }

  template <class Alpha, class Vec1, class Beta, class Vec2>
  void spmv(Alpha alpha, const Vec1& x, Beta beta, Vec2& y) const
  {
    const auto one = math::identity<scalar_type>();
    const auto zero = math::zero<scalar_type>();

    // y = beta y + alpha S x, where S = Kpp - Kpu Kuu^-1 Kup
    if (prm.adjust_p == 1) {
      backend::spmv(alpha, P->system_matrix(), x, beta, y);
      backend::vmul(alpha, *Ld, x, one, y);
    }
    else if (prm.adjust_p == 2) {
      backend::spmv(alpha, *Lm, x, beta, y);
    }
    else {
      backend::spmv(alpha, P->system_matrix(), x, beta, y);
    }

    backend::spmv(one, *Kup, x, zero, *tmp);

    if (prm.approx_schur) {
      backend::vmul(one, *M, *tmp, zero, *u);
    }
    else {
      backend::clear(*u);
      (*U)(*tmp, *u);
    }

    backend::spmv(-alpha, *Kpu, *u, one, y);
  }

  std::shared_ptr<matrix> system_matrix_ptr() const
  {
    return K;
  }

  const matrix& system_matrix() const
  {
    return *K;
  }

  size_t bytes() const
  {
    size_t b = 0;

    b += backend::bytes(*K);
    b += backend::bytes(*Kup);
    b += backend::bytes(*Kpu);
    b += backend::bytes(*x2u);
    b += backend::bytes(*x2p);
    b += backend::bytes(*u2x);
    b += backend::bytes(*p2x);
    b += backend::bytes(*rhs_u);
    b += backend::bytes(*rhs_p);
    b += backend::bytes(*u);
    b += backend::bytes(*p);
    b += backend::bytes(*tmp);
    b += backend::bytes(*U);
    b += backend::bytes(*P);

    if (M)
      b += backend::bytes(*M);
    if (Ld)
      b += backend::bytes(*Ld);
    if (Lm)
      b += backend::bytes(*Lm);

    return b;
  }

  params prm;

 private:

  size_t n, np, nu;

  std::shared_ptr<matrix> K, Lm, Kup, Kpu, x2u, x2p, u2x, p2x;
  std::shared_ptr<vector> rhs_u, rhs_p, u, p, tmp;
  std::shared_ptr<typename backend_type::matrix_diagonal> M;
  std::shared_ptr<typename backend_type::matrix_diagonal> Ld;

  std::shared_ptr<USolver> U;
  std::shared_ptr<PSolver> P;

  void init(const std::shared_ptr<build_matrix>& K, const backend_params& bprm)
  {
    this->K = backend_type::copy_matrix(K, bprm);

    // Extract matrix subblocks.
    auto Kuu = std::make_shared<build_matrix>();
    auto Kpu = std::make_shared<build_matrix>();
    auto Kup = std::make_shared<build_matrix>();
    auto Kpp = std::make_shared<build_matrix>();

    std::vector<ptrdiff_t> idx(n);

    for (size_t i = 0; i < n; ++i)
      idx[i] = (prm.pmask[i] ? np++ : nu++);

    Kuu->set_size(nu, nu, true);
    Kup->set_size(nu, np, true);
    Kpu->set_size(np, nu, true);
    Kpp->set_size(np, np, true);

    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      for (ptrdiff_t i = begin; i < (begin + size); ++i) {
        ptrdiff_t ci = idx[i];
        char pi = prm.pmask[i];
        for (auto k = backend::row_begin(*K, i); k; ++k) {
          char pj = prm.pmask[k.col()];

          if (pi) {
            if (pj) {
              ++Kpp->ptr[ci + 1];
            }
            else {
              ++Kpu->ptr[ci + 1];
            }
          }
          else {
            if (pj) {
              ++Kup->ptr[ci + 1];
            }
            else {
              ++Kuu->ptr[ci + 1];
            }
          }
        }
      }
    });

    Kuu->set_nonzeros(Kuu->scan_row_sizes());
    Kup->set_nonzeros(Kup->scan_row_sizes());
    Kpu->set_nonzeros(Kpu->scan_row_sizes());
    Kpp->set_nonzeros(Kpp->scan_row_sizes());

    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      for (ptrdiff_t i = begin; i < (begin + size); ++i) {
        ptrdiff_t ci = idx[i];
        char pi = prm.pmask[i];

        ptrdiff_t uu_head = 0, up_head = 0, pu_head = 0, pp_head = 0;

        if (pi) {
          pu_head = Kpu->ptr[ci];
          pp_head = Kpp->ptr[ci];
        }
        else {
          uu_head = Kuu->ptr[ci];
          up_head = Kup->ptr[ci];
        }

        for (auto k = backend::row_begin(*K, i); k; ++k) {
          ptrdiff_t j = k.col();
          value_type v = k.value();
          ptrdiff_t cj = idx[j];
          char pj = prm.pmask[j];

          if (pi) {
            if (pj) {
              Kpp->col[pp_head] = cj;
              Kpp->val[pp_head] = v;
              ++pp_head;
            }
            else {
              Kpu->col[pu_head] = cj;
              Kpu->val[pu_head] = v;
              ++pu_head;
            }
          }
          else {
            if (pj) {
              Kup->col[up_head] = cj;
              Kup->val[up_head] = v;
              ++up_head;
            }
            else {
              Kuu->col[uu_head] = cj;
              Kuu->val[uu_head] = v;
              ++uu_head;
            }
          }
        }
      }
    });

    if (prm.verbose >= 2) {
      IO::mm_write("Kuu.mtx", *Kuu);
      IO::mm_write("Kpp.mtx", *Kpp);
    }

    std::shared_ptr<numa_vector<value_type>> Kuu_dia;

    if (prm.simplec_dia) {
      Kuu_dia = std::make_shared<numa_vector<value_type>>(nu);
      arccoreParallelFor(0, nu, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
        for (ptrdiff_t i = begin; i < (begin + size); ++i) {
          value_type s = math::zero<value_type>();
          for (ptrdiff_t j = Kuu->ptr[i], e = Kuu->ptr[i + 1]; j < e; ++j) {
            s += math::norm(Kuu->val[j]);
          }
          (*Kuu_dia)[i] = math::inverse(s);
        }
      });
    }
    else {
      Kuu_dia = diagonal(*Kuu, /*invert = */ true);
    }

    if (prm.adjust_p == 1) {
      // Use (Kpp - dia(Kpu * dia(Kuu)^-1 * Kup))
      // to setup the P preconditioner.
      auto L = std::make_shared<numa_vector<value_type>>(np, false);

      arccoreParallelFor(0, np, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
        for (ptrdiff_t i = begin; i < (begin + size); ++i) {
          value_type s = math::zero<value_type>();
          for (ptrdiff_t j = Kpu->ptr[i], e = Kpu->ptr[i + 1]; j < e; ++j) {
            ptrdiff_t k = Kpu->col[j];
            value_type v = Kpu->val[j];
            for (ptrdiff_t jj = Kup->ptr[k], ee = Kup->ptr[k + 1]; jj < ee; ++jj) {
              if (Kup->col[jj] == i) {
                s += v * (*Kuu_dia)[k] * Kup->val[jj];
                break;
              }
            }
          }

          (*L)[i] = s;
          for (ptrdiff_t j = Kpp->ptr[i], e = Kpp->ptr[i + 1]; j < e; ++j) {
            if (Kpp->col[j] == i) {
              Kpp->val[j] -= s;
              break;
            }
          }
        }
      });
      Ld = backend_type::copy_vector(L, bprm);
    }
    else if (prm.adjust_p == 2) {
      Lm = backend_type::copy_matrix(Kpp, bprm);

      // Use (Kpp - Kpu * dia(Kuu)^-1 * Kup)
      // to setup the P preconditioner.
      numa_vector<value_type> val(Kup->nbNonZero());

      arccoreParallelFor(0, nu, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
        for (ptrdiff_t i = begin; i < (begin + size); ++i) {
          value_type d = (*Kuu_dia)[i];
          for (ptrdiff_t j = Kup->ptr[i], e = Kup->ptr[i + 1]; j < e; ++j) {
            val[j] = d * Kup->val[j];
          }
        }
      });

      build_matrix Kup_hat;

      Kup_hat.own_data = false;
      Kup_hat.setNbRow(nu);
      Kup_hat.ncols = np;
      Kup_hat.setNbNonZero(Kup->nbNonZero());
      Kup_hat.ptr.setPointerZeroCopy(Kup->ptr.data());
      Kup_hat.col.setPointerZeroCopy(Kup->col.data());
      Kup_hat.val.setPointerZeroCopy(val.data());

      Kpp = sum(math::identity<value_type>(), *Kpp, -math::identity<value_type>(), *product(*Kpu, Kup_hat));
    }

    U = std::make_shared<USolver>(*Kuu, prm.usolver, bprm);
    P = std::make_shared<PSolver>(*Kpp, prm.psolver, bprm);

    this->Kup = backend_type::copy_matrix(Kup, bprm);
    this->Kpu = backend_type::copy_matrix(Kpu, bprm);

    rhs_u = backend_type::create_vector(nu, bprm);
    rhs_p = backend_type::create_vector(np, bprm);

    u = backend_type::create_vector(nu, bprm);
    p = backend_type::create_vector(np, bprm);

    tmp = backend_type::create_vector(nu, bprm);

    if (prm.approx_schur)
      M = backend_type::copy_vector(Kuu_dia, bprm);

    // Scatter/Gather matrices
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

      for (size_t i = 0; i < n; ++i) {
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

      for (size_t i = 0; i < n; ++i) {
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
  }

  friend std::ostream& operator<<(std::ostream& os, const SchurPressureCorrectionPreconditioner& p)
  {
    os << "Schur complement (two-stage preconditioner)" << std::endl;
    os << "  Unknowns: " << p.n << "(" << p.np << ")" << std::endl;
    os << "  Nonzeros: " << backend::nonzeros(p.system_matrix()) << std::endl;
    os << "  Memory:  " << human_readable_memory(p.bytes()) << std::endl;
    os << std::endl;
    os << "[ U ]\n"
       << *p.U << std::endl;
    os << "[ P ]\n"
       << *p.P << std::endl;

    return os;
  }

  void report(const std::string& name, const SolverResult& r) const
  {
    if (prm.verbose >= 1) {
      std::cout << name << " (" << r.nbIteration() << ", " << r.residual() << ")\n";
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina::preconditioner

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::backend
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class US, class PS, class Alpha, class Beta, class Vec1, class Vec2>
struct spmv_impl<Alpha, preconditioner::SchurPressureCorrectionPreconditioner<US, PS>, Vec1, Beta, Vec2>
{
  static void apply(Alpha alpha, const preconditioner::SchurPressureCorrectionPreconditioner<US, PS>& A, const Vec1& x, Beta beta, Vec2& y)
  {
    A.spmv(alpha, x, beta, y);
  }
};

template <class US, class PS, class Vec1, class Vec2, class Vec3>
struct residual_impl<preconditioner::SchurPressureCorrectionPreconditioner<US, PS>, Vec1, Vec2, Vec3>
{
  static void apply(const Vec1& rhs, const preconditioner::SchurPressureCorrectionPreconditioner<US, PS>& A, const Vec2& x, Vec3& r)
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
