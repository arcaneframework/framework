// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DistributedAMG.h                                            (C) 2000-2026 */
/*                                                                           */
/* Distributed memory AMG preconditioner.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_DISTRIBUTEDAMG_H
#define ARCCORE_ALINA_DISTRIBUTEDAMG_H
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

#include <iostream>
#include <iomanip>
#include <list>
#include <memory>

#include "arccore/alina/BackendInterface.h"
#include "arccore/alina/MessagePassingUtils.h"
#include "arccore/alina/DistributedMatrix.h"
#include "arccore/alina/DistributedSkylineLUDirectSolver.h"
#include "arccore/alina/SimpleMatrixPartitioner.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend,
          class Coarsening,
          class Relaxation,
          class DirectSolver = DistributedSkylineLUDirectSolver<typename Backend::value_type>,
          class Repartition = SimpleMatrixPartitioner<Backend>>
class DistributedAMG
{
 public:

  using backend_type = Backend;
  using BackendType = Backend;

  typedef typename Backend::params backend_params;
  typedef typename Backend::value_type value_type;
  typedef typename math::scalar_of<value_type>::type scalar_type;
  typedef DistributedMatrix<Backend> matrix;
  typedef typename Backend::vector vector;

  struct params
  {
    typedef typename Coarsening::params coarsening_params;
    typedef typename Relaxation::params relax_params;
    typedef typename DirectSolver::params direct_params;
    typedef typename Repartition::params repart_params;

    coarsening_params coarsening; ///< Coarsening parameters.
    relax_params relax; ///< Relaxation parameters.
    direct_params direct; ///< Direct solver parameters.
    repart_params repart; ///< Repartition parameters.

    /*!
     * \brief Specifies when level is coarse enough to be solved directly.
     *
     * If number of variables at a next level in the hierarchy becomes
     * lower than this threshold, then the hierarchy construction is
     * stopped and the linear system is solved directly at this level.
     */
    Int32 coarse_enough = DirectSolver::coarse_enough();

    /*!
     * \brief Use direct solver at the coarsest level.
     *
     * When set, the coarsest level is solved with a direct solver.
     * Otherwise a smoother is used as a solver.
     */
    bool direct_coarse = true;

    /*!
     * \brief Maximum number of levels.
     *
     * If this number is reached while the size of the last level is
     * greater that `coarse_enough`, then the coarsest level will not
     * be solved exactly, but will use a smoother.
     */
    Int32 max_levels = std::numeric_limits<Int32>::max();

    /// Number of pre-relaxations.
    Int32 npre = 1;

    /// Number of post-relaxations.
    Int32 npost = 1;

    /// Number of cycles (1 for V-cycle, 2 for W-cycle, etc.).
    Int32 ncycle = 1;

    /// Number of cycles to make as part of preconditioning.
    Int32 pre_cycles = 1;

    /// Keep matrices in internal format to allow for quick rebuild of the hierarchy
    bool allow_rebuild = false;

    params() = default;

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, coarsening)
    , ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, relax)
    , ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, direct)
    , ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, repart)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, coarse_enough)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, direct_coarse)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, max_levels)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, npre)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, npost)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, ncycle)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, pre_cycles)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, allow_rebuild)
    {
      p.check_params({ "coarsening", "relax", "direct", "repart", "coarse_enough", "direct_coarse", "max_levels", "npre", "npost", "ncycle", "pre_cycles", "allow_rebuild" });

      Alina::precondition(max_levels > 0, "max_levels should be positive");
    }

    void get(Alina::PropertyTree& p, const std::string& path = "") const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, coarsening);
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, relax);
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, direct);
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, repart);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, coarse_enough);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, direct_coarse);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, max_levels);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, npre);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, npost);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, ncycle);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, pre_cycles);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, allow_rebuild);
    }
  } prm;

  template <class Matrix>
  DistributedAMG(mpi_communicator comm,
                 const Matrix& A,
                 const params& prm = params(),
                 const backend_params& bprm = backend_params())
  : prm(prm)
  , comm(comm)
  , repart(prm.repart)
  {
    init(std::make_shared<matrix>(comm, A, backend::nbRow(A)), bprm);
  }

  DistributedAMG(mpi_communicator comm,
                 std::shared_ptr<matrix> A,
                 const params& prm = params(),
                 const backend_params& bprm = backend_params())
  : prm(prm)
  , comm(comm)
  , repart(prm.repart)
  {
    init(A, bprm);
  }

  /*!
   * \brief Rebuild the hierarchy using the new system matrix.
   *
   * This requires for prm.allow_rebuild to be set. The transfer
   * operators created during the initial setup are reused.
   */
  template <class Matrix>
  void rebuild(const Matrix& M,
               const backend_params& bprm = backend_params())
  {
    rebuild(std::make_shared<matrix>(comm, M, backend::nbRow(M)), bprm);
  }

  template <class OtherBackend>
  typename std::enable_if<!std::is_same<Backend, OtherBackend>::value, void>::type
  rebuild(std::shared_ptr<DistributedMatrix<OtherBackend>> A,
          const backend_params& bprm = backend_params())
  {
    return rebuild(std::make_shared<matrix>(*A), bprm);
  }

  void rebuild(std::shared_ptr<matrix> A,
               const backend_params& bprm = backend_params())
  {
    precondition(prm.allow_rebuild, "allow_rebuild is not set!");
    precondition(A->glob_rows() == system_matrix().glob_rows() &&
                 A->glob_cols() == system_matrix().glob_cols(),
                 "Matrix dimensions differ from the original ones!");

    ARCCORE_ALINA_TIC("rebuild");
    this->A = A;
    Coarsening C(prm.coarsening);
    for (auto& level : levels) {
      A = level.rebuild(A, C, prm, bprm);
    }
    this->A->move_to_backend(bprm);
    ARCCORE_ALINA_TOC("rebuild");
  }

  template <class Vec1, class Vec2>
  void cycle(const Vec1& rhs, Vec2&& x) const
  {
    cycle(levels.begin(), rhs, x);
  }

  template <class Vec1, class Vec2>
  void apply(const Vec1& rhs, Vec2&& x) const
  {
    if (prm.pre_cycles) {
      backend::clear(x);
      for (unsigned i = 0; i < prm.pre_cycles; ++i)
        cycle(levels.begin(), rhs, x);
    }
    else {
      backend::copy(rhs, x);
    }
  }

  /// Returns the system matrix from the finest level.
  std::shared_ptr<matrix> system_matrix_ptr() const
  {
    return A;
  }

  const matrix& system_matrix() const
  {
    return *system_matrix_ptr();
  }

 private:

  struct DistributedAMGLevel
  {
    ptrdiff_t nrows, nnz;
    int active_procs;

    std::shared_ptr<matrix> A, P, R;
    std::shared_ptr<vector> f, u, t;
    std::shared_ptr<Relaxation> relax;
    std::shared_ptr<DirectSolver> solve;

    DistributedAMGLevel() = default;

    DistributedAMGLevel(std::shared_ptr<matrix> a,
                        params& prm,
                        const backend_params& bprm,
                        bool direct = false)
    : nrows(a->glob_rows())
    , nnz(a->glob_nonzeros())
    , f(Backend::create_vector(a->loc_rows(), bprm))
    , u(Backend::create_vector(a->loc_rows(), bprm))
    {
      int active = (a->loc_rows() > 0);
      active_procs = a->comm().reduceSum(active);

      sort_rows(*a);

      if (direct) {
        ARCCORE_ALINA_TIC("direct solver");
        solve = std::make_shared<DirectSolver>(a->comm(), *a, prm.direct);
        ARCCORE_ALINA_TOC("direct solver");
      }
      else {
        A = a;
        t = Backend::create_vector(a->loc_rows(), bprm);

        ARCCORE_ALINA_TIC("relaxation");
        relax = std::make_shared<Relaxation>(*a, prm.relax, bprm);
        ARCCORE_ALINA_TOC("relaxation");
      }
    }

    std::shared_ptr<matrix> step_down(Coarsening& C, const Repartition& repart)
    {
      ARCCORE_ALINA_TIC("transfer operators");
      std::tie(P, R) = C.transfer_operators(*A);

      ARCCORE_ALINA_TIC("sort");
      sort_rows(*P);
      sort_rows(*R);
      ARCCORE_ALINA_TOC("sort");

      ARCCORE_ALINA_TOC("transfer operators");

      if (P->glob_cols() == 0) {
        // Zero-sized coarse level in AMG (diagonal matrix?)
        return std::shared_ptr<matrix>();
      }

      ARCCORE_ALINA_TIC("coarse operator");
      auto Ac = C.coarse_operator(*A, *P, *R);
      ARCCORE_ALINA_TOC("coarse operator");

      if (repart.is_needed(*Ac)) {
        ARCCORE_ALINA_TIC("partition");
        auto I = repart(*Ac, block_size(C));
        auto J = transpose(*I);

        P = product(*P, *I);
        R = product(*J, *R);
        Ac = product(*J, *product(*Ac, *I));
        ARCCORE_ALINA_TOC("partition");
      }

      return Ac;
    }

    std::shared_ptr<matrix> rebuild(std::shared_ptr<matrix> A,
                                    const Coarsening& C,
                                    const params& prm,
                                    const backend_params& bprm)
    {
      if (relax) {
        relax = std::make_shared<Relaxation>(*A, prm.relax, bprm);
        std::cout << "DistributedAMG: relaxation=" << relax << "\n";
      }

      if (solve) {
        solve = std::make_shared<DirectSolver>(A->comm(), *A, prm.direct);
      }

      if (this->A) {
        this->A = A;
      }

      if (P && R) {
        A = C.coarse_operator(*A, *P, *R);
      }

      if (this->A) {
        this->A->move_to_backend(bprm);
      }

      return A;
    }

    void move_to_backend(const backend_params& bprm, bool keep_src = false)
    {
      ARCCORE_ALINA_TIC("move to backend");
      if (A)
        A->move_to_backend(bprm);
      if (P)
        P->move_to_backend(bprm, keep_src);
      if (R)
        R->move_to_backend(bprm, keep_src);
      ARCCORE_ALINA_TOC("move to backend");
    }

    ptrdiff_t rows() const
    {
      return nrows;
    }

    ptrdiff_t nonzeros() const
    {
      return nnz;
    }
  };

  typedef typename std::list<DistributedAMGLevel>::const_iterator level_iterator;

  mpi_communicator comm;
  std::shared_ptr<matrix> A;
  Repartition repart;
  std::list<DistributedAMGLevel> levels;

  void init(std::shared_ptr<matrix> A, const backend_params& bprm)
  {
    A->comm().check(A->glob_rows() == A->glob_cols(), "Matrix should be square!");

    this->A = A;
    Coarsening C(prm.coarsening);
    //std::cout << "DistributedAMGInit: Coarsening=" << C << "\n";
    bool need_coarse = true;

    while (A->glob_rows() > prm.coarse_enough) {
      levels.push_back(DistributedAMGLevel(A, prm, bprm));

      if (levels.size() >= prm.max_levels) {
        levels.back().move_to_backend(bprm, prm.allow_rebuild);
        break;
      }

      A = levels.back().step_down(C, repart);
      levels.back().move_to_backend(bprm, prm.allow_rebuild);

      if (!A) {
        // Zero-sized coarse level. Probably the system matrix on
        // this level is diagonal, should be easily solvable with a
        // couple of smoother iterations.
        need_coarse = false;
        break;
      }
    }

    if (!A || A->glob_rows() > prm.coarse_enough) {
      // The coarse matrix is still too big to be solved directly.
      need_coarse = false;
    }

    if (A && need_coarse) {
      levels.push_back(DistributedAMGLevel(A, prm, bprm, prm.direct_coarse));
      levels.back().move_to_backend(bprm, prm.allow_rebuild);
    }

    ARCCORE_ALINA_TIC("move to backend");
    this->A->move_to_backend(bprm, prm.allow_rebuild);
    ARCCORE_ALINA_TOC("move to backend");
  }

  template <class Vec1, class Vec2>
  void cycle(level_iterator lvl, const Vec1& rhs, Vec2& x) const
  {
    level_iterator nxt = lvl, end = levels.end();
    ++nxt;

    if (nxt == end) {
      if (lvl->solve) {
        ARCCORE_ALINA_TIC("direct solver");
        (*lvl->solve)(rhs, x);
        ARCCORE_ALINA_TOC("direct solver");
      }
      else {
        ARCCORE_ALINA_TIC("relax");
        for (size_t i = 0; i < prm.npre; ++i)
          lvl->relax->apply_pre(*lvl->A, rhs, x, *lvl->t);
        for (size_t i = 0; i < prm.npost; ++i)
          lvl->relax->apply_post(*lvl->A, rhs, x, *lvl->t);
        ARCCORE_ALINA_TOC("relax");
      }
    }
    else {
      for (size_t j = 0; j < prm.ncycle; ++j) {
        ARCCORE_ALINA_TIC("relax");
        for (size_t i = 0; i < prm.npre; ++i)
          lvl->relax->apply_pre(*lvl->A, rhs, x, *lvl->t);
        ARCCORE_ALINA_TOC("relax");

        backend::residual(rhs, *lvl->A, x, *lvl->t);

        backend::spmv(math::identity<scalar_type>(), *lvl->R, *lvl->t, math::zero<scalar_type>(), *nxt->f);

        backend::clear(*nxt->u);
        cycle(nxt, *nxt->f, *nxt->u);

        backend::spmv(math::identity<scalar_type>(), *lvl->P, *nxt->u, math::identity<scalar_type>(), x);

        ARCCORE_ALINA_TIC("relax");
        for (size_t i = 0; i < prm.npost; ++i)
          lvl->relax->apply_post(*lvl->A, rhs, x, *lvl->t);
        ARCCORE_ALINA_TOC("relax");
      }
    }
  }

  template <class B, class C, class R, class D, class I>
  friend std::ostream& operator<<(std::ostream& os, const DistributedAMG<B, C, R, D, I>& a);
};

template <class B, class C, class R, class D, class I>
std::ostream& operator<<(std::ostream& os, const DistributedAMG<B, C, R, D, I>& a)
{
  typedef typename DistributedAMG<B, C, R, D, I>::DistributedAMGLevel level;
  ScopedStreamModifier ss(os);

  size_t sum_dof = 0;
  size_t sum_nnz = 0;

  for (const level& lvl : a.levels) {
    sum_dof += lvl.rows();
    sum_nnz += lvl.nonzeros();
  }

  os << "Preconditioner: DistributedAMG\n";
  //os << "Coarsening: " << a.prm.coarsening.type << "\n";
  //os << "Relaxation: " << a.prm.relax.type << "\n";

  os << "Number of levels:    " << a.levels.size()
     << "\nOperator complexity: " << std::fixed << std::setprecision(2)
     << 1.0 * sum_nnz / a.levels.front().nonzeros()
     << "\nGrid complexity:     " << std::fixed << std::setprecision(2)
     << 1.0 * sum_dof / a.levels.front().rows()
     << "\n\nlevel     unknowns       nonzeros\n"
     << "---------------------------------\n";

  size_t depth = 0;
  for (const level& lvl : a.levels) {
    os << std::setw(5) << depth++
       << std::setw(13) << lvl.rows()
       << std::setw(15) << lvl.nonzeros() << " ("
       << std::setw(5) << std::fixed << std::setprecision(2)
       << 100.0 * lvl.nonzeros() / sum_nnz
       << "%) [" << lvl.active_procs << "]" << std::endl;
  }

  return os;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
