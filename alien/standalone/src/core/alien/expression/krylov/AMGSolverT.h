// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <ostream>
#include <vector>

#include <alien/core/backend/MatrixConverterRegisterer.h>
#include <alien/core/backend/VectorConverterRegisterer.h>
#include <alien/core/backend/KernelSolverT.h>
#include <alien/core/backend/LinearSolver.h>

namespace Alien
{

template <typename TagT,
          typename AlgebraT,
          typename SolverT,
          typename PrecondT>
class AMGSolverT
    : public KernelSolverT<TagT>
{
 public:
  // clang-format off
  using KernelSolverType = KernelSolverT<TagT> ;
  using SolverType       = SolverT;
  using PrecondType      = PrecondT;
  using AlgebraType      = AlgebraT;
  using MatrixType       = typename AlgebraType::MatrixType ;
  using VectorType       = typename AlgebraType::VectorType ;
  using ValueType        = typename AlgebraType::ValueType ;
  // clang-format on
  AMGSolverT(AlgebraT& alg, SolverT& solver, PrecondT& precond)
  : m_algebra(alg)
  , m_solver(solver)
  , m_precond(precond)
  {}

  virtual ~AMGSolverT(){}

  void init()
  {
    if constexpr (requires{m_solver.init();}) {
      m_solver.init() ;
    }
  }

  void start() {
    if constexpr (requires{m_solver.start();}) {
      m_solver.init() ;
    }

  }

  //! Initialize the linear solver
  void init(MatrixType const& A) {
    m_matrix = &A ;
    m_precond.init() ;
  }


  bool solve(const VectorType& b, VectorType& x)
  {
    auto iter = Alien::Iteration<AlgebraType>{m_algebra,b,0.,1,nullptr} ;
    m_solver.solve(m_precond,iter,*m_matrix,b,x) ;
    return true ;
  }
 private:
  AlgebraType&      m_algebra ;
  SolverType&       m_solver;
  PrecondType&      m_precond ;

  MatrixType const* m_matrix = nullptr ;
};

template <typename TagT,
          typename AlgebraT,
          typename AMGSolverTagT>
class KernelAMGSolverT
    : public KernelSolverT<TagT>
{
 public:
  // clang-format off
  using BaseSolverType   = KernelSolverT<TagT> ;
  using AlgebraType      = AlgebraT;
  using MatrixType       = typename AlgebraType::MatrixType ;
  using VectorType       = typename AlgebraType::VectorType ;
  using ValueType        = typename AlgebraType::ValueType ;

  using SolverMatrixType  = typename AlgebraTraits<AMGSolverTagT>::matrix_type;
  using SolverVectorType  = typename AlgebraTraits<AMGSolverTagT>::vector_type;
  using SolverAlgebraType = typename AlgebraTraits<AMGSolverTagT>::algebra_type;
  using AMGSolverType     = KernelSolverT<AMGSolverTagT>;

  using MatrixConvType     = MatrixConverterT<TagT,AMGSolverTagT> ;
  using VectorConvFromType = VectorConverterT<TagT,AMGSolverTagT> ;
  using VectorConvToType   = VectorConverterT<AMGSolverTagT,TagT> ;

  // clang-format on
  KernelAMGSolverT(AlgebraT& alg, AMGSolverType* solver)
  : BaseSolverType()
  , m_algebra(alg)
  , m_amg_solver(solver)
  {
    const BackEndId backend_id        = AlgebraTraits<TagT>::BackEndId() ;
    const BackEndId solver_backend_id = AlgebraTraits<AMGSolverTagT>::BackEndId() ;
    m_matrix_converter =
        MatrixConverterRegisterer::getConverter(backend_id,solver_backend_id);
    m_vector_converter_from =
        VectorConverterRegisterer::getConverter(backend_id,solver_backend_id);
    m_vector_converter_to =
        VectorConverterRegisterer::getConverter(solver_backend_id,backend_id);
  }

  // clang-format on
    KernelAMGSolverT(AlgebraT& alg, Alien::ILinearSolver* solver)
    : BaseSolverType()
    , m_algebra(alg)
    {
      auto solver_ptr = dynamic_cast<Alien::LinearSolver<AMGSolverTagT>*>(solver) ;
      if(solver_ptr)
        m_amg_solver = dynamic_cast<AMGSolverType*>(solver_ptr->implem()) ;
      else
        m_amg_solver = dynamic_cast<AMGSolverType*>(solver) ;
      assert(m_amg_solver) ;

      const BackEndId backend_id        = AlgebraTraits<TagT>::name() ;
      const BackEndId solver_backend_id = AlgebraTraits<AMGSolverTagT>::name() ;
      m_matrix_converter =
          dynamic_cast<MatrixConvType*>(MatrixConverterRegisterer::getConverter(backend_id,solver_backend_id));
      m_vector_converter_from =
          dynamic_cast<VectorConvFromType*>(VectorConverterRegisterer::getConverter(backend_id,solver_backend_id));
      m_vector_converter_to =
          dynamic_cast<VectorConvToType*>(VectorConverterRegisterer::getConverter(solver_backend_id,backend_id));
    }

  virtual ~KernelAMGSolverT(){}

  void init()
  {
    m_amg_solver->init() ;
  }

  void start() {

  }

  //! Initialize the linear solver
  void init(MatrixType const& A)
  {
    m_matrix = &A ;
    if(m_solver_matrix.get()==nullptr)
    {
      auto ptr = new SolverMatrixType(m_matrix->impls()) ;
      m_solver_matrix.reset(ptr) ;
    }
    m_matrix_converter->convert(A, *m_solver_matrix);
    m_amg_solver->init(*m_solver_matrix) ;
  }


  bool solve(const VectorType& b, VectorType& x)
  {
    if(m_solver_b.get()==nullptr)
    {
      auto ptr =  new SolverVectorType(nullptr) ;
      ptr->init(AlgebraType::resource(*m_matrix),true) ;
      m_solver_b.reset(ptr) ;
    }
    m_vector_converter_from->convert(b, *m_solver_b);
    if(m_solver_x.get()==nullptr)
    {
      auto ptr =  new SolverVectorType(nullptr) ;
      ptr->init(AlgebraType::resource(*m_matrix),true) ;
      m_solver_x.reset(ptr) ;
    }
    m_amg_solver->solve(*m_solver_b,*m_solver_x) ;
    m_vector_converter_to->convert(*m_solver_x,x);
    return true ;
  }
 private:
  AlgebraType&                      m_algebra ;
  AMGSolverType*                    m_amg_solver = nullptr;
  MatrixType const*                 m_matrix = nullptr ;
  std::unique_ptr<SolverMatrixType> m_solver_matrix;
  std::unique_ptr<SolverVectorType> m_solver_b;
  std::unique_ptr<SolverVectorType> m_solver_x;
  MatrixConvType*                   m_matrix_converter = nullptr ;
  VectorConvFromType*               m_vector_converter_from = nullptr ;
  VectorConvToType*                 m_vector_converter_to = nullptr ;
};

} // namespace Alien
