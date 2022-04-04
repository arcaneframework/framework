// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephParams.cc                                                   (C) 2010 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#include "AlephArcane.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlephParams::
AlephParams()
: TraceAccessor(nullptr)
, m_param_epsilon(1.0e-10)
, m_param_max_iteration(1024)
, m_param_preconditioner_method(TypesSolver::DIAGONAL)
, m_param_solver_method(TypesSolver::PCG)
, m_param_gamma(-1)
, m_param_alpha(-1.0)
, m_param_xo_user(false)
, m_param_check_real_residue(false)
, m_param_print_real_residue(false)
, m_param_debug_info(false)
, m_param_min_rhs_norm(1.e-20)
, m_param_convergence_analyse(false)
, m_param_stop_error_strategy(true)
, m_param_write_matrix_to_file_error_strategy(false)
, m_param_write_matrix_name_error_strategy("SolveErrorAlephMatrix.dbg")
, m_param_listing_output(false)
, m_param_threshold(0.0)
, m_param_print_cpu_time_resolution(false)
, m_param_amg_coarsening_method(0)
, m_param_output_level(0)
, m_param_amg_cycle(1)
, m_param_amg_solver_iterations(1)
, m_param_amg_smoother_iterations(1)
, m_param_amg_smootherOption(TypesSolver::SymHybGSJ_smoother)
, m_param_amg_coarseningOption(TypesSolver::ParallelRugeStuben)
, m_param_amg_coarseSolverOption(TypesSolver::CG_coarse_solver)
, m_param_keep_solver_structure(false)
, m_param_sequential_solver(false)
, m_param_criteria_stop(TypesSolver::RB)
{
  //debug() << "\33[1;4;33m\t[AlephParams] New"<<"\33[0m";
}

AlephParams::
AlephParams(ITraceMng* tm,
            Real epsilon, // epsilon de convergence
            Integer max_iteration, // nb max iterations
            TypesSolver::ePreconditionerMethod preconditioner_method, //  préconditionnement utilisé (defaut DIAG)
            TypesSolver::eSolverMethod solver_method, // méthode de résolution par defaut PCG
            Integer gamma, // destine au parametrage des préconditionnements
            Real alpha, // destine au parametrage des préconditionnements
            bool xo_user, // permet a l'utilisateur d'initialiser le PGC avec un Xo different de zero
            bool check_real_residue,
            bool print_real_residue,
            bool debug_info,
            Real min_rhs_norm,
            bool convergence_analyse,
            bool stop_error_strategy,
            bool write_matrix_to_file_error_strategy,
            String write_matrix_name_error_strategy,
            bool listing_output,
            Real threshold,
            bool print_cpu_time_resolution,
            Integer amg_coarsening_method,
            Integer output_level,
            Integer amg_cycle,
            Integer amg_solver_iterations,
            Integer amg_smoother_iterations,
            TypesSolver::eAmgSmootherOption amg_smootherOption,
            TypesSolver::eAmgCoarseningOption amg_coarseningOption,
            TypesSolver::eAmgCoarseSolverOption amg_coarseSolverOption,
            bool keep_solver_structure,
            bool sequential_solver,
            TypesSolver::eCriteriaStop param_criteria_stop)
: TraceAccessor(tm)
, m_param_epsilon(epsilon)
, m_param_max_iteration(max_iteration)
, m_param_preconditioner_method(preconditioner_method)
, m_param_solver_method(solver_method)
, m_param_gamma(gamma)
, m_param_alpha(alpha)
, m_param_xo_user(xo_user)
, m_param_check_real_residue(check_real_residue)
, m_param_print_real_residue(print_real_residue)
, m_param_debug_info(debug_info)
, m_param_min_rhs_norm(min_rhs_norm)
, m_param_convergence_analyse(convergence_analyse)
, m_param_stop_error_strategy(stop_error_strategy)
, m_param_write_matrix_to_file_error_strategy(write_matrix_to_file_error_strategy)
, m_param_write_matrix_name_error_strategy(write_matrix_name_error_strategy)
, m_param_listing_output(listing_output)
, m_param_threshold(threshold)
, m_param_print_cpu_time_resolution(print_cpu_time_resolution)
, m_param_amg_coarsening_method(amg_coarsening_method)
, m_param_output_level(output_level)
, m_param_amg_cycle(amg_cycle)
, m_param_amg_solver_iterations(amg_solver_iterations)
, m_param_amg_smoother_iterations(amg_smoother_iterations)
, m_param_amg_smootherOption(amg_smootherOption)
, m_param_amg_coarseningOption(amg_coarseningOption)
, m_param_amg_coarseSolverOption(amg_coarseSolverOption)
, m_param_keep_solver_structure(keep_solver_structure)
, m_param_sequential_solver(sequential_solver)
, m_param_criteria_stop(param_criteria_stop)
{
  //debug() << "\33[1;4;33m\t[AlephParams] New"<<"\33[0m";
}

AlephParams::
~AlephParams()
{
  //debug() << "\33[1;4;33m\t[~AlephParams]"<<"\33[0m";
}

// set
void AlephParams::setEpsilon(const Real epsilon)
{
  m_param_epsilon = epsilon;
}
void AlephParams::setMaxIter(const Integer max_iteration)
{
  m_param_max_iteration = max_iteration;
}
void AlephParams::setPrecond(const TypesSolver::ePreconditionerMethod preconditioner_method)
{
  m_param_preconditioner_method = preconditioner_method;
}
void AlephParams::setMethod(const TypesSolver::eSolverMethod solver_method)
{
  m_param_solver_method = solver_method;
}
void AlephParams::setAlpha(const Real alpha)
{
  m_param_alpha = alpha;
}
void AlephParams::setGamma(const Integer gamma)
{
  m_param_gamma = gamma;
}
void AlephParams::setXoUser(const bool xo_user)
{
  m_param_xo_user = xo_user;
}
void AlephParams::setCheckRealResidue(const bool check_real_residue)
{
  m_param_check_real_residue = check_real_residue;
}
void AlephParams::setPrintRealResidue(const bool print_real_residue)
{
  m_param_print_real_residue = print_real_residue;
}
void AlephParams::setDebugInfo(const bool debug_info)
{
  m_param_debug_info = debug_info;
}
void AlephParams::setMinRHSNorm(const Real min_rhs_norm)
{
  m_param_min_rhs_norm = min_rhs_norm;
}
void AlephParams::setConvergenceAnalyse(const bool convergence_analyse)
{
  m_param_convergence_analyse = convergence_analyse;
}
void AlephParams::setStopErrorStrategy(const bool stop_error_strategy)
{
  m_param_stop_error_strategy = stop_error_strategy;
}
void AlephParams::setWriteMatrixToFileErrorStrategy(const bool write_matrix_to_file_error_strategy)
{
  m_param_write_matrix_to_file_error_strategy = write_matrix_to_file_error_strategy;
}
void AlephParams::setWriteMatrixNameErrorStrategy(const String& write_matrix_name_error_strategy)
{
  m_param_write_matrix_name_error_strategy = write_matrix_name_error_strategy;
}
void AlephParams::setDDMCParameterListingOutput(const bool listing_output)
{
  m_param_listing_output = listing_output;
}
void AlephParams::setDDMCParameterAmgDiagonalThreshold(const Real threshold)
{
  m_param_threshold = threshold;
}
void AlephParams::setPrintCpuTimeResolution(const bool print_cpu_time_resolution)
{
  m_param_print_cpu_time_resolution = print_cpu_time_resolution;
}

void AlephParams::
setAmgCoarseningMethod(const TypesSolver::eAmgCoarseningMethod method)
{
  switch (method) {
  case TypesSolver::AMG_COARSENING_AUTO:
    m_param_amg_coarsening_method = 6;
    break;
  case TypesSolver::AMG_COARSENING_HYPRE_0:
    m_param_amg_coarsening_method = 0;
    break;
  case TypesSolver::AMG_COARSENING_HYPRE_1:
    m_param_amg_coarsening_method = 1;
    break;
  case TypesSolver::AMG_COARSENING_HYPRE_3:
    m_param_amg_coarsening_method = 3;
    break;
  case TypesSolver::AMG_COARSENING_HYPRE_6:
    m_param_amg_coarsening_method = 6;
    break;
  case TypesSolver::AMG_COARSENING_HYPRE_7:
    m_param_amg_coarsening_method = 7;
    break;
  case TypesSolver::AMG_COARSENING_HYPRE_8:
    m_param_amg_coarsening_method = 8;
    break;
  case TypesSolver::AMG_COARSENING_HYPRE_9:
    m_param_amg_coarsening_method = 9;
    break;
  case TypesSolver::AMG_COARSENING_HYPRE_10:
    m_param_amg_coarsening_method = 10;
    break;
  case TypesSolver::AMG_COARSENING_HYPRE_11:
    m_param_amg_coarsening_method = 11;
    break;
  case TypesSolver::AMG_COARSENING_HYPRE_21:
    m_param_amg_coarsening_method = 21;
    break;
  case TypesSolver::AMG_COARSENING_HYPRE_22:
    m_param_amg_coarsening_method = 22;
    break;
  default:
    throw NotImplementedException(A_FUNCINFO);
  }
}
void AlephParams::setOutputLevel(const Integer output_level)
{
  m_param_output_level = output_level;
}
void AlephParams::setAmgCycle(const Integer amg_cycle)
{
  m_param_amg_cycle = amg_cycle;
}
void AlephParams::setAmgSolverIter(const Integer amg_solver_iterations)
{
  m_param_amg_solver_iterations = amg_solver_iterations;
}
void AlephParams::setAmgSmootherIter(const Integer amg_smoother_iterations)
{
  m_param_amg_smoother_iterations = amg_smoother_iterations;
}
void AlephParams::setAmgSmootherOption(const TypesSolver::eAmgSmootherOption amg_smootherOption)
{
  m_param_amg_smootherOption = amg_smootherOption;
}
void AlephParams::setAmgCoarseningOption(const TypesSolver::eAmgCoarseningOption amg_coarseningOption)
{
  m_param_amg_coarseningOption = amg_coarseningOption;
}
void AlephParams::setAmgCoarseSolverOption(const TypesSolver::eAmgCoarseSolverOption amg_coarseSolverOption)
{
  m_param_amg_coarseSolverOption = amg_coarseSolverOption;
}
void AlephParams::setKeepSolverStructure(const bool keep_solver_structure)
{
  m_param_keep_solver_structure = keep_solver_structure;
}
void AlephParams::setSequentialSolver(const bool sequential_solver)
{
  m_param_sequential_solver = sequential_solver;
}
void AlephParams::setCriteriaStop(const TypesSolver::eCriteriaStop criteria_stop)
{
  m_param_criteria_stop = criteria_stop;
}

// get
Real AlephParams::epsilon() const
{
  return m_param_epsilon;
}
int AlephParams::maxIter() const
{
  return m_param_max_iteration;
}
Real AlephParams::alpha() const
{
  return m_param_alpha;
}
int AlephParams::gamma() const
{
  return m_param_gamma;
}
TypesSolver::ePreconditionerMethod AlephParams::precond()
{
  return m_param_preconditioner_method;
}
TypesSolver::eSolverMethod AlephParams::method()
{
  return m_param_solver_method;
}
bool AlephParams::xoUser() const
{
  return m_param_xo_user;
}
bool AlephParams::checkRealResidue() const
{
  return m_param_check_real_residue;
}
bool AlephParams::printRealResidue() const
{
  return m_param_print_real_residue;
}
bool AlephParams::debugInfo()
{
  return m_param_debug_info;
}
Real AlephParams::minRHSNorm()
{
  return m_param_min_rhs_norm;
}
bool AlephParams::convergenceAnalyse()
{
  return m_param_convergence_analyse;
}
bool AlephParams::stopErrorStrategy()
{
  return m_param_stop_error_strategy;
}
bool AlephParams::writeMatrixToFileErrorStrategy()
{
  return m_param_write_matrix_to_file_error_strategy;
}
String AlephParams::writeMatrixNameErrorStrategy()
{
  return m_param_write_matrix_name_error_strategy;
}
bool AlephParams::DDMCParameterListingOutput() const
{
  return m_param_listing_output;
}
Real AlephParams::DDMCParameterAmgDiagonalThreshold() const
{
  return m_param_threshold;
}
bool AlephParams::printCpuTimeResolution() const
{
  return m_param_print_cpu_time_resolution;
}
int AlephParams::amgCoarseningMethod() const
{
  return m_param_amg_coarsening_method;
} // -1 pour Sloop
int AlephParams::getOutputLevel() const
{
  return m_param_output_level;
}
int AlephParams::getAmgCycle() const
{
  return m_param_amg_cycle;
}
int AlephParams::getAmgSolverIter() const
{
  return m_param_amg_solver_iterations;
}
int AlephParams::getAmgSmootherIter() const
{
  return m_param_amg_smoother_iterations;
}
TypesSolver::eAmgSmootherOption AlephParams::getAmgSmootherOption() const
{
  return m_param_amg_smootherOption;
}
TypesSolver::eAmgCoarseningOption AlephParams::getAmgCoarseningOption() const
{
  return m_param_amg_coarseningOption;
}
TypesSolver::eAmgCoarseSolverOption AlephParams::getAmgCoarseSolverOption() const
{
  return m_param_amg_coarseSolverOption;
}
bool AlephParams::getKeepSolverStructure() const
{
  return m_param_keep_solver_structure;
}
bool AlephParams::getSequentialSolver() const
{
  return m_param_sequential_solver;
}
TypesSolver::eCriteriaStop AlephParams::getCriteriaStop() const
{
  return m_param_criteria_stop;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
