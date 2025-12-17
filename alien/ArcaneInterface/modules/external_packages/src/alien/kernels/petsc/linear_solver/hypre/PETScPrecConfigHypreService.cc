// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <alien/kernels/petsc/linear_solver/hypre/PETScPrecConfigHypreService.h>
#include <ALIEN/axl/PETScPrecConfigHypre_StrongOptions.h>
// using namespace Arcane;
// using namespace Alien;

#include <arccore/base/NotImplementedException.h>
#include <arccore/message_passing/IMessagePassingMng.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// BEGIN_LINEARALGEBRA2SERVICE_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace Alien {
/** Constructeur de la classe */
#ifdef ALIEN_USE_ARCANE
PETScPrecConfigHypreService::PETScPrecConfigHypreService(
    const Arcane::ServiceBuildInfo& sbi)
: ArcanePETScPrecConfigHypreObject(sbi)
, PETScConfig(sbi.subDomain()->parallelMng()->isParallel())
{
  ;
}
#endif
PETScPrecConfigHypreService::PETScPrecConfigHypreService(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsPETScPrecConfigHypre> options)
: ArcanePETScPrecConfigHypreObject(options)
, PETScConfig(parallel_mng->commSize() > 1)
{
}

void
PETScPrecConfigHypreService::configure(PC& pc, [[maybe_unused]] const ISpace& space,
                                       [[maybe_unused]] const MatrixDistribution& distribution)
{
  alien_debug([&] { cout() << "configure PETSc hypre preconditioner"; });
  // if(options()->fieldSplitMode())
  checkError("Set preconditioner", PCSetType(pc, PCHYPRE));

  // for more options see
  // http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/PC/PCHYPRE.html

  switch (options()->type()) {
  case PETScPrecConfigHypreOptions::PILUT:
    checkError("Set Hypre preconditioner", PCHYPRESetType(pc, "pilut"));
    break;
  case PETScPrecConfigHypreOptions::AMG:
    checkError("Set Hypre preconditioner", PCHYPRESetType(pc, "boomeramg"));
    checkError("Set Hypre coarsening",
        PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_coarsen_type", "Falgout"));
    checkError("Set Hypre Interpolation type",
        PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_interp_type", "classical"));
    checkError("Set Hypre Relax type",
        PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_relax_type_all", "SOR/Jacobi"));
    checkError("Hypre AMG SetDebugFlag",
        PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_print_debug", "0"));
    PCSetFromOptions(pc);
    // Default option is reversed in PETSc (CF-Relaxation is default)
    // checkError("Set Hypre Relax order", PetscOptionsSetValue(NULL,
    // "-pc_hypre_boomeramg_CF","1"));
    if(options()->outputLevel()>0)
    {
      checkError("Hypre AMG SetDebugFlag",       PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_print_debug", "1"));
      checkError("Hypre AMG SetDebugFlag",       PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_print_statistics", "3"));
    }
    break ;
  case PETScPrecConfigHypreOptions::AMGN:
    /*
     * -pc_hypre_boomeramg_nodal_coarsen <n> -pc_hypre_boomeramg_vec_interp_variant <v>
     *
    <n>
    1 : Frobenius norm
    2 : sum of absolute values of elements in each block
    3 : largest element in each block (not absolute value)
    4 : row-sum norm
    6 : sum of all values in each block

    <v>
    1 : GM approach 1 (Global Matrix)
    2 : GM approach 2 (to be preferred over 1)
    3 : LN approach (Local Neighbourg)
     *
     */
    {
      checkError("Set Hypre preconditioner",     PCHYPRESetType(pc, "boomeramg"));
      checkError("Set Hypre coarsening",         PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_coarsen_type",   options()->coarsenType().localstr()));
      checkError("Set Hypre Interpolation type", PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_interp_type",    options()->interpType().localstr()));
      checkError("Set Hypre Relax type",         PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_relax_type_all", options()->relaxType().localstr()));

      if(options()->amgTol()>0)
      {
        Arcane::String factor = Arcane::String::format("{0}",options()->amgTol()) ;
        checkError("Set Hypre AMG tol",        PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_truncfactor",factor.localstr())) ;
      }

      if(options()->maxLevels()>0)
      {
        Arcane::String max_level = Arcane::String::format("{0}",options()->maxLevels()) ;
        checkError("Set Hypre max level",        PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_max_levels",max_level.localstr())) ;
      }

      if(options()->truncFactor()>0)
      {
        Arcane::String factor = Arcane::String::format("{0}",options()->truncFactor()) ;
        checkError("Set Hypre trunc-factor",        PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_truncfactor",factor.localstr())) ;
      }
      if(options()->strongThreshold()>0)
      {
        Arcane::String factor = Arcane::String::format("{0}",options()->strongThreshold()) ;
        checkError("Set Hypre strong threshold",        PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_strong_threshold",factor.localstr())) ;
      }
      if(options()->maxRowSum()>0)
      {
        Arcane::String factor = Arcane::String::format("{0}",options()->maxRowSum()) ;
        checkError("Set Hypre maw row sum",        PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_max_row_sum",factor.localstr())) ;
      }
      if(options()->pmaxElements()>0)
      {
        Arcane::String pmax_elements = Arcane::String::format("{0}",options()->pmaxElements()) ;
        checkError("Set Hypre Pmax Elem per row",  PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_P_max",pmax_elements.localstr())) ;
      }
      if(options()->nodalCoarsen()>0)
      {
        Arcane::String nodal_coarsen      = Arcane::String::format("{0}",options()->nodalCoarsen()) ;
        Arcane::String vec_interp_variant = Arcane::String::format("{0}",options()->vecInterpVariant()) ;
        checkError("Set Hypre Nodal Coarsen algo", PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_nodal_coarsen",nodal_coarsen.localstr()));
        checkError("Set Hypre Vec Interp Variant", PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_vec_interp_variant",vec_interp_variant.localstr()));
      }

      {
       checkError("Agg NL",PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_smooth_num_levels","0")) ;
       checkError("Agg NL",PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_agg_nl","0")) ;
       checkError("Agg NP",PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_agg_num_paths","1")) ;
       checkError("Sweep All",PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_grid_sweeps_all","1")) ;
       checkError("Sweep Down",PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_grid_sweeps_down","1")) ;
       checkError("Sweep Up",PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_grid_sweeps_up","1")) ;
       checkError("Sweep Coarse",PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_grid_sweeps_coarse","1")) ;
       checkError("Measure",PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_measure_type","local")) ;
      }
      
      if(options()->outputLevel()>0)
      {
        checkError("Hypre AMG SetDebugFlag",       PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_print_debug", "1"));
        checkError("Hypre AMG SetDebugFlag",       PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_print_statistics", "3"));
      }
      PCSetFromOptions(pc);
      // Default option is reversed in PETSc (CF-Relaxation is default)
      // checkError("Set Hypre Relax order", PetscOptionsSetValue(NULL,"-pc_hypre_boomeramg_CF","1"));
    }
    break;
  case PETScPrecConfigHypreOptions::ParaSails:
    checkError("Set Hypre preconditioner", PCHYPRESetType(pc, "parasails"));
    break;
  case PETScPrecConfigHypreOptions::Euclid:
    checkError("Set Hypre preconditioner", PCHYPRESetType(pc, "euclid"));
    break;
  default:
    throw Arccore::NotImplementedException(A_FUNCINFO, "Undefined Hypre type");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_PETSCPRECCONFIGHYPRE(Hypre, PETScPrecConfigHypreService);

} // namespace Alien

REGISTER_STRONG_OPTIONS_PETSCPRECCONFIGHYPRE();
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// END_LINEARALGEBRA2SERVICE_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
