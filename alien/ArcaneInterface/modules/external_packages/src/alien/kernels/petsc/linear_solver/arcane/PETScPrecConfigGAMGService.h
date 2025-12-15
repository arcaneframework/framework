#pragma once

#include <alien/kernels/petsc/PETScPrecomp.h>
#include <alien/AlienExternalPackagesPrecomp.h>

#include <alien/kernels/petsc/linear_solver/IPETScPC.h>
#include <alien/kernels/petsc/linear_solver/IPETScKSP.h>
#include <alien/kernels/petsc/linear_solver/PETScConfig.h>

#include <ALIEN/axl/PETScPrecConfigGAMG_axl.h>



/*
 * PCGAMG
   Geometric algebraic multigrid (AMG) preconditioner

    Options Database Keys
    -pc_gamg_type <type,default=agg> - one of agg, geo, or classical (only smoothed aggregation, agg, supported)

    -pc_gamg_repartition <bool,default=false> - repartition the degrees of freedom across the coarse grids as they are determined

    -pc_gamg_asm_use_agg <bool,default=false> - use the aggregates from the coasening process to defined the subdomains on each level for the PCASM smoother. That is using -mg_levels_pc_type asm

    -pc_gamg_process_eq_limit <limit, default=50> - PCGAMG will reduce the number of MPI ranks used directly on the coarse grids so that there are around equations on each process that has degrees of freedom

    -pc_gamg_coarse_eq_limit <limit, default=50> - Set maximum number of equations on coarsest grid to aim for.

    -pc_gamg_reuse_interpolation <bool,default=true> - when rebuilding the algebraic multigrid preconditioner reuse the previously computed interpolations (should always be true)

    -pc_gamg_threshold[] <thresh,default=[- 1,…]> - Before aggregating the graph PCGAMG will remove small values from the graph on each level (< 0 does no filtering)

    -pc_gamg_threshold_scale <scale,default=1> - Scaling of threshold on each coarser grid if not specified

    Options Database Keys for Aggregation
    -pc_gamg_agg_nsmooths <nsmooth, default=1> - number of smoothing steps to use with smooth aggregation to construct prolongation

    -pc_gamg_aggressive_coarsening <n,default=1> - number of aggressive coarsening (MIS-2) levels from finest.

    -pc_gamg_aggressive_square_graph <bool,default=true> - Use square graph
 for coarsening. Otherwise, MIS-k (k=2) is used, see PCGAMGMISkSetAggressive().

    -pc_gamg_mis_k_minimum_degree_ordering <bool,default=false>- Use minimum degree ordering in greedy MIS algorithm

    -pc_gamg_pc_gamg_asm_hem_aggs <n,default=0> - Number of HEM aggregation steps for PCASM smoother

    -pc_gamg_aggressive_mis_k <n,default=2> - Number (k) distance in MIS coarsening (>2 is ‘aggressive’)

    Options Database Keys for Multigrid
    -pc_mg_cycle_type - v or w, see PCMGSetCycleType()

    -pc_mg_distinct_smoothup - configure the up and down (pre and post) smoothers separately, see PCMGSetDistinctSmoothUp()

    -pc_mg_type - (one of) additive multiplicative full kascade

    -pc_mg_levels - Number of levels of multigrid to use. GAMG has a heuristic so pc_mg_levels is not usually used with GAMG

    Notes
    To obtain good performance for PCGAMG for vector valued problems you must call MatSetBlockSize()
    to indicate the number of degrees of freedom per grid point call MatSetNearNullSpace() (or     PCSetCoordinates() if solving the equations of elasticity) to indicate the near null space of the operator

 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_EXTERNAL_PACKAGES_EXPORT PETScPrecConfigGAMGService
    : public ArcanePETScPrecConfigGAMGObject,
      public PETScConfig
{
 public:
/** Constructeur de la classe */
#ifdef ALIEN_USE_ARCANE
  PETScPrecConfigGAMGService(const Arcane::ServiceBuildInfo& sbi);
#endif

  PETScPrecConfigGAMGService(
      Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      std::shared_ptr<IOptionsPETScPrecConfigGAMG> options);

  /** Destructeur de la classe */
  virtual ~PETScPrecConfigGAMGService() {}

 public:
  //! Initialisation
  void configure(PC& pc, const ISpace& space, const MatrixDistribution& distribution);

  //! Check need of KSPSetUp before calling this PC configure
  virtual bool needPrematureKSPSetUp() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
