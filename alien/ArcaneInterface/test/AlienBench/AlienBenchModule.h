// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ALIENBENCHMODULE_H
#define ALIENBENCHMODULE_H

#include "AlienBench_axl.h"

#include <arcane/random/Uniform01.h>
#include <arcane/random/LinearCongruential.h>

#include "arcane/core/UnstructuredMeshConnectivity.h"
#include "arcane/core/ItemGenericInfoListView.h"
#include "arcane/core/MeshUtils.h"

#ifdef ALIEN_USE_SYCL
#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/accelerator/Reduce.h"
#include "arcane/accelerator/Accelerator.h"
#include "arcane/accelerator/RunCommandEnumerate.h"
#endif

class MemoryAllocationTracker;

using namespace Arcane;

class AlienBenchModule : public ArcaneAlienBenchObject
{
 public:
  //! Constructor
  AlienBenchModule(const Arcane::ModuleBuildInfo& mbi)
  : ArcaneAlienBenchObject(mbi)
  , m_uniform(m_generator)
#ifdef ALIEN_USE_SYCL
  , m_node_index_in_cells(platform::getAcceleratorHostMemoryAllocator())
  , m_runner(mbi.subDomain()->acceleratorMng()->defaultRunner())
  , m_default_queue(mbi.subDomain()->acceleratorMng()->defaultQueue())
  , m_cell_is_own (VariableBuildInfo(mbi.mesh(),"CellIsOwn"))
  , m_cell_cell_connection_index(platform::getDefaultDataAllocator())
  , m_cell_cell_connection_offset(platform::getDefaultDataAllocator())
#endif
  {
  }

  //! Destructor
  virtual ~AlienBenchModule(){};

 public:
  //! Initialization
  void init();
  //! Run the test
  void test();

 private:
  void _test(Timer& pbuild_timer,
             CellGroup& areaU,
             CellCellGroup& cell_cell_connection,
             CellCellGroup& all_cell_cell_connection,
             Arccore::UniqueArray<Arccore::Integer>& allUindex,
             Alien::Vector& vectorB,
             Alien::Vector& vectorBB,
             Alien::Vector& vectorX,
             Alien::Vector& coordX,
             Alien::Vector& coordY,
             Alien::Vector& coordZ,
             Alien::Matrix& matrixA);

  void _test(Timer& pbuild_timer,
             CellGroup& areaU,
             CellCellGroup& cell_cell_connection,
             CellCellGroup& all_cell_cell_connection,
             Arccore::UniqueArray<Arccore::Integer>& allUindex,
             Alien::BlockVector& vectorB,
             Alien::BlockVector& vectorBB,
             Alien::BlockVector& vectorX,
             Alien::Vector& coordX,
             Alien::Vector& coordY,
             Alien::Vector& coordZ,
             Alien::BlockMatrix& matrixA);

  void _fillSystemCPU(Timer& pbuild_timer,
                      CellCellGroup& cell_cell_connection,
                      CellCellGroup& all_cell_cell_connection,
                      Arccore::UniqueArray<Arccore::Integer>& allUIndex,
                      Alien::Vector& vectorB,
                      Alien::Vector& vectorBB,
                      Alien::Vector& vectorX,
                      Alien::Matrix& matrixA) ;


  void _fillSystemCPU(Timer& pbuild_timer,
                      CellCellGroup& cell_cell_connection,
                      CellCellGroup& all_cell_cell_connection,
                      Arccore::UniqueArray<Arccore::Integer>& allUIndex,
                      Alien::BlockVector& vectorB,
                      Alien::BlockVector& vectorBB,
                      Alien::BlockVector& vectorX,
                      Alien::BlockMatrix& matrixA) ;

#ifdef ALIEN_USE_SYCL
  void _testSYCLWithUSM( Timer& pbuild_timer,
                         CellGroup& areaU,
                         CellCellGroup& cell_cell_connection,
                         CellCellGroup& all_cell_cell_connection,
                         Arccore::UniqueArray<Arccore::Integer>& allUIndex,
                         Alien::Vector& vectorB,
                         Alien::Vector& vectorBB,
                         Alien::Vector& vectorX,
                         Alien::Vector& coordX,
                         Alien::Vector& coordY,
                         Alien::Vector& coordZ,
                         Alien::Matrix& matrixA);
  void _testSYCL(Timer& pbuild_timer,
                 CellGroup& areaU,
                 CellCellGroup& cell_cell_connection,
                 CellCellGroup& all_cell_cell_connection,
                 Arccore::UniqueArray<Arccore::Integer>& allUIndex,
                 Alien::Vector& vectorB,
                 Alien::Vector& vectorBB,
                 Alien::Vector& vectorX,
                 Alien::Vector& coordX,
                 Alien::Vector& coordY,
                 Alien::Vector& coordZ,
                 Alien::Matrix& matrixA);

  void _fillSystemSYCLWithUSM(Timer& pbuild_timer,
                       CellCellGroup& cell_cell_connection,
                       CellCellGroup& all_cell_cell_connection,
                       Arccore::UniqueArray<Arccore::Integer>& allUIndex,
                       Alien::Vector& vectorB,
                       Alien::Vector& vectorBB,
                       Alien::Vector& vectorX,
                       Alien::Matrix& matrixA) ;

  void _fillSystemSYCL(Timer& pbuild_timer,
                       CellCellGroup& cell_cell_connection,
                       CellCellGroup& all_cell_cell_connection,
                       Arccore::UniqueArray<Arccore::Integer>& allUIndex,
                       Alien::Vector& vectorB,
                       Alien::Vector& vectorBB,
                       Alien::Vector& vectorX,
                       Alien::Matrix& matrixA) ;
#endif
 private:
  ARCCORE_HOST_DEVICE Real funcn(Real3 x) const;
  ARCCORE_HOST_DEVICE Real funck(Real3 x) const;
  Real dii(const Cell& ci) const;
  ARCCORE_HOST_DEVICE Real dii([[maybe_unused]] Integer ci) const {
    return m_diag_coeff ;
  }
  Real fij(const Cell& ci, const Cell& cj) const;
  ARCCORE_HOST_DEVICE Real fij(Integer ci, Integer cj, Arcane::Real3 xi, Arcane::Real3 xj) const ;

  eItemKind m_stencil_kind = Arcane::IK_Face;
  bool m_homogeneous = false;
  Real m_diag_coeff = 0.;
  Real m_off_diag_coeff = 0.5;
  Real m_lambdax = 1.;
  Real m_lambday = 1.;
  Real m_lambdaz = 1.;
  Real m_alpha = 1.;
  Real m_sigma = 0.;
  IParallelMng* m_parallel_mng = nullptr;

  Arcane::CellGroup m_areaU;
  Arcane::random::MinstdRand m_generator;
  mutable Arcane::random::Uniform01<Arcane::random::MinstdRand> m_uniform;

  Alien::MatrixDistribution m_mdist;
  Alien::VectorDistribution m_vdist;

  bool                           m_use_accelerator       = false ;
  bool                           m_with_usm              = false ;
#ifdef ALIEN_USE_SYCL
  UniqueArray<Int16>             m_node_index_in_cells;
  Arcane::Accelerator::Runner*   m_runner                = nullptr;
  Arcane::Accelerator::RunQueue* m_default_queue         = nullptr;


  VariableCellInt16 m_cell_is_own; //!< Numéro du sous-domaine associé à la maille
  UnstructuredMeshConnectivityView m_connectivity_view;
  UniqueArray<Integer> m_cell_cell_connection_index;
  UniqueArray<Integer> m_cell_cell_connection_offset;
#endif
};

#endif
