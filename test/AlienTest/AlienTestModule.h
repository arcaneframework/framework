#ifndef ALIENTESTMODULE_H
#define ALIENTESTMODULE_H

#include "AlienTest_axl.h"

#include <arcane/random/Uniform01.h>
#include <arcane/random/LinearCongruential.h>

class MemoryAllocationTracker;

using namespace Arcane;

class AlienTestModule : public ArcaneAlienTestObject
{
 public:
  //! Constructor
  AlienTestModule(const Arcane::ModuleBuildInfo& mbi)
  : ArcaneAlienTestObject(mbi)
  , m_memory_tracker(NULL)
  , m_uniform(m_generator)
  {
  }

  //! Destructor
  virtual ~AlienTestModule(){};

 public:
  //! Initialization
  void init();
  //! Run the test
  void test();

 private:
  void buildAndFillInVector(Alien::VectorData& vectorB, const double& value);

  void buildAndFillInBlockVector(Alien::VectorData& vectorB, const double& value);

  void buildAndFillInBlockVector(Alien::VectorData& vectorB,
      ConstArrayView<Integer> allXIndex, ConstArrayView<Real> value);
  void buildAndFillInBlockVector(Alien::VectorData& vectorB,
      ConstArrayView<Integer> allUIndex, ConstArray2View<Integer> allPIndex,
      ConstArrayView<Integer> allXIndex, ConstArrayView<Real> value);

  template <typename Profiler>
  void profileMatrix(const CellCellGroup& cell_cell_connection, const ItemGroup areaP,
      const UniqueArray2<Integer>& allPIndex,
      const Arccore::UniqueArray<Integer>& allUIndex,
      const Arccore::UniqueArray<Integer>& allXIndex, Profiler& profiler);

  //  template<typename StreamBuilderT>
  //  void streamProfileMatrix(const CellCellGroup& cell_cell_connection,
  //      const ItemGroup areaU,
  //      const ItemGroup areaP,
  //      const UniqueArray2<Integer>& allPIndex,
  //      const Alien::IntegerUniqueArray& allUIndex,
  //      const Alien::IntegerUniqueArray& allXIndex,
  //      StreamBuilderT & inserters);
  //
  //  void streamFillInMatrix(const CellCellGroup& cell_cell_connection,
  //      const ItemGroup areaP,
  //      const UniqueArray2<Integer>& allPIndex,
  //      const Alien::IntegerUniqueArray& allUIndex,
  //      const Alien::IntegerUniqueArray& allXIndex,
  //      Alien::StreamMatrixBuilder & inserters);
  //
  //  void streamBlockFillInMatrix(const CellCellGroup& cell_cell_connection,
  //      const ItemGroup areaP,
  //      const UniqueArray2<Integer>& allPIndex,
  //      const Alien::IntegerUniqueArray& allUIndex,
  //      const Alien::IntegerUniqueArray& allXIndex,
  //      Alien::StreamMatrixBuilder & inserters);
  //
  //  void streamBlockFillInMatrix(const CellCellGroup& cell_cell_connection,
  //      const ItemGroup areaP,
  //      const UniqueArray2<Integer>& allPIndex,
  //      const Alien::IntegerUniqueArray& allUIndex,
  //      const Alien::IntegerUniqueArray& allXIndex,
  //      Alien::StreamVBlockMatrixBuilder & inserters);

  template <typename Builder>
  void profiledFillInMatrix(const CellCellGroup& cell_cell_connection,
      const ItemGroup areaP, const UniqueArray2<Integer>& allPIndex,
      const Arccore::UniqueArray<Integer>& allUIndex,
      const Arccore::UniqueArray<Integer>& allXIndex, Builder& builder);

  template <typename Builder>
  void fillInMatrix(const CellCellGroup& cell_cell_connection, const ItemGroup areaP,
      const UniqueArray2<Integer>& allPIndex,
      const Arccore::UniqueArray<Integer>& allUIndex,
      const Arccore::UniqueArray<Integer>& allXIndex, Builder& builder);

  Alien::SolverStatus solve(Alien::ILinearSolver* solver, Alien::MatrixData& matrixA,
      Alien::VectorData& vectorB, Alien::VectorData& vectorX,
      AlienTestOptionTypes::eBuilderType builderType);

  void vectorVariableUpdate(
      Alien::VectorData& vectorB, Alien::ArcaneTools::IIndexManager::Entry indexSetU);

  void checkVectorValues(Alien::IVector& VectorX, const double& value);

  void checkDotProductWithManyAlgebra(
      Alien::IVector& vectorB, Alien::IVector& vectorX, Alien::Space& space);

 private:
  eItemKind m_stencil_kind;
  Real m_diag_coeff;
  Integer m_vect_size;
  IParallelMng* m_parallel_mng;
  Integer m_n_extra_indices;

 private:
  MemoryAllocationTracker* m_memory_tracker;
  Real getAllocatedMemory();
  Real getMaxMemory();
  void resetMaxMemory();
  Int64 getAllocationCount();

  // Type de statistiques
  class BuildingStat
  {
   public:
    BuildingStat(AlienTestModule* owner, Arcane::Timer& timer)
    : m_owner(owner)
    , m_timer(timer)
    , memory(0)
    , memory_max(0)
    , time(0)
    , allocation_count(0)
    {
    }
    void start()
    {
      memory = m_owner->getAllocatedMemory();
      m_owner->resetMaxMemory();
      m_timer.start();
      allocation_count = m_owner->getAllocationCount();
    }
    void stop()
    {
      m_timer.stop();
      memory_max = m_owner->getMaxMemory() - memory;
      memory = m_owner->getAllocatedMemory() - memory;
      time = m_timer.lastActivationTime();
      allocation_count = m_owner->getAllocationCount() - allocation_count;
    }

   private:
    AlienTestModule* m_owner;
    Arcane::Timer& m_timer;

   public:
    Real memory, memory_max, time;
    Int64 allocation_count;
  };

  Real fij(const Cell& ci, const Cell& cj);

  Arcane::CellGroup m_areaP;
  Arcane::CellGroup m_areaT;
  Arcane::random::MinstdRand m_generator;
  mutable Arcane::random::Uniform01<Arcane::random::MinstdRand> m_uniform;

  Alien::MatrixDistribution m_mdist;
  Alien::VectorDistribution m_vdist;
};

#endif
