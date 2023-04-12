// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IAlephPETSc.h                                               (C) 2000-2023 */
/*                                                                           */
/* Interface de Aleph pour PETSc.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ALEPH_IALEPHPETSC_H
#define ARCANE_ALEPH_IALEPHPETSC_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AlephTopologyPETSc
: public IAlephTopology
{
 public:

  AlephTopologyPETSc(ITraceMng* tm,
                     AlephKernel* kernel,
                     Integer index,
                     Integer nb_row_size)
  : IAlephTopology(tm, kernel, index, nb_row_size)
  {
    if (!m_participating_in_solver) {
      debug() << "\33[1;32m\t[AlephTopologyPETSc] Not concerned with this solver, returning\33[0m";
      return;
    }
    debug() << "\33[1;32m\t\t[AlephTopologyPETSc] @" << this << "\33[0m";
    if (!m_kernel->isParallel()) {
      PETSC_COMM_WORLD = PETSC_COMM_SELF;
    }
    else {
      PETSC_COMM_WORLD = *(MPI_Comm*)(kernel->subParallelMng(index)->getMPICommunicator());
    }
    PetscInitializeNoArguments();
  }
  ~AlephTopologyPETSc() override
  {
    debug() << "\33[1;5;32m\t\t\t[~AlephTopologyPETSc]\33[0m";
  }

 public:

  void backupAndInitialize() override {}
  void restore() override {}
};

class AlephVectorPETSc
: public IAlephVector
{
 public:

  AlephVectorPETSc(ITraceMng*, AlephKernel*, Integer);
  void AlephVectorCreate(void) override;
  void AlephVectorSet(const double*, const int*, Integer) override;
  int AlephVectorAssemble(void) override;
  void AlephVectorGet(double*, const int*, Integer) override;
  void writeToFile(const String) override;
  Real LinftyNorm(void);

 public:

  Vec m_petsc_vector;
  PetscInt jSize, jUpper, jLower;
};

class AlephMatrixPETSc
: public IAlephMatrix
{
 public:

  AlephMatrixPETSc(ITraceMng*, AlephKernel*, Integer);
  void AlephMatrixCreate(void) override;
  void AlephMatrixSetFilled(bool) override;
  int AlephMatrixAssemble(void) override;
  void AlephMatrixFill(int, int*, int*, double*) override;
  Real LinftyNormVectorProductAndSub(AlephVector*, AlephVector*);
  bool isAlreadySolved(AlephVectorPETSc*, AlephVectorPETSc*,
                       AlephVectorPETSc*, Real*, AlephParams*);
  int AlephMatrixSolve(AlephVector*, AlephVector*,
                       AlephVector*, Integer&, Real*, AlephParams*) override;
  void writeToFile(const String) override;

 private:

  Mat m_petsc_matrix;
  KSP m_ksp_solver;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
class PETScAlephFactoryImpl
: public AbstractService
, public IAlephFactoryImpl
{
 public:

  explicit PETScAlephFactoryImpl(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  {}
  ~PETScAlephFactoryImpl() override
  {
    for (auto* v : m_IAlephVectors)
      delete v;
    for (auto* v : m_IAlephMatrixs)
      delete v;
    for (auto* v : m_IAlephTopologys)
      delete v;
  }

 public:

  void initialize() override {}
  IAlephTopology* createTopology(ITraceMng* tm,
                                 AlephKernel* kernel,
                                 Integer index,
                                 Integer nb_row_size) override
  {
    IAlephTopology* new_topology = new AlephTopologyPETSc(tm, kernel, index, nb_row_size);
    m_IAlephTopologys.add(new_topology);
    return new_topology;
  }
  IAlephVector* createVector(ITraceMng* tm,
                             AlephKernel* kernel,
                             Integer index) override
  {
    IAlephVector* new_vector = new AlephVectorPETSc(tm, kernel, index);
    m_IAlephVectors.add(new_vector);
    return new_vector;
  }

  IAlephMatrix* createMatrix(ITraceMng* tm,
                             AlephKernel* kernel,
                             Integer index) override
  {
    IAlephMatrix* new_matrix = new AlephMatrixPETSc(tm, kernel, index);
    m_IAlephMatrixs.add(new_matrix);
    return new_matrix;
  }

 private:

  UniqueArray<IAlephVector*> m_IAlephVectors;
  UniqueArray<IAlephMatrix*> m_IAlephMatrixs;
  UniqueArray<IAlephTopology*> m_IAlephTopologys;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
