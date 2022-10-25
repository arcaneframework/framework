// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*****************************************************************************
 * IAlephPETSc.h                                                    (C) 2013 *
 *****************************************************************************/
#ifndef _ALEPH_INTERFACE_PETSC_H_
#define _ALEPH_INTERFACE_PETSC_H_

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AlephTopologyPETSc : public IAlephTopology
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
  ~AlephTopologyPETSc()
  {
    debug() << "\33[1;5;32m\t\t\t[~AlephTopologyPETSc]\33[0m";
  }

 public:
  void backupAndInitialize() {}
  void restore() {}
};

class AlephVectorPETSc : public IAlephVector
{
 public:
  AlephVectorPETSc(ITraceMng*, AlephKernel*, Integer);
  void AlephVectorCreate(void);
  void AlephVectorSet(const double*, const int*, Integer);
  int AlephVectorAssemble(void);
  void AlephVectorGet(double*, const int*, Integer);
  void writeToFile(const String);
  Real LinftyNorm(void);

 public:
  Vec m_petsc_vector;
  PetscInt jSize, jUpper, jLower;
};

class AlephMatrixPETSc : public IAlephMatrix
{
 public:
  AlephMatrixPETSc(ITraceMng*, AlephKernel*, Integer);
  void AlephMatrixCreate(void);
  void AlephMatrixSetFilled(bool);
  int AlephMatrixAssemble(void);
  void AlephMatrixFill(int, int*, int*, double*);
  Real LinftyNormVectorProductAndSub(AlephVector*, AlephVector*);
  bool isAlreadySolved(AlephVectorPETSc*, AlephVectorPETSc*,
                       AlephVectorPETSc*, Real*, AlephParams*);
  int AlephMatrixSolve(AlephVector*, AlephVector*,
                       AlephVector*, Integer&, Real*, AlephParams*);
  void writeToFile(const String);

 private:
  Mat m_petsc_matrix;
  KSP m_ksp_solver;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
class PETScAlephFactoryImpl : public AbstractService
, public IAlephFactoryImpl
{
 public:
  PETScAlephFactoryImpl(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  {}
  ~PETScAlephFactoryImpl()
  {
    for (auto* v : m_IAlephVectors )
      delete v;
    for (auto* v : m_IAlephMatrixs )
      delete v;
    for (auto* v : m_IAlephTopologys )
      delete v;
  }

 public:
  virtual void initialize() {}
  virtual IAlephTopology* createTopology(ITraceMng* tm,
                                         AlephKernel* kernel,
                                         Integer index,
                                         Integer nb_row_size)
  {
    IAlephTopology* new_topology = new AlephTopologyPETSc(tm, kernel, index, nb_row_size);
    m_IAlephTopologys.add(new_topology);
    return new_topology;
  }
  virtual IAlephVector* createVector(ITraceMng* tm,
                                     AlephKernel* kernel,
                                     Integer index)
  {
    IAlephVector* new_vector = new AlephVectorPETSc(tm, kernel, index);
    m_IAlephVectors.add(new_vector);
    return new_vector;
  }

  virtual IAlephMatrix* createMatrix(ITraceMng* tm,
                                     AlephKernel* kernel,
                                     Integer index)
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

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // _ALEPH_INTERFACE_PETSC_H_
