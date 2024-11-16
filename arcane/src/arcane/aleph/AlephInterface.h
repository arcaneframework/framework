// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephInterface.h                                            (C) 2000-2024 */
/*                                                                           */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ALEPH_ALEPHINTERFACE_H
#define ARCANE_ALEPH_ALEPHINTERFACE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/aleph/AlephKernel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AlephKernel;
class IAlephVector;
class AlephVector;

/******************************************************************************
 * IAlephTopology
 *****************************************************************************/
class IAlephTopology
: public TraceAccessor
{
 public:

  IAlephTopology(ITraceMng* tm,
                 AlephKernel* kernel,
                 Integer index,
                 Integer nb_row_size)
  : TraceAccessor(tm)
  , m_index(index)
  , m_kernel(kernel)
  , m_participating_in_solver(kernel->subParallelMng(index) != NULL)
  {
    ARCANE_UNUSED(nb_row_size);
    debug() << "\33[1;34m\t\t[IAlephTopology] NEW IAlephTopology"
            << "\33[0m";
    debug() << "\33[1;34m\t\t[IAlephTopology] m_participating_in_solver="
            << m_participating_in_solver << "\33[0m";
  }
  ~IAlephTopology()
  {
    debug() << "\33[1;5;34m\t\t[~IAlephTopology]"
            << "\33[0m";
  }

 public:

  virtual void backupAndInitialize() = 0;
  virtual void restore() = 0;

 protected:

  Integer m_index;
  AlephKernel* m_kernel;
  bool m_participating_in_solver;
};

/******************************************************************************
 * IAlephVector
 *****************************************************************************/
class IAlephVector
: public TraceAccessor
{
 public:

  IAlephVector(ITraceMng* tm,
               AlephKernel* kernel,
               Integer index)
  : TraceAccessor(tm)
  , m_index(index)
  , m_kernel(kernel)
  {
    debug() << "\33[1;34m\t\t[IAlephVector] NEW IAlephVector"
            << "\33[0m";
  }
  ~IAlephVector()
  {
    debug() << "\33[1;5;34m\t\t[~IAlephVector]"
            << "\33[0m";
  }

 public:

  virtual void AlephVectorCreate(void) = 0;
  virtual void AlephVectorSet(const double*, const AlephInt*, Integer) = 0;
  virtual int AlephVectorAssemble(void) = 0;
  virtual void AlephVectorGet(double*, const AlephInt*, Integer) = 0;
  virtual void writeToFile(const String) = 0;

 protected:

  Integer m_index;
  AlephKernel* m_kernel;
};

/******************************************************************************
 * IAlephMatrix
 *****************************************************************************/
class IAlephMatrix
: public TraceAccessor
{
 public:

  IAlephMatrix(ITraceMng* tm,
               AlephKernel* kernel,
               Integer index)
  : TraceAccessor(tm)
  , m_index(index)
  , m_kernel(kernel)
  {
    debug() << "\33[1;34m\t\t[IAlephMatrix] NEW IAlephMatrix"
            << "\33[0m";
  }
  ~IAlephMatrix()
  {
    debug() << "\33[1;5;34m\t\t[~IAlephMatrix]"
            << "\33[0m";
  }

 public:

  virtual void AlephMatrixCreate(void) = 0;
  virtual void AlephMatrixSetFilled(bool) = 0;
  virtual int AlephMatrixAssemble(void) = 0;
  virtual void AlephMatrixFill(int, AlephInt*, AlephInt*, double*) = 0;
  virtual int AlephMatrixSolve(AlephVector*,
                               AlephVector*,
                               AlephVector*,
                               Integer&,
                               Real*,
                               AlephParams*) = 0;
  virtual void writeToFile(const String) = 0;

 protected:

  Integer m_index;
  AlephKernel* m_kernel;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IAlephFactory
: public TraceAccessor
{
 public:

  explicit IAlephFactory(ITraceMng* tm)
  : TraceAccessor(tm)
  {
    debug() << "\33[1;34m[IAlephFactory] NEW IAlephFactory"
            << "\33[0m";
  }
  ~IAlephFactory()
  {
    debug() << "\33[1;5;34m[~IAlephFactory]"
            << "\33[0m";
  }
  virtual IAlephTopology* GetTopology(AlephKernel*, Integer, Integer) = 0;
  virtual IAlephVector* GetVector(AlephKernel*, Integer) = 0;
  virtual IAlephMatrix* GetMatrix(AlephKernel*, Integer) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une fabrique d'implémentation pour Aleph.
 *
 * Cette interface est utilisée par AlephFactory pour choisir la
 * bibliothèque d'algèbre linéaire sous-jacente (par exemple sloop, hypre,...)
 */
class IAlephFactoryImpl
{
 public:

  virtual ~IAlephFactoryImpl() {}
  virtual void initialize() = 0;
  virtual IAlephTopology* createTopology(ITraceMng*, AlephKernel*, Integer, Integer) = 0;
  virtual IAlephVector* createVector(ITraceMng*, AlephKernel*, Integer) = 0;
  virtual IAlephMatrix* createMatrix(ITraceMng*, AlephKernel*, Integer) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // _I_ALEPH_INTERFACE_H_
