// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiDatatype.h                                               (C) 2000-2025 */
/*                                                                           */
/* Encapsulation d'un MPI_Datatype.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_MPIDATATYPE_H
#define ARCCORE_MESSAGEPASSINGMPI_MPIDATATYPE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

// TODO: a supprimer
#include "arccore/base/FatalErrorException.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Opérateurs de réduction pour les types complexes (Real2, Real3, Real2x2 et Real3x3)
class IMpiReduceOperator
{
 public:
  virtual ~IMpiReduceOperator(){}
  virtual MPI_Op reduceOperator(eReduceType rt) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Opérateur de réduction interne MPI (MPI_MAX, MPI_MIN, MPI_SUM)
class BuiltInMpiReduceOperator
: public IMpiReduceOperator
{
 public:
  MPI_Op reduceOperator(eReduceType rt) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Opérateurs de réduction pour les types classiques
template<typename RealType>
class StdMpiReduceOperator
: public IMpiReduceOperator
{
 public:
  StdMpiReduceOperator(bool is_commutative);
  void destroy();
  MPI_Op reduceOperator(eReduceType rt) override;
 private:
  MPI_Op m_min_operator;
  MPI_Op m_max_operator;
  MPI_Op m_sum_operator;
 private:
  static void ARCCORE_MPIOP_CALL _MinOperator(void* a,void* b,int* len,MPI_Datatype* type);
  static void ARCCORE_MPIOP_CALL _MaxOperator(void* a,void* b,int* len,MPI_Datatype* type);
  static void ARCCORE_MPIOP_CALL _SumOperator(void* a,void* b,int* len,MPI_Datatype* type);
  void _create(bool is_commutative);
};


template<typename RealType> inline
StdMpiReduceOperator<RealType>::
StdMpiReduceOperator(bool is_commutative)
{
  m_min_operator = MPI_OP_NULL;
  m_max_operator = MPI_OP_NULL;
  m_sum_operator = MPI_OP_NULL;
  _create(is_commutative);
}

template<typename RealType> inline
void ARCCORE_MPIOP_CALL StdMpiReduceOperator<RealType>::
_MinOperator(void* a,void* b,int* len,MPI_Datatype* type)
{
  ARCCORE_UNUSED(type);
  RealType* ra = (RealType*)a;
  RealType* rb = (RealType*)b;
  Integer s = *len;
  for( Integer i=0; i<s; ++i ){
    RealType vb = rb[i];
    RealType va = ra[i];
    rb[i] = std::min(va,vb);
  }
}

template<typename RealType> inline
void ARCCORE_MPIOP_CALL StdMpiReduceOperator<RealType>::
_MaxOperator(void* a,void* b,int* len,MPI_Datatype* type)
{
  ARCCORE_UNUSED(type);
  RealType* ra = (RealType*)a;
  RealType* rb = (RealType*)b;
  Integer s = *len;
  for( Integer i=0; i<s; ++i ){
    RealType vb = rb[i];
    RealType va = ra[i];
    rb[i] = std::max(va,vb);
  }
}

template<typename RealType> inline
void ARCCORE_MPIOP_CALL StdMpiReduceOperator<RealType>::
_SumOperator(void* a,void* b,int* len,MPI_Datatype* type)
{
  ARCCORE_UNUSED(type);
  RealType* ra = (RealType*)a;
  RealType* rb = (RealType*)b;
  Integer s = *len;
  for( Integer i=0; i<s; ++i ){
    RealType vb = rb[i];
    RealType va = ra[i];
    rb[i] = va + vb;
  }
}

template<typename RealType> inline
void StdMpiReduceOperator<RealType>::
_create(bool is_commutative)
{
  int commutative = (is_commutative) ? 1 : 0;
  MPI_Op_create(_MinOperator,commutative,&m_min_operator);
  MPI_Op_create(_MaxOperator,commutative,&m_max_operator);
  MPI_Op_create(_SumOperator,commutative,&m_sum_operator);
}

template<typename RealType> inline
void StdMpiReduceOperator<RealType>::
destroy()
{
  if (m_min_operator!=MPI_OP_NULL){
    MPI_Op_free(&m_min_operator);
    m_min_operator = MPI_OP_NULL;
  }
  if (m_max_operator!=MPI_OP_NULL){
    MPI_Op_free(&m_max_operator);
    m_max_operator = MPI_OP_NULL;
  }
  if (m_sum_operator!=MPI_OP_NULL){
    MPI_Op_free(&m_sum_operator);
    m_sum_operator = MPI_OP_NULL;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename RealType> inline
MPI_Op StdMpiReduceOperator<RealType>::
reduceOperator(eReduceType rt)
{
  MPI_Op op = MPI_OP_NULL;
  switch(rt){
  case ReduceMax: op = m_max_operator; break;
  case ReduceMin: op = m_min_operator; break;
  case ReduceSum: op = m_sum_operator; break;
  }
  if (op==MPI_OP_NULL)
    ARCCORE_FATAL("Reduce operation unknown or not implemented");
  return op;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Encapsulation d'un MPI_Datatype.
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiDatatype
{
 public:

  MpiDatatype(MPI_Datatype datatype);
  MpiDatatype(MPI_Datatype datatype,bool is_built_in,IMpiReduceOperator* reduce_operator);
  ~MpiDatatype();

 public:

  MPI_Op reduceOperator(eReduceType reduce_type)
  {
    return m_reduce_operator->reduceOperator(reduce_type);
  }
  MPI_Datatype datatype() const { return m_datatype; }

 public:

 private:

 private:

  MPI_Datatype m_datatype;
  IMpiReduceOperator* m_reduce_operator;
  bool m_is_built_in;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
