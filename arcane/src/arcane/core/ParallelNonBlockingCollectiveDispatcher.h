// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelNonBlockingCollectiveDispatcher.h                   (C) 2000-2025 */
/*                                                                           */
/* Interface du gestionnaire du parallélisme sur un domaine.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_PARALLELNONBLOCKINGCOLLECTIVEDISPATCHER_H
#define ARCANE_CORE_PARALLELNONBLOCKINGCOLLECTIVEDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IParallelNonBlockingCollective.h"
#include "arcane/core/IParallelNonBlockingCollectiveDispatch.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T>
class IParallelNonBlockingCollectiveDispatchT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Redirige la gestion des messages des sous-domaines
 * suivant le type de l'argument.
 */
class ARCANE_CORE_EXPORT ParallelNonBlockingCollectiveDispatcher
: public IParallelNonBlockingCollective
{
 public:

  explicit ParallelNonBlockingCollectiveDispatcher(IParallelMng* pm);
  ~ParallelNonBlockingCollectiveDispatcher() override;

 public:

  IParallelMng* parallelMng() const override { return m_parallel_mng; }

 protected:

  void _setDispatchers(IParallelNonBlockingCollectiveDispatchT<char>* c,
                       IParallelNonBlockingCollectiveDispatchT<signed char>* sc,
                       IParallelNonBlockingCollectiveDispatchT<unsigned char>* uc,
                       IParallelNonBlockingCollectiveDispatchT<short>* s,
                       IParallelNonBlockingCollectiveDispatchT<unsigned short>* us,
                       IParallelNonBlockingCollectiveDispatchT<int>* i,
                       IParallelNonBlockingCollectiveDispatchT<unsigned int>* ui,
                       IParallelNonBlockingCollectiveDispatchT<long>* l,
                       IParallelNonBlockingCollectiveDispatchT<unsigned long>* ul,
                       IParallelNonBlockingCollectiveDispatchT<long long>* ll,
                       IParallelNonBlockingCollectiveDispatchT<unsigned long long>* ull,
#ifdef ARCANE_REAL_NOT_BUILTIN
                       IParallelNonBlockingCollectiveDispatchT<Real>* r,
#endif
                       IParallelNonBlockingCollectiveDispatchT<float>* f,
                       IParallelNonBlockingCollectiveDispatchT<double>* d,
                       IParallelNonBlockingCollectiveDispatchT<long double>* ld,
                       IParallelNonBlockingCollectiveDispatchT<Real2>* r2,
                       IParallelNonBlockingCollectiveDispatchT<Real3>* r3,
                       IParallelNonBlockingCollectiveDispatchT<Real2x2>* r22,
                       IParallelNonBlockingCollectiveDispatchT<Real3x3>* r33,
                       IParallelNonBlockingCollectiveDispatchT<HPReal>* hpr
                       );

  ITimeStats* timeStats();

 private:

  IParallelMng* m_parallel_mng = nullptr;

 public:

#define ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(field,type)          \
public: \
  virtual Request allGather(ConstArrayView<type> send_buf,ArrayView<type> recv_buf); \
  virtual Request gather(ConstArrayView<type> send_buf,ArrayView<type> recv_buf,Integer rank); \
  virtual Request allGatherVariable(ConstArrayView<type> send_buf,Array<type>& recv_buf); \
  virtual Request gatherVariable(ConstArrayView<type> send_buf,Array<type>& recv_buf,Integer rank); \
  virtual Request scatterVariable(ConstArrayView<type> send_buf,ArrayView<type> recv_buf,Integer root); \
  virtual Request allReduce(eReduceType rt,ConstArrayView<type> send_buf,ArrayView<type> v); \
  virtual Request broadcast(ArrayView<type> send_buf,Int32 rank); \
  virtual Request allToAll(ConstArrayView<type> send_buf,ArrayView<type> recv_buf,Integer count); \
  virtual Request allToAllVariable(ConstArrayView<type> send_buf,Int32ConstArrayView send_count, \
                                Int32ConstArrayView send_index,ArrayView<type> recv_buf, \
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index); \
protected:                                                              \
  IParallelNonBlockingCollectiveDispatchT<type>* field;                                   
  

  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_char,char)
  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_unsigned_char,unsigned char)
  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_signed_char,signed char)
  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_short,short)
  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_unsigned_short,unsigned short)
  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_int,int)
  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_unsigned_int,unsigned int)
  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_long,long)
  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_unsigned_long,unsigned long)
  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_long_long,long long)
  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_unsigned_long_long,unsigned long long)
  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_float,float)
  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_double,double)
  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_long_double,long double)
#ifdef ARCANE_REAL_NOT_BUILTIN
  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_real,Real)
#endif
  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_real2,Real2)
  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_real3,Real3)
  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_real2x2,Real2x2)
  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_real3x3,Real3x3)
  ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE(m_hpreal,HPReal)

#undef ARCANE_PARALLEL_NONBLOCKINGCOLLECTIVE_DISPATCH_PROTOTYPE

 public:

  virtual IParallelNonBlockingCollectiveDispatchT<char>* dispatcher(char*);
  virtual IParallelNonBlockingCollectiveDispatchT<signed char>* dispatcher(signed char*);
  virtual IParallelNonBlockingCollectiveDispatchT<unsigned char>* dispatcher(unsigned char*);
  virtual IParallelNonBlockingCollectiveDispatchT<short>* dispatcher(short*);
  virtual IParallelNonBlockingCollectiveDispatchT<unsigned short>* dispatcher(unsigned short*);
  virtual IParallelNonBlockingCollectiveDispatchT<int>* dispatcher(int*);
  virtual IParallelNonBlockingCollectiveDispatchT<unsigned int>* dispatcher(unsigned int*);
  virtual IParallelNonBlockingCollectiveDispatchT<long>* dispatcher(long*);
  virtual IParallelNonBlockingCollectiveDispatchT<unsigned long>* dispatcher(unsigned long*);
  virtual IParallelNonBlockingCollectiveDispatchT<long long>* dispatcher(long long*);
  virtual IParallelNonBlockingCollectiveDispatchT<unsigned long long>* dispatcher(unsigned long long*);
#ifdef ARCANE_REAL_NOT_BUILTIN
  virtual IParallelNonBlockingCollectiveDispatchT<Real>* dispatcher(Real*);
#endif
  virtual IParallelNonBlockingCollectiveDispatchT<float>* dispatcher(float*);
  virtual IParallelNonBlockingCollectiveDispatchT<double>* dispatcher(double*);
  virtual IParallelNonBlockingCollectiveDispatchT<long double>* dispatcher(long double*);
  virtual IParallelNonBlockingCollectiveDispatchT<Real2>* dispatcher(Real2*);
  virtual IParallelNonBlockingCollectiveDispatchT<Real3>* dispatcher(Real3*);
  virtual IParallelNonBlockingCollectiveDispatchT<Real2x2>* dispatcher(Real2x2*);
  virtual IParallelNonBlockingCollectiveDispatchT<Real3x3>* dispatcher(Real3x3*);
  virtual IParallelNonBlockingCollectiveDispatchT<HPReal>* dispatcher(HPReal*);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
