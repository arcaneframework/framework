// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiParallelDispatch.cc                                      (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de parallélisme utilisant MPI.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/HPReal.h"
#include "arcane/utils/APReal.h"

#include "arcane/IParallelMng.h"

#include "arcane/parallel/mpi/MpiDatatype.h"
#include "arcane/parallel/mpi/MpiAdapter.h"
#include "arcane/parallel/mpi/MpiParallelDispatch.h"
#include "arcane/parallel/mpi/MpiLock.h"

#include "arccore/message_passing/Messages.h"

#include "arccore/message_passing_mpi/MpiTypeDispatcherImpl.h"

#include <limits>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace MP = ::Arccore::MessagePassing;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> MpiParallelDispatchT<Type>::
MpiParallelDispatchT(ITraceMng* tm,IMessagePassingMng* parallel_mng,MpiAdapter* adapter,MpiDatatype* datatype)
: TraceAccessor(tm)
, m_mp_dispatcher(new MP::Mpi::MpiTypeDispatcher<Type>(parallel_mng,adapter,datatype))
, m_min_max_sum_datatype(MPI_DATATYPE_NULL)
, m_min_max_sum_operator(MPI_OP_NULL)
{
  _initialize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> MpiParallelDispatchT<Type>::
~MpiParallelDispatchT()
{
  finalize();
  delete m_mp_dispatcher;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/ 

template<class Type> void MpiParallelDispatchT<Type>::
finalize()
{
  if (m_min_max_sum_datatype!=MPI_DATATYPE_NULL){
    MPI_Type_free(&m_min_max_sum_datatype);
    m_min_max_sum_datatype = MPI_DATATYPE_NULL;
  }
  if (m_min_max_sum_operator!=MPI_OP_NULL){
    MPI_Op_free(&m_min_max_sum_operator);
    m_min_max_sum_operator = MPI_OP_NULL;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void MpiParallelDispatchT<Type>::
_initialize()
{
  MinMaxSumInfo mmsi;

  int blen[2];
  MPI_Aint indices[2];
  MPI_Datatype oldtypes[2];

  blen[0] = 2;
  indices[0] = 0;
  oldtypes[0] = MpiBuiltIn::datatype(Integer());

  blen[1] = 3;
  indices[1] = (char*)&mmsi.m_min_value - (char*)&mmsi;
  oldtypes[1] = _mpiDatatype();

  MPI_Type_create_struct(2,blen,indices,oldtypes,&m_min_max_sum_datatype);
  MPI_Type_commit(&m_min_max_sum_datatype);

  MPI_Op_create(_MinMaxSumOperator,1,&m_min_max_sum_operator);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void ARCANE_MPIOP_CALL MpiParallelDispatchT<Type>::
_MinMaxSumOperator(void* a,void* b, int* len,MPI_Datatype* type)
{
  ARCANE_UNUSED(type);

  Integer n = *len;
  MinMaxSumInfo * va = static_cast<MinMaxSumInfo*>(a);
  MinMaxSumInfo * vb = static_cast<MinMaxSumInfo*>(b);
  for(Integer i=0;i<n;++i) {
    MinMaxSumInfo& ma = va[i];
    MinMaxSumInfo& mb = vb[i];
    // Il faut bien etre certain qu'en cas de valeurs egales
    // le rang retourne est le meme pour tout le monde
    if (ma.m_min_value==mb.m_min_value){
      mb.m_min_rank = math::min(mb.m_min_rank,ma.m_min_rank);
    }
    else if (ma.m_min_value<mb.m_min_value){
      mb.m_min_value = ma.m_min_value;
      mb.m_min_rank = ma.m_min_rank;
    }
    if (mb.m_max_value==ma.m_max_value){
   mb.m_max_rank = math::min(mb.m_max_rank,ma.m_max_rank);
    }
    else if (mb.m_max_value<ma.m_max_value){
      mb.m_max_value = ma.m_max_value;
      mb.m_max_rank = ma.m_max_rank;
    }
    mb.m_sum_value = (Type)(ma.m_sum_value + mb.m_sum_value);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void MpiParallelDispatchT<Type>::
computeMinMaxSumNoInit(Type& min_val,Type& max_val,Type& sum_val,
                       Int32& min_rank,Int32& max_rank)
{
  MinMaxSumInfo mmsi;
  mmsi.m_min_rank = min_rank;
  mmsi.m_max_rank = max_rank;
  mmsi.m_min_value = min_val;
  mmsi.m_max_value = max_val;
  mmsi.m_sum_value = sum_val;
  MinMaxSumInfo mmsi_ret;
  _adapter()->allReduce(&mmsi,&mmsi_ret,1,m_min_max_sum_datatype,
                        m_min_max_sum_operator);
  min_val = mmsi_ret.m_min_value;
  max_val = mmsi_ret.m_max_value;
  sum_val = mmsi_ret.m_sum_value;
  min_rank = mmsi_ret.m_min_rank;
  max_rank = mmsi_ret.m_max_rank;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void MpiParallelDispatchT<Type>::
computeMinMaxSum(Type val,Type& min_val,Type& max_val,Type& sum_val,
                 Int32& min_rank,Int32& max_rank)
{
  min_rank = _adapter()->commRank();
  max_rank = _adapter()->commRank();
  min_val = val;
  max_val = val;
  sum_val = val;
  computeMinMaxSumNoInit(min_val,max_val,sum_val,min_rank,max_rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void MpiParallelDispatchT<Type>::
computeMinMaxSum(ConstArrayView<Type> values,
                 ArrayView<Type> min_values,
                 ArrayView<Type> max_values,
                 ArrayView<Type> sum_values,
                 ArrayView<Int32> min_ranks,
                 ArrayView<Int32> max_ranks)
{
  const Integer n = values.size();
  UniqueArray<MinMaxSumInfo> mmsi(n);
  const Integer comm_rank = m_mp_dispatcher->adapter()->commRank();
  for(Integer i=0;i<n;++i) {
    mmsi[i].m_min_rank = comm_rank;
    mmsi[i].m_max_rank = comm_rank;
    mmsi[i].m_min_value = values[i];
    mmsi[i].m_max_value = values[i];
    mmsi[i].m_sum_value = values[i];
  }  
  UniqueArray<MinMaxSumInfo> mmsi_ret(n);
  _adapter()->allReduce(mmsi.data(),mmsi_ret.data(),n,m_min_max_sum_datatype,
                        m_min_max_sum_operator);
  for(Integer i=0;i<n;++i) {
    min_values[i] = mmsi_ret[i].m_min_value;
    max_values[i] = mmsi_ret[i].m_max_value;
    sum_values[i] = mmsi_ret[i].m_sum_value;
    min_ranks[i] = mmsi_ret[i].m_min_rank;
    max_ranks[i] = mmsi_ret[i].m_max_rank;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void MpiParallelDispatchT<Type>::
sendRecv(ConstArrayView<Type> send_buffer,ArrayView<Type> recv_buffer,Int32 rank)
{
  MPI_Datatype type = _mpiDatatype();
  _adapter()->directSendRecv(send_buffer.data(),send_buffer.size(),
                             recv_buffer.data(),recv_buffer.size(),
                            rank,sizeof(Type),type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Type MpiParallelDispatchT<Type>::
scan(eReduceType op,Type send_buf)
{
  MPI_Datatype type = _mpiDatatype();
  Type recv_buf = send_buf;
  _adapter()->scan(&send_buf,&recv_buf,1,type,_mpiReduceOperator(op));
  return recv_buf;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void MpiParallelDispatchT<Type>::
scan(eReduceType op,ArrayView<Type> send_buf)
{
  MPI_Datatype type = _mpiDatatype();
  Integer s = send_buf.size();
  UniqueArray<Type> recv_buf(s);
  _adapter()->scan(send_buf.data(),recv_buf.data(),s,type,_mpiReduceOperator(op));
  send_buf.copy(recv_buf);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> MPI_Datatype  MpiParallelDispatchT<Type>::
_mpiDatatype()
{
  return m_mp_dispatcher->datatype()->datatype();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> MPI_Op MpiParallelDispatchT<Type>::
_mpiReduceOperator(eReduceType rt)
{
  return m_mp_dispatcher->datatype()->reduceOperator(rt);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> MpiAdapter* MpiParallelDispatchT<Type>::
_adapter()
{
  return m_mp_dispatcher->adapter();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> MpiDatatype* MpiParallelDispatchT<Type>::
datatype() const
{
  return m_mp_dispatcher->datatype();
}

template<class Type> ITypeDispatcher<Type>* MpiParallelDispatchT<Type>::
toArccoreDispatcher()
{
  return m_mp_dispatcher;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class MpiParallelDispatchT<char>;
template class MpiParallelDispatchT<signed char>;
template class MpiParallelDispatchT<unsigned char>;
template class MpiParallelDispatchT<short>;
template class MpiParallelDispatchT<unsigned short>;
template class MpiParallelDispatchT<int>;
template class MpiParallelDispatchT<unsigned int>;
template class MpiParallelDispatchT<long>;
template class MpiParallelDispatchT<unsigned long>;
template class MpiParallelDispatchT<long long>;
template class MpiParallelDispatchT<unsigned long long>;
template class MpiParallelDispatchT<float>;
template class MpiParallelDispatchT<double>;
template class MpiParallelDispatchT<long double>;
template class MpiParallelDispatchT<APReal>;
template class MpiParallelDispatchT<Real2>;
template class MpiParallelDispatchT<Real3>;
template class MpiParallelDispatchT<Real2x2>;
template class MpiParallelDispatchT<Real3x3>;
template class MpiParallelDispatchT<HPReal>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{
using namespace Arcane;
template class MpiTypeDispatcher<APReal>;
template class MpiTypeDispatcher<Real2>;
template class MpiTypeDispatcher<Real3>;
template class MpiTypeDispatcher<Real2x2>;
template class MpiTypeDispatcher<Real3x3>;
template class MpiTypeDispatcher<HPReal>;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
