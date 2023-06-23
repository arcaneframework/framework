// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiParallelDispatch.cc                                      (C) 2000-2023 */
/*                                                                           */
/* Gestionnaire de parallélisme utilisant les threads et MPI.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/APReal.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/IThreadBarrier.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/MeshVariableRef.h"
#include "arcane/IParallelMng.h"
#include "arcane/ItemGroup.h"
#include "arcane/IMesh.h"
#include "arcane/IBase.h"

#include "arcane/parallel/mpithread/HybridParallelDispatch.h"
#include "arcane/parallel/mpithread/HybridParallelMng.h"
#include "arcane/parallel/mpithread/HybridMessageQueue.h"
#include "arcane/parallel/mpi/MpiParallelMng.h"
#include "arcane/parallel/mpi/MpiParallelDispatch.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//TODO: Fusionner avec ce qui est possible dans SharedMemoryParallelDispatch

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> HybridParallelDispatch<Type>::
HybridParallelDispatch(ITraceMng* tm,HybridParallelMng* pm,HybridMessageQueue* message_queue,
                       ArrayView<HybridParallelDispatch<Type>*> all_dispatchs)
: TraceAccessor(tm)
, m_parallel_mng(pm)
, m_local_rank(pm->localRank())
, m_local_nb_rank(pm->localNbRank())
, m_global_rank(pm->commRank())
, m_global_nb_rank(pm->commSize())
, m_mpi_rank(pm->mpiParallelMng()->commRank())
, m_mpi_nb_rank(pm->mpiParallelMng()->commSize())
, m_all_dispatchs(all_dispatchs)
, m_message_queue(message_queue)
, m_mpi_dispatcher(0)
{
  m_reduce_infos.m_index = 0;

  // Ce tableau a été dimensionné par le créateur de cette instance.
  // Il faut juste mettre à jour la valeur correspondant à son rang
  m_all_dispatchs[m_local_rank] = this;

  // Récupère le dispatcher MPI pour ce type.
  MpiParallelMng* mpi_pm = pm->mpiParallelMng();
  IParallelDispatchT<Type>* pd = mpi_pm->dispatcher((Type*)nullptr);
  if (!pd)
    ARCANE_FATAL("null dispatcher");

  m_mpi_dispatcher = dynamic_cast<MpiParallelDispatchT<Type>*>(pd);
  if (!m_mpi_dispatcher)
    ARCANE_FATAL("null mpi dispatcher");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> HybridParallelDispatch<Type>::
~HybridParallelDispatch()
{
  finalize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/ 

template<class Type> void HybridParallelDispatch<Type>::
finalize()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T>
class _ThreadIntegralType
{
 public:
  typedef FalseType IsIntegral;
};

#define ARCANE_DEFINE_INTEGRAL_TYPE(datatype)\
template<>\
class _ThreadIntegralType<datatype>\
{\
 public:\
  typedef TrueType IsIntegral;\
}

ARCANE_DEFINE_INTEGRAL_TYPE(long long);
ARCANE_DEFINE_INTEGRAL_TYPE(long);
ARCANE_DEFINE_INTEGRAL_TYPE(int);
ARCANE_DEFINE_INTEGRAL_TYPE(short);
ARCANE_DEFINE_INTEGRAL_TYPE(unsigned long long);
ARCANE_DEFINE_INTEGRAL_TYPE(unsigned long);
ARCANE_DEFINE_INTEGRAL_TYPE(unsigned int);
ARCANE_DEFINE_INTEGRAL_TYPE(unsigned short);
ARCANE_DEFINE_INTEGRAL_TYPE(double);
ARCANE_DEFINE_INTEGRAL_TYPE(float);
ARCANE_DEFINE_INTEGRAL_TYPE(HPReal);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace{

template<class Type> void 
 _computeMinMaxSum2(ArrayView<HybridParallelDispatch<Type>*> all_dispatchs,
                    Int32 my_rank,Type& min_val,Type& max_val,Type& sum_val,
                    Int32& min_rank,Int32& max_rank,Int32 nb_rank,FalseType)
{
  ARCANE_UNUSED(all_dispatchs);
  ARCANE_UNUSED(my_rank);
  ARCANE_UNUSED(min_val);
  ARCANE_UNUSED(max_val);
  ARCANE_UNUSED(sum_val);
  ARCANE_UNUSED(min_rank);
  ARCANE_UNUSED(max_rank);
  ARCANE_UNUSED(nb_rank);

  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void 
 _computeMinMaxSum2(ArrayView<HybridParallelDispatch<Type>*> all_dispatchs,
                    Int32 my_rank,Type& min_val,Type& max_val,Type& sum_val,
                    Int32& min_rank,Int32& max_rank,Int32 nb_rank,TrueType)
{
  ARCANE_UNUSED(my_rank);

  HybridParallelDispatch<Type>* mtpd0 = all_dispatchs[0];
  Type cval0 = mtpd0->m_reduce_infos.reduce_value;
  Type _min_val = cval0;
  Type _max_val = cval0;
  Type _sum_val = cval0;
  Integer _min_rank = 0;
  Integer _max_rank = 0;
  for( Integer i=1; i<nb_rank; ++i ){
    HybridParallelDispatch<Type>* mtpd = all_dispatchs[i];
    Type cval = mtpd->m_reduce_infos.reduce_value;
    Int32 grank = mtpd->globalRank();
    if (cval<_min_val){
      _min_val = cval;
      _min_rank = grank;
    }
    if (_max_val<cval){
      _max_val = cval;
      _max_rank = grank;
    }
    _sum_val = (Type)(_sum_val + cval);
  }
  min_val = _min_val;
  max_val = _max_val;
  sum_val = _sum_val;
  min_rank = _min_rank;
  max_rank = _max_rank;
}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void HybridParallelDispatch<Type>::
computeMinMaxSum(Type val,Type& min_val,Type& max_val,Type& sum_val,
                 Int32& min_rank,Int32& max_rank)
{
  typedef typename _ThreadIntegralType<Type>::IsIntegral IntegralType;
  m_reduce_infos.reduce_value = val;
  _collectiveBarrier();
  _computeMinMaxSum2(m_all_dispatchs,m_global_rank,min_val,max_val,sum_val,min_rank,max_rank,m_local_nb_rank,IntegralType());
  if (m_local_rank==0){
    /*pinfo() << "COMPUTE_MIN_MAX_SUM_B rank=" << m_global_rank
            << " min_rank=" << min_rank
            << " max_rank=" << max_rank
            << " min_val=" << min_val
            << " max_val=" << max_val
            << " sum_val=" << sum_val;*/
    m_mpi_dispatcher->computeMinMaxSumNoInit(min_val,max_val,sum_val,min_rank,max_rank);
    /*pinfo() << "COMPUTE_MIN_MAX_SUM_A rank=" << m_global_rank
            << " min_rank=" << min_rank
            << " max_rank=" << max_rank;*/

    m_min_max_sum_infos.m_min_value = min_val;
    m_min_max_sum_infos.m_max_value = max_val;
    m_min_max_sum_infos.m_sum_value = sum_val;
    m_min_max_sum_infos.m_min_rank = min_rank;
    m_min_max_sum_infos.m_max_rank = max_rank;
  }
  _collectiveBarrier();
  m_min_max_sum_infos = m_all_dispatchs[0]->m_min_max_sum_infos;
  min_val = m_min_max_sum_infos.m_min_value;
  max_val = m_min_max_sum_infos.m_max_value;
  sum_val = m_min_max_sum_infos.m_sum_value;
  min_rank = m_min_max_sum_infos.m_min_rank;
  max_rank = m_min_max_sum_infos.m_max_rank;
  _collectiveBarrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void HybridParallelDispatch<Type>::
computeMinMaxSum(ConstArrayView<Type> values,
                 ArrayView<Type> min_values,
                 ArrayView<Type> max_values,
                 ArrayView<Type> sum_values,
                 ArrayView<Int32> min_ranks,
                 ArrayView<Int32> max_ranks)
{
  // Implémentation sous-optimale qui ne vectorise pas le calcul
  // (c'est actuellement un copier-coller d'au-dessus mis dans une boucle)
  typedef typename _ThreadIntegralType<Type>::IsIntegral IntegralType;
  Integer n = values.size();
  for(Integer i=0;i<n;++i) {
    m_reduce_infos.reduce_value = values[i];
    _collectiveBarrier();
    _computeMinMaxSum2(m_all_dispatchs,m_global_rank,min_values[i],max_values[i],sum_values[i],min_ranks[i],max_ranks[i],m_local_nb_rank,IntegralType());
    if (m_local_rank==0){
      /*pinfo() << "COMPUTE_MIN_MAX_SUM_B rank=" << m_global_rank
        << " min_rank=" << min_rank
        << " max_rank=" << max_rank
        << " min_val=" << min_val
        << " max_val=" << max_val
        << " sum_val=" << sum_val;*/
      m_mpi_dispatcher->computeMinMaxSumNoInit(min_values[i],max_values[i],sum_values[i],min_ranks[i],max_ranks[i]);
      /*pinfo() << "COMPUTE_MIN_MAX_SUM_A rank=" << m_global_rank
        << " min_rank=" << min_rank
        << " max_rank=" << max_rank;*/

      m_min_max_sum_infos.m_min_value = min_values[i];
      m_min_max_sum_infos.m_max_value = max_values[i];
      m_min_max_sum_infos.m_sum_value = sum_values[i];
      m_min_max_sum_infos.m_min_rank = min_ranks[i];
      m_min_max_sum_infos.m_max_rank = max_ranks[i];
    }
    _collectiveBarrier();
    m_min_max_sum_infos = m_all_dispatchs[0]->m_min_max_sum_infos;
    min_values[i] = m_min_max_sum_infos.m_min_value;
    max_values[i] = m_min_max_sum_infos.m_max_value;
    sum_values[i] = m_min_max_sum_infos.m_sum_value;
    min_ranks[i] = m_min_max_sum_infos.m_min_rank;
    max_ranks[i] = m_min_max_sum_infos.m_max_rank;
    _collectiveBarrier();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void HybridParallelDispatch<Type>::
broadcast(Span<Type> send_buf,Int32 rank)
{
  m_broadcast_view = send_buf;
  _collectiveBarrier();
  FullRankInfo fri = FullRankInfo::compute(MP::MessageRank(rank),m_local_nb_rank);
  int mpi_rank = fri.mpiRankValue();
  if (m_mpi_rank==mpi_rank){
    // J'ai le meme rang MPI que celui qui fait le broadcast
    if (m_global_rank==rank){
      //TODO: passage 64bit.
      m_parallel_mng->mpiParallelMng()->broadcast(send_buf.smallView(),mpi_rank);
    }
    else{
      m_all_dispatchs[m_local_rank]->m_broadcast_view.copy(m_all_dispatchs[fri.localRankValue()]->m_broadcast_view);
    }
  }
  else{
    if (m_local_rank==0){
      //TODO: passage 64bit.
      m_parallel_mng->mpiParallelMng()->broadcast(send_buf.smallView(),mpi_rank);
    }
  }
  _collectiveBarrier();
  if (m_mpi_rank!=mpi_rank){
    if (m_local_rank!=0)
      m_all_dispatchs[m_local_rank]->m_broadcast_view.copy(m_all_dispatchs[0]->m_broadcast_view);
  }
  _collectiveBarrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void HybridParallelDispatch<Type>::
allGather(Span<const Type> send_buf,Span<Type> recv_buf)
{
  //TODO: fusionner avec allGatherVariable()
  m_const_view = send_buf;
  _collectiveBarrier();
  Int64 total_size = 0;
  for( Int32 i=0; i<m_local_nb_rank; ++i ){
    total_size += m_all_dispatchs[i]->m_const_view.size();
  }
  if (m_local_rank==0){
    Int64 index = 0;
    UniqueArray<Type> local_buf(total_size);
    for( Integer i=0; i<m_local_nb_rank; ++i ){
      Span<const Type> view = m_all_dispatchs[i]->m_const_view;
      Int64 size = view.size();
      for( Int64 j=0; j<size; ++j )
        local_buf[j+index] = view[j];
      index += size;
    }
    IParallelMng* pm = m_parallel_mng->mpiParallelMng();
    //TODO: 64bit
    pm->allGather(local_buf,recv_buf.smallView());
    m_const_view = recv_buf;
  }
  _collectiveBarrier();
  if (m_local_rank!=0){
    Span<const Type> view = m_all_dispatchs[0]->m_const_view;
    recv_buf.copy(view);
  }
  _collectiveBarrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void HybridParallelDispatch<Type>::
gather(Span<const Type> send_buf,Span<Type> recv_buf,Int32 root_rank)
{
  UniqueArray<Type> tmp_buf;
  if (m_global_rank==root_rank)
    allGather(send_buf,recv_buf);
  else{
    tmp_buf.resize(send_buf.size() * m_global_nb_rank);
    allGather(send_buf,tmp_buf);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void HybridParallelDispatch<Type>::
allGatherVariable(Span<const Type> send_buf,Array<Type>& recv_buf)
{
  m_const_view = send_buf;
  _collectiveBarrier();
  Int64 total_size = 0;
  for( Integer i=0; i<m_local_nb_rank; ++i ){
    total_size += m_all_dispatchs[i]->m_const_view.size();
  }
  if (m_local_rank==0){
    Int64 index = 0;
    UniqueArray<Type> local_buf(total_size);
    for( Integer i=0; i<m_local_nb_rank; ++i ){
      Span<const Type> view = m_all_dispatchs[i]->m_const_view;
      Int64 size = view.size();
      for( Int64 j=0; j<size; ++j )
        local_buf[j+index] = view[j];
      index += size;
    }
    m_parallel_mng->mpiParallelMng()->allGatherVariable(local_buf,recv_buf);
    m_const_view = recv_buf.constView();
  }
  _collectiveBarrier();
  if (m_local_rank!=0){
    Span<const Type> view = m_all_dispatchs[0]->m_const_view;
    recv_buf.resize(view.size());
    recv_buf.copy(view);
  }
  _collectiveBarrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void HybridParallelDispatch<Type>::
gatherVariable(Span<const Type> send_buf,Array<Type>& recv_buf,Int32 root_rank)
{
  UniqueArray<Type> tmp_buf;
  if (m_global_rank==root_rank)
    allGatherVariable(send_buf,recv_buf);
  else
    allGatherVariable(send_buf,tmp_buf);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
void HybridParallelDispatch<Type>::
scatterVariable(Span<const Type> send_buf, Span<Type> recv_buf, Int32 root)
{
  m_const_view = send_buf;
  m_recv_view = recv_buf;

  _collectiveBarrier();

  // On calcule le nombre d'élément que veut tous les threads de notre processus.
  Int64 total_size = 0;
  for (Integer i = 0; i < m_local_nb_rank; ++i) {
    total_size += m_all_dispatchs[i]->m_recv_view.size();
  }

  _collectiveBarrier();

  // Les échanges MPI s'effectuent uniquement par les threads leaders des processus.
  if (m_local_rank == 0) {
    FullRankInfo fri(FullRankInfo::compute(MessageRank(root), m_local_nb_rank));

    UniqueArray<Type> local_recv_buf(total_size);

    // Si le thread "root" est dans notre processus.
    if (m_mpi_rank == fri.mpiRankValue()) {
      // Le thread leader s'occupe de l'échange.
      m_parallel_mng->mpiParallelMng()->scatterVariable(m_all_dispatchs[fri.localRankValue()]->m_const_view.smallView(),
                                                        local_recv_buf, fri.mpiRankValue());
    }
    // Les autres threads leaders mettent leurs buffers d'envoi (qu'importe ce
    // qu'ils contiennent, c'est un scatter).
    else {
      m_parallel_mng->mpiParallelMng()->scatterVariable(m_const_view.smallView(), local_recv_buf, fri.mpiRankValue());
    }

    // On a plus qu'à répartir les données reçues entre les threads.
    Integer compt = 0;
    for (Integer i = 0; i < m_local_nb_rank; ++i) {
      Int64 size = m_all_dispatchs[i]->m_recv_view.size();
      for (Integer j = 0; j < size; ++j) {
        m_all_dispatchs[i]->m_recv_view[j] = local_recv_buf[compt++];
      }
    }
  }
  _collectiveBarrier();
  recv_buf.copy(m_recv_view);
  _collectiveBarrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void HybridParallelDispatch<Type>::
allToAll(Span<const Type> send_buf,Span<Type> recv_buf,Int32 count)
{
  Int32 global_nb_rank = m_global_nb_rank;
  //TODO: Faire une version sans allocation
  Int32UniqueArray send_count(global_nb_rank,count);
  Int32UniqueArray recv_count(global_nb_rank,count);

  Int32UniqueArray send_indexes(global_nb_rank);
  Int32UniqueArray recv_indexes(global_nb_rank);
  for( Integer i=0; i<global_nb_rank; ++i ){
    send_indexes[i] = count * i;
    recv_indexes[i] = count * i;
  }
  this->allToAllVariable(send_buf,send_count,send_indexes,recv_buf,recv_count,recv_indexes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void HybridParallelDispatch<Type>::
allToAllVariable(Span<const Type> g_send_buf,
                 Int32ConstArrayView g_send_count,
                 Int32ConstArrayView g_send_index,
                 Span<Type> g_recv_buf,
                 Int32ConstArrayView g_recv_count,
                 Int32ConstArrayView g_recv_index
                 )
{
  m_alltoallv_infos.send_buf = g_send_buf;
  m_alltoallv_infos.send_count = g_send_count;
  m_alltoallv_infos.send_index = g_send_index;
  m_alltoallv_infos.recv_buf = g_recv_buf;
  m_alltoallv_infos.recv_count = g_recv_count;
  m_alltoallv_infos.recv_index = g_recv_index;

  _collectiveBarrier();

  UniqueArray<Type> tmp_recv_buf;

  // PREMIERE IMPLEMENTATION
  // Le proc de rang local 0 fait tout le travail.

  if (m_local_rank==0){

    Int32UniqueArray tmp_send_count(m_mpi_nb_rank);
    tmp_send_count.fill(0);
    Int32UniqueArray tmp_recv_count(m_mpi_nb_rank);
    tmp_recv_count.fill(0);

    Int64 total_send_size = 0;
    Int64 total_recv_size = 0;

    for( Integer i=0; i<m_local_nb_rank; ++i ){
      const AllToAllVariableInfo& vinfo = m_all_dispatchs[i]->m_alltoallv_infos;
      total_send_size += vinfo.send_buf.size();
      total_recv_size += vinfo.recv_buf.size();
    }

    UniqueArray<Type> tmp_send_buf(total_send_size);
    tmp_recv_buf.resize(total_recv_size);

    // Calcule le nombre d'éléments à envoyer et recevoir pour chaque proc.
    for( Integer i=0; i<m_local_nb_rank; ++i ){
      const AllToAllVariableInfo& vinfo = m_all_dispatchs[i]->m_alltoallv_infos;

      for( Integer z=0; z<m_global_nb_rank; ++z ){

        FullRankInfo fri(FullRankInfo::compute(MP::MessageRank(z),m_local_nb_rank));
        Int32 fri_mpi_rank = fri.mpiRankValue();

        Int32 nb_send = vinfo.send_count[z];

        tmp_send_count[fri_mpi_rank] += nb_send;
        tmp_recv_count[fri_mpi_rank] += vinfo.recv_count[z];

#if 0
        info() << "my_local=" << i << " dest=" << z
               << " send_count=" << vinfo.send_count[z] << " send_index=" << vinfo.send_index[z]
               << " recv_count=" << vinfo.recv_count[z] << " recv_index=" << vinfo.recv_index[z];
        {
          Integer vindex = vinfo.send_index[z];
          for( Integer w=0, wn=vinfo.send_count[z]; w<wn; ++w ){
            info() << "V=" << vinfo.send_buf[ vindex + w ];
          }
        }
#endif
      }
    }

    Int32UniqueArray tmp_send_index(m_mpi_nb_rank);
    Int32UniqueArray tmp_recv_index(m_mpi_nb_rank);
    tmp_send_index[0] = 0;
    tmp_recv_index[0] = 0;
    for( Integer k=1, nmpi=m_mpi_nb_rank; k<nmpi; ++k ){
      tmp_send_index[k] = tmp_send_index[k-1] + tmp_send_count[k-1];
      tmp_recv_index[k] = tmp_recv_index[k-1] + tmp_recv_count[k-1];
    }

    for( Integer i=0; i<m_local_nb_rank; ++i ){
      const AllToAllVariableInfo& vinfo = m_all_dispatchs[i]->m_alltoallv_infos;

      for( Integer z=0; z<m_global_nb_rank; ++ z){

        FullRankInfo fri(FullRankInfo::compute(MP::MessageRank(z),m_local_nb_rank));
        Int32 fri_mpi_rank = fri.mpiRankValue();

        Integer nb_send = vinfo.send_count[z];
        {

          Integer tmp_current_index = tmp_send_index[fri_mpi_rank];
          Integer local_current_index = vinfo.send_index[z];
          for( Integer j=0; j<nb_send; ++j )
            tmp_send_buf[j+tmp_current_index] = vinfo.send_buf[j+local_current_index];
          tmp_send_index[fri_mpi_rank] += nb_send;
        }


      }
    }

    tmp_send_index[0] = 0;
    tmp_recv_index[0] = 0;
    for( Integer k=1, nmpi=m_mpi_nb_rank; k<nmpi; ++k ){
      tmp_send_index[k] = tmp_send_index[k-1] + tmp_send_count[k-1];
      tmp_recv_index[k] = tmp_recv_index[k-1] + tmp_recv_count[k-1];
    }



    /*    Integer send_index = 0;
    for( Integer i=0; i<m_local_nb_rank; ++i ){
      ConstArrayView<Type> send_view = m_all_dispatchs[i]->m_alltoallv_infos.send_buf;
      Integer send_size = send_view.size();
      info() << "ADD_TMP_SEND_BUF send_index=" << send_index << " size=" << send_size;
      for( Integer j=0; j<send_size; ++j )
        tmp_send_buf[j+send_index] = send_view[j];
      send_index += send_size;
    }
    */

#if 0
    info() << "AllToAllV nb_send=" << total_send_size << " nb_recv=" << total_recv_size;
    for( Integer k=0; k<m_mpi_nb_rank; ++k ){
      info() << "INFOS Rank=" << k << " send_count=" << tmp_send_count[k] << " recv_count=" << tmp_recv_count[k]
             << " send_index=" << tmp_send_index[k] << " recv_index=" << tmp_recv_index[k];
    }

    for( Integer i=0; i<tmp_send_buf.size(); ++i )
      info() << "SEND_BUF[" << i << "] = " << tmp_send_buf[i];

    for( Integer k=0; k<m_mpi_nb_rank; ++k ){
      info() << "SEND Rank=" << k << " send_count=" << tmp_send_count[k] << " recv_count=" << tmp_recv_count[k]
             << " send_index=" << tmp_send_index[k] << " recv_index=" << tmp_recv_index[k];
      Integer vindex = tmp_send_index[k];
      for( Integer w=0, wn=tmp_send_count[k]; w<wn; ++w ){
        info() << "V=" << tmp_send_buf[ vindex + w ];
      }
    }
#endif

    m_parallel_mng->mpiParallelMng()->allToAllVariable(tmp_send_buf,tmp_send_count,
                                                       tmp_send_index,tmp_recv_buf,
                                                       tmp_recv_count,tmp_recv_index);

#if 0
    for( Integer i=0; i<tmp_recv_buf.size(); ++i )
      info() << "RECV_BUF[" << i << "] = " << tmp_recv_buf[i];

    for( Integer k=0; k<m_mpi_nb_rank; ++k ){
      info() << "RECV Rank=" << k << " send_count=" << tmp_send_count[k] << " recv_count=" << tmp_recv_count[k]
             << " send_index=" << tmp_send_index[k] << " recv_index=" << tmp_recv_index[k];
      Integer vindex = tmp_recv_index[k];
      for( Integer w=0, wn=tmp_recv_count[k]; w<wn; ++w ){
        info() << "V=" << tmp_recv_buf[ vindex + w ];
      }
    }
#endif

    m_const_view = tmp_recv_buf.constView();


    for( Integer z=0; z<m_global_nb_rank; ++ z){
      FullRankInfo fri(FullRankInfo::compute(MP::MessageRank(z),m_local_nb_rank));
      Int32 fri_mpi_rank = fri.mpiRankValue();

      for( Integer i=0; i<m_local_nb_rank; ++i ){
        AllToAllVariableInfo& vinfo = m_all_dispatchs[i]->m_alltoallv_infos;
        Span<Type> my_buf = vinfo.recv_buf;
        Int64 recv_size = vinfo.recv_count[z];
        Int64 recv_index = tmp_recv_index[fri_mpi_rank];
        
        Span<const Type> recv_view = tmp_recv_buf.span().subSpan(recv_index,recv_size);

        Int64 my_recv_index = vinfo.recv_index[z];

        //info() << "GET i=" << i << " z=" << z << " size=" << recv_size << " index=" << recv_index
        //       << " mpi_rank=" << fri_mpi_rank << " my_index=" << my_recv_index;

        tmp_recv_index[fri_mpi_rank] = CheckedConvert::toInt32(tmp_recv_index[fri_mpi_rank] + recv_size);

        for( Int64 j=0; j<recv_size; ++j )
          my_buf[j+my_recv_index] = recv_view[j];

        //for( Integer j=0; j<recv_size; ++j )
        //info() << "V=" << recv_view[j];

        my_recv_index += recv_size;
      }
    }

  }
  _collectiveBarrier();

  //info() << "END_PHASE_1_ALL_TO_ALL_V my_rank=" << m_global_rank << " (local=" << m_local_rank << ")";

  //_collectiveBarrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> auto HybridParallelDispatch<Type>::
send(Span<const Type> send_buffer,Int32 rank,bool is_blocked) -> Request
{
  eBlockingType block_mode = (is_blocked) ? MP::Blocking : MP::NonBlocking;
  PointToPointMessageInfo p2p_message(MessageRank(rank),block_mode);
  return send(send_buffer,p2p_message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void HybridParallelDispatch<Type>::
send(ConstArrayView<Type> send_buf,Int32 rank)
{
  send(send_buf,rank,true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Parallel::Request HybridParallelDispatch<Type>::
receive(Span<Type> recv_buffer,Int32 rank,bool is_blocked)
{
  eBlockingType block_mode = (is_blocked) ? MP::Blocking : MP::NonBlocking;
  PointToPointMessageInfo p2p_message(MessageRank(rank),block_mode);
  return receive(recv_buffer,p2p_message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Request HybridParallelDispatch<Type>::
send(Span<const Type> send_buffer,const PointToPointMessageInfo& message2)
{
  PointToPointMessageInfo message(message2);
  bool is_blocking = message.isBlocking();
  message.setEmiterRank(MessageRank(m_global_rank));
  Request r = m_message_queue->addSend(message, ConstMemoryView(send_buffer));
  if (is_blocking){
    m_message_queue->waitAll(ArrayView<MP::Request>(1,&r));
    return Request();
  }
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Request HybridParallelDispatch<Type>::
receive(Span<Type> recv_buffer,const PointToPointMessageInfo& message2)
{
  PointToPointMessageInfo message(message2);
  message.setEmiterRank(MessageRank(m_global_rank));
  bool is_blocking = message.isBlocking();
  Request r = m_message_queue->addReceive(message,ReceiveBufferInfo(MutableMemoryView(recv_buffer)));
  if (is_blocking){
    m_message_queue->waitAll(ArrayView<Request>(1,&r));
    return Request();
  }
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void HybridParallelDispatch<Type>::
recv(ArrayView<Type> recv_buffer,Integer rank)
{
  recv(recv_buffer,rank,true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void HybridParallelDispatch<Type>::
sendRecv(ConstArrayView<Type> send_buffer,ArrayView<Type> recv_buffer,Integer proc)
{
  ARCANE_UNUSED(send_buffer);
  ARCANE_UNUSED(recv_buffer);
  ARCANE_UNUSED(proc);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Type HybridParallelDispatch<Type>::
allReduce(eReduceType op,Type send_buf)
{
  m_reduce_infos.reduce_value = send_buf;
  //pinfo() << "ALL REDUCE BEGIN RANK=" << m_global_rank << " TYPE=" << (int)op << " MY=" << send_buf;
  cout.flush();
  _collectiveBarrier();
  if (m_local_rank==0){
    Type ret = m_all_dispatchs[0]->m_reduce_infos.reduce_value;
    switch(op){
    case Parallel::ReduceMin:
      for( Integer i=1; i<m_local_nb_rank; ++i )
        ret = math::min(ret,m_all_dispatchs[i]->m_reduce_infos.reduce_value);
      break;
    case Parallel::ReduceMax:
      for( Integer i=1; i<m_local_nb_rank; ++i )
        ret = math::max(ret,m_all_dispatchs[i]->m_reduce_infos.reduce_value);
      break;
    case Parallel::ReduceSum:
      for( Integer i=1; i<m_local_nb_rank; ++i )
        ret = (Type)(ret + m_all_dispatchs[i]->m_reduce_infos.reduce_value);
      break;
    default:
      ARCANE_FATAL("Bad reduce type");
    }
    ret = m_parallel_mng->mpiParallelMng()->reduce(op,ret);
    m_all_dispatchs[0]->m_reduce_infos.reduce_value = ret;
    //pinfo() << "ALL REDUCE RANK=" << m_local_rank << " TYPE=" << (int)op << " MY=" << send_buf << " GLOBAL=" << ret << '\n';
  }
  _collectiveBarrier();
  Type ret = m_all_dispatchs[0]->m_reduce_infos.reduce_value;
  _collectiveBarrier();
  return ret;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void HybridParallelDispatch<Type>::
allReduce(eReduceType op,Span<Type> send_buf)
{
  //TODO: fusionner avec allReduce simple
  m_reduce_infos.reduce_buf = send_buf;
  ++m_reduce_infos.m_index;
  Int64 buf_size = send_buf.size();
  UniqueArray<Type> ret(buf_size);
  //cout << "ALL REDUCE BEGIN RANk=" << m_local_rank << " TYPE=" << (int)op << " MY=" << send_buf << '\n';
  //cout.flush();
  _collectiveBarrier();
  {
    Integer index0 = m_all_dispatchs[0]->m_reduce_infos.m_index;
    for( Integer i=0; i<m_local_nb_rank; ++i ){
      Integer indexi = m_all_dispatchs[i]->m_reduce_infos.m_index;
      if (index0!=m_all_dispatchs[i]->m_reduce_infos.m_index){
        ARCANE_FATAL("INTERNAL: incoherent all reduce i0={0} in={1} n={2}",
                     index0,indexi,i);
      }
    }
  }

  if (m_local_rank==0){
    for( Integer j=0; j<buf_size; ++j )
      ret[j] = m_all_dispatchs[0]->m_reduce_infos.reduce_buf[j];
    switch(op){
    case Parallel::ReduceMin:
      for( Integer i=1; i<m_local_nb_rank; ++i )
        for( Integer j=0; j<buf_size; ++j )
          ret[j] = math::min(ret[j],m_all_dispatchs[i]->m_reduce_infos.reduce_buf[j]);
      break;
    case Parallel::ReduceMax:
      for( Integer i=1; i<m_local_nb_rank; ++i )
        for( Integer j=0; j<buf_size; ++j )
          ret[j] = math::max(ret[j],m_all_dispatchs[i]->m_reduce_infos.reduce_buf[j]);
      break;
    case Parallel::ReduceSum:
      for( Integer i=1; i<m_local_nb_rank; ++i )
        for( Integer j=0; j<buf_size; ++j )
          ret[j] = (Type)(ret[j] + m_all_dispatchs[i]->m_reduce_infos.reduce_buf[j]);
      break;
    default:
      ARCANE_FATAL("Bad reduce type");
    }
    m_parallel_mng->mpiParallelMng()->reduce(op,ret);
    send_buf.copy(ret);
  }

  _collectiveBarrier();

  if (m_local_rank!=0){
    Span<const Type> global_buf = m_all_dispatchs[0]->m_reduce_infos.reduce_buf;
    send_buf.copy(global_buf);
  }

  _collectiveBarrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Request HybridParallelDispatch<Type>::
nonBlockingAllReduce(eReduceType op,Span<const Type> send_buf,Span<Type> recv_buf)
{
  ARCANE_UNUSED(op);
  ARCANE_UNUSED(send_buf);
  ARCANE_UNUSED(recv_buf);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
template<class Type> Request HybridParallelDispatch<Type>::
nonBlockingAllGather(Span<const Type> send_buf, Span<Type> recv_buf)
{
  ARCANE_UNUSED(send_buf);
  ARCANE_UNUSED(recv_buf);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Request HybridParallelDispatch<Type>::
nonBlockingBroadcast(Span<Type> send_buf, Int32 rank)
{
  ARCANE_UNUSED(send_buf);
  ARCANE_UNUSED(rank);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Request HybridParallelDispatch<Type>::
nonBlockingGather(Span<const Type> send_buf, Span<Type> recv_buf, Int32 rank)
{
  ARCANE_UNUSED(send_buf);
  ARCANE_UNUSED(recv_buf);
  ARCANE_UNUSED(rank);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Request HybridParallelDispatch<Type>::
nonBlockingAllToAll(Span<const Type> send_buf, Span<Type> recv_buf, Int32 count)
{
  ARCANE_UNUSED(send_buf);
  ARCANE_UNUSED(recv_buf);
  ARCANE_UNUSED(count);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Request HybridParallelDispatch<Type>::
nonBlockingAllToAllVariable(Span<const Type> send_buf, ConstArrayView<Int32> send_count,
                            ConstArrayView<Int32> send_index, Span<Type> recv_buf,
                            ConstArrayView<Int32> recv_count, ConstArrayView<Int32> recv_index)
{
  ARCANE_UNUSED(send_buf);
  ARCANE_UNUSED(recv_buf);
  ARCANE_UNUSED(send_count);
  ARCANE_UNUSED(recv_count);
  ARCANE_UNUSED(send_index);
  ARCANE_UNUSED(recv_index);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Type HybridParallelDispatch<Type>::
scan(eReduceType op,Type send_buf)
{
  ARCANE_UNUSED(op);
  ARCANE_UNUSED(send_buf);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void HybridParallelDispatch<Type>::
scan(eReduceType op,ArrayView<Type> send_buf)
{
  ARCANE_UNUSED(op);
  ARCANE_UNUSED(send_buf);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Request HybridParallelDispatch<Type>::
gather(Arccore::MessagePassing::GatherMessageInfo<Type>&)
{
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void HybridParallelDispatch<Type>::
_collectiveBarrier()
{
  m_parallel_mng->getThreadBarrier()->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class HybridParallelDispatch<char>;
template class HybridParallelDispatch<signed char>;
template class HybridParallelDispatch<unsigned char>;
template class HybridParallelDispatch<short>;
template class HybridParallelDispatch<unsigned short>;
template class HybridParallelDispatch<int>;
template class HybridParallelDispatch<unsigned int>;
template class HybridParallelDispatch<long>;
template class HybridParallelDispatch<unsigned long>;
template class HybridParallelDispatch<long long>;
template class HybridParallelDispatch<unsigned long long>;
template class HybridParallelDispatch<float>;
template class HybridParallelDispatch<double>;
template class HybridParallelDispatch<long double>;
template class HybridParallelDispatch<Real2>;
template class HybridParallelDispatch<Real3>;
template class HybridParallelDispatch<Real2x2>;
template class HybridParallelDispatch<Real3x3>;
template class HybridParallelDispatch<HPReal>;
template class HybridParallelDispatch<APReal>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
