// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GraphDistributor.h                                          (C) 2000-2024 */
/*                                                                           */
/* This file provides declaration and definition of a class used to          */
/* redistribute the graph accross another set of processors.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_GRAPHDISTRIBUTOR_H
#define ARCANE_STD_GRAPHDISTRIBUTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/CheckedConvert.h"

#include "arcane/IParallelTopology.h"
#include "arcane/ParallelMngUtils.h"

#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Redistribute graph data to another "communicator"
 *
 * \abstract By redistributing graph data, we can use smaller communicators to
 *           compute partitioning, which is better for efficiency.
 *
 * La classe doit être initialisée en appelant soit initWithOneRankPerNode(),
 * soit initWithMaxRank().
 */
class GraphDistributor
{
 public:

  GraphDistributor(IParallelMng* pm)
  : m_pm_ini(pm)
  , m_targetSize(-1)
  , m_targetRank(-1)
  , m_skip(true)
  , m_contribute(false)
  {
  }

  //! Automatic distribution : one partitioning process per node
  void initWithOneRankPerNode(bool allow_only_one_rank)
  {
    m_is_init = true;
    m_targetRank = -1;
    m_skip = true;
    m_contribute = false;
    if (!m_pm_ini)
      return;

    auto topo{ ParallelMngUtils::createTopologyRef(m_pm_ini) };
    if (topo->isMasterMachine()) {
      m_contribute = true;
    }

    Int32 machineRank = topo->machineRank();
    Int32ConstArrayView targetRanks = topo->masterMachineRanks();

    m_targetRank = targetRanks[machineRank];
    m_targetSize = targetRanks.size();

    if ((m_targetSize != m_pm_ini->commSize()) // Only useful if number of processes change.
        && (allow_only_one_rank || m_targetSize > 1)) { // And if more than 1 node for parmetis
      m_skip = false;
      m_pm_sub = m_pm_ini->createSubParallelMngRef(targetRanks);
    }
    else { // All ranks have to work.
      // Still making a new communicator for safety when using with third party library
      m_contribute = true;
      m_targetRank = m_pm_ini->commRank();
      m_targetSize = m_pm_ini->commSize();
      Int32UniqueArray keptRanks(m_targetSize);
      for (int i = 0; i < m_targetSize; i++)
        keptRanks[i] = i;
      m_pm_sub = m_pm_ini->createSubParallelMngRef(keptRanks);
    }

    m_pm_ini->traceMng()->info() << "Running on " << m_targetSize << " nodes";
  }

  // Max is hard-coded to work on integers.
  void initWithMaxRank(Int32 targetSize)
  {
    m_is_init = true;
    m_targetSize = targetSize;
    m_targetRank = -1;
    m_skip = true;
    m_contribute = false;

    if (!m_pm_ini)
      return;

    if (m_pm_ini->commRank() < targetSize) { // At this time, no duplication
      m_contribute = 1;
    }
    Int64 my_rank = m_pm_ini->commRank();
    Int64 x = my_rank * m_targetSize;
    m_targetRank = CheckedConvert::toInt32(x / m_pm_ini->commSize());

    Int32UniqueArray keepProc(m_targetSize);
    Int32 step = m_targetSize / m_pm_ini->commSize();
    step = (step == 0) ? 1 : step;
    for (int i = 0; i < m_targetSize; ++i) {
      keepProc[i] = i * step;
    }

    m_pm_sub = m_pm_ini->createSubParallelMngRef(keepProc);

    if (m_targetSize != m_pm_ini->commSize()) {
      m_skip = false;
    }
  }

 public:

  Int32 size() const { return m_targetSize; }

  bool contribute() const { return m_contribute; }

  //< Do the redistribution pm -> newComm
  template <typename DataT>
  SharedArray<DataT> convert(ConstArrayView<DataT> in, Array<DataT>* pattern = nullptr,
                             bool is_indirection = false) const
  {
    if (!m_is_init)
      ARCANE_FATAL("Missing initialisation");
    if (m_skip) {
      SharedArray<DataT> out(in);
      if (pattern != NULL) {
        Integer size = in.size();
        if (is_indirection)
          size -= 1;
        pattern->resize(size, m_targetRank);
      }
      return out;
    }
    ConstArrayView<DataT> toSnd;

    Int32 nInfos = 2;
    if (is_indirection) {
      toSnd = in.subView(0, in.size() - 1);
      nInfos += 1; // need to store end of array
    }
    else {
      toSnd = in;
    }

    Int32 commSize = m_pm_ini->commSize();
    UniqueArray<Int32> sndCnt(nInfos * commSize, -1);
    UniqueArray<Parallel::Request> req;
    UniqueArray<Int32> n_wanted(nInfos);
    n_wanted[0] = m_targetRank;
    n_wanted[1] = toSnd.size();
    if (is_indirection)
      n_wanted[2] = static_cast<Int32>(in[in.size() - 1]);

    m_pm_ini->allGather(n_wanted, sndCnt);

    UniqueArray<Int32> sndNbr(commSize, 0);
    UniqueArray<Int32> rcvNbr(commSize, 0);
    UniqueArray<Int32> sndDsp(commSize, 0);
    UniqueArray<Int32> rcvDsp(commSize, 0);

    sndNbr[m_targetRank] = toSnd.size();

    if (pattern != NULL) {
      pattern->resize(0);
    }

    Int32 myRank = m_pm_ini->commRank();
    Int32 begin = 0;
    for (int i = 0; i < commSize; ++i) {
      if (sndCnt[nInfos * i] == myRank) { // We have to receive this message
        rcvNbr[i] = sndCnt[nInfos * i + 1];
        rcvDsp[i] = begin;
        begin += rcvNbr[i];

        if (pattern != NULL)
          pattern->addRange(i, rcvNbr[i]);
      }
    }
    if (contribute() && is_indirection)
      begin += 1; // Trick: add one to mark end of array
    SharedArray<DataT> out(begin, -1);

    m_pm_ini->allToAllVariable(toSnd, sndNbr, sndDsp, out, rcvNbr, rcvDsp);

    if (contribute() && is_indirection) { // We have to update offsets
      DataT offset = 0;
      DataT* my_iter = out.data();
      for (int i = 0; i < commSize; ++i) {
        if (sndCnt[nInfos * i] == myRank) { // We have to receive this message
          Int32 nRecv = sndCnt[nInfos * i + 1];
          DataT* my_end(my_iter + nRecv);
          for (; my_iter != my_end; ++my_iter)
            (*my_iter) += offset;
          offset += sndCnt[nInfos * i + 2];
        }
      }
      out[out.size() - 1] = offset;
    }

    return out;
  }

  //< Do the backward redistribution newComm -> pm
  template <typename DataT>
  SharedArray<DataT> convertBack(ConstArrayView<DataT> in, Int32 nRecv) const
  {
    if (!m_is_init)
      ARCANE_FATAL("Missing initialisation");
    if (m_skip) {
      SharedArray<DataT> out(in);
      return out;
    }

    Int32 nInfos = 2;
    Int32 commSize = m_pm_ini->commSize();
    UniqueArray<Int32> sndCnt(nInfos * commSize, -1);
    UniqueArray<Parallel::Request> req;
    UniqueArray<Int32> n_wanted(nInfos);
    n_wanted[0] = m_targetRank;
    n_wanted[1] = nRecv;

    m_pm_ini->allGather(n_wanted, sndCnt);

    UniqueArray<Int32> sndNbr(commSize, 0);
    UniqueArray<Int32> rcvNbr(commSize, 0);
    UniqueArray<Int32> sndDsp(commSize, 0);
    UniqueArray<Int32> rcvDsp(commSize, 0);

    rcvNbr[m_targetRank] = nRecv;

    Int32 myRank = m_pm_ini->commRank();
    Int32 begin = 0;
    for (int i = 0; i < commSize; ++i) {
      if (sndCnt[nInfos * i] == myRank) { // We have to receive this message
        sndNbr[i] = sndCnt[nInfos * i + 1];
        sndDsp[i] = begin;
        begin += sndNbr[i];
      }
    }
    SharedArray<DataT> out(nRecv, -1);

    m_pm_ini->allToAllVariable(in, sndNbr, sndDsp, out, rcvNbr, rcvDsp);

    return out;
  }

  IParallelMng* subParallelMng() const
  {
    IParallelMng* pm = m_pm_sub.get();
    if (pm)
      return pm;
    return m_pm_ini->sequentialParallelMng();
  }

 public:

  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane. Use subParallelMng()->communicator() instead")
  MPI_Comm getCommunicator() const
  {
    if (!m_pm_sub)
      return MPI_COMM_NULL;
    Parallel::Communicator comm = m_pm_sub->communicator();
    return (MPI_Comm)comm;
  }

  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane. Use subParallelMng() instead")
  IParallelMng* parallelManager() const
  {
    return m_pm_sub.get();
  }

 private:

  IParallelMng* m_pm_ini = nullptr;
  Ref<IParallelMng> m_pm_sub;
  Int32 m_targetSize = -1; // Taille du sous-communicateur
  Int32 m_targetRank = -1; // Rang dans le sous-communicateur
  bool m_skip = false; // Pas de redistribution
  bool m_contribute = false;
  bool m_is_init = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#endif
