// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiNodeWindow.h                                            (C) 2000-2025 */
/*                                                                           */
/* TODO.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPINODEWINDOW_H
#define ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPINODEWINDOW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/INodeWindow.h"
#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

#include <cstring>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

template<class Type>
class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiNodeWindow
: public INodeWindow<Type>
{
 public:

  explicit MpiNodeWindow(Integer nb_elem_local_section, const MPI_Comm& comm, Int32 my_node_rank)
  : m_nb_elem_local(nb_elem_local_section)
  , m_comm(comm)
  , m_my_rank(my_node_rank)
  {
    MPI_Info win_info;
    MPI_Info_create(&win_info);
    void* ptr_win = nullptr;

    MPI_Info_set(win_info, "alloc_shared_noncontig", "false");

    MPI_Comm_rank(m_comm,&m_my_rank);

    int error = MPI_Win_allocate_shared(m_nb_elem_local * sizeof(Type), sizeof(Type), win_info, m_comm, &ptr_win, &m_win);

    MPI_Info_free(&win_info);
  }

  ~MpiNodeWindow() override
  {
    MPI_Win_free(&m_win);
  }

 public:

  Int64 sizeLocalSegment() const override
  {
    return sizeOtherRankSegment(m_my_rank);
  }

  Int64 sizeOtherRankSegment(int rank) const override
  {
    MPI_Aint size_win;
    int size_type;
    Type* ptr_win = nullptr;

    int error = MPI_Win_shared_query(m_win, rank, &size_win, &size_type, &ptr_win);

    return (size_win / size_type);
  }


  ArrayView<Type> localSegmentView() override
  {
    return otherRankSegmentView(m_my_rank);
  }

  ArrayView<Type> otherRankSegmentView(int rank) override
  {
    MPI_Aint size_win;
    int size_type;
    Type* ptr_win = nullptr;

    int error = MPI_Win_shared_query(m_win, rank, &size_win, &size_type, &ptr_win);

    Integer nb_elem = static_cast<Integer>(size_win / size_type);
    return ArrayView<Type>(nb_elem, ptr_win);
  }

  ConstArrayView<Type> localSegmentConstView() const override
  {
    return otherRankSegmentConstView(m_my_rank);
  }

  ConstArrayView<Type> otherRankSegmentConstView(int rank) const override
  {
    MPI_Aint size_win;
    int size_type;
    Type* ptr_win = nullptr;

    int error = MPI_Win_shared_query(m_win, rank, &size_win, &size_type, &ptr_win);

    Integer nb_elem = static_cast<Integer>(size_win / size_type);
    return ConstArrayView<Type>(nb_elem, ptr_win);
  }

  Type* data() override
  {
    return dataOtherRank(m_my_rank);
  }

  Type* dataOtherRank(int rank) override
  {
    MPI_Aint size_win;
    int size_type;
    Type* ptr_win = nullptr;

    int error = MPI_Win_shared_query(m_win, rank, &size_win, &size_type, &ptr_win);
    return ptr_win;
  }

 private:

  Integer m_nb_elem_local;
  MPI_Win m_win;
  MPI_Comm m_comm;
  Int32 m_my_rank;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

