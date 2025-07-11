// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiNodeWindowBase.h                                            (C) 2000-2025 */
/*                                                                           */
/* TODO.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPINODEWINDOW_H
#define ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPINODEWINDOW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/INodeWindowBase.h"
#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

#include <cstring>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiNodeWindowBase
: public INodeWindowBase
{
 public:

  explicit MpiNodeWindowBase(Integer nb_elem_local_section, Integer sizeof_type, const MPI_Comm& comm, Int32 my_node_rank)
  : m_nb_elem_local(nb_elem_local_section)
  , m_sizeof_type(sizeof_type)
  , m_win()
  , m_comm(comm)
  , m_my_rank(my_node_rank)
  {
    MPI_Info win_info;
    MPI_Info_create(&win_info);
    void* ptr_win = nullptr;

    MPI_Info_set(win_info, "alloc_shared_noncontig", "false");

    MPI_Comm_rank(m_comm, &m_my_rank);

    int error = MPI_Win_allocate_shared(m_nb_elem_local * sizeof_type, sizeof_type, win_info, m_comm, &ptr_win, &m_win);

    MPI_Info_free(&win_info);
  }

  ~MpiNodeWindowBase() override
  {
    MPI_Win_free(&m_win);
  }

 public:

  Integer sizeofOneElem() const override
  {
    return m_sizeof_type;
  }

  Integer sizeLocalSegment() const override
  {
    return sizeOtherRankSegment(m_my_rank);
  }

  Integer sizeOtherRankSegment(int rank) const override
  {
    MPI_Aint size_win;
    int size_type;
    void* ptr_win = nullptr;

    int error = MPI_Win_shared_query(m_win, rank, &size_win, &size_type, &ptr_win);

    return (size_win / size_type);
  }

  void* data() override
  {
    return dataOtherRank(m_my_rank);
  }

  void* dataOtherRank(int rank) override
  {
    MPI_Aint size_win;
    int size_type;
    void* ptr_win = nullptr;

    int error = MPI_Win_shared_query(m_win, rank, &size_win, &size_type, &ptr_win);
    return ptr_win;
  }

 private:

  Integer m_nb_elem_local;
  MPI_Win m_win;
  MPI_Comm m_comm;
  Int32 m_my_rank;
  Integer m_sizeof_type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
