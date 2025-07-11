// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiAiONodeWindowBase.h                                            (C) 2000-2025 */
/*                                                                           */
/* TODO.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIAIONODEWINDOW_H
#define ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIAIONODEWINDOW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/INodeWindowBase.h"
#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

#include <cstring>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiAiONodeWindowBase
: public INodeWindowBase
{
 public:

  explicit MpiAiONodeWindowBase(void* node_window, MPI_Aint offset, const MPI_Comm& comm, Int32 my_node_rank)
  : m_node_window(node_window)
  , m_nb_elem_local(0)
  , m_offset(offset)
  , m_comm(comm)
  , m_my_rank(my_node_rank)
  , m_size_type(0)
  {
    m_win = reinterpret_cast<MPI_Win*>(static_cast<char*>(m_node_window) - offset);
    void* ptr_win = nullptr;
    int error = MPI_Win_shared_query(*m_win, m_my_rank, &m_nb_elem_local, &m_size_type, &ptr_win);
  }

  ~MpiAiONodeWindowBase() override = default;

 public:

  Integer sizeofOneElem() const override
  {
    return m_size_type;
  }

  Integer sizeLocalSegment() const override
  {
    return static_cast<Integer>(m_nb_elem_local);
  }

  Integer sizeOtherRankSegment(int rank) const override
  {
    MPI_Aint size_win;
    int size_type;
    void* ptr_win = nullptr;

    int error = MPI_Win_shared_query(*m_win, rank, &size_win, &size_type, &ptr_win);

    return static_cast<Integer>((size_win - m_offset) / size_type);
  }

  void* data() override
  {
    return m_node_window;
  }

  void* dataOtherRank(int rank) override
  {
    MPI_Aint size_win;
    int size_type;
    void* ptr_win = nullptr;

    int error = MPI_Win_shared_query(*m_win, rank, &size_win, &size_type, &ptr_win);

    return (static_cast<char*>(ptr_win) + m_offset);
  }

 private:

  void* m_node_window;
  MPI_Win* m_win;
  MPI_Aint m_nb_elem_local;
  MPI_Aint m_offset;
  MPI_Comm m_comm;
  Int32 m_my_rank;
  Integer m_size_type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
