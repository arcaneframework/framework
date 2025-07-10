// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiAiONodeWindow.h                                            (C) 2000-2025 */
/*                                                                           */
/* TODO.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIAIONODEWINDOW_H
#define ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIAIONODEWINDOW_H
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
class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiAiONodeWindow
: public INodeWindow<Type>
{
 public:

  explicit MpiAiONodeWindow(Type* node_window, MPI_Aint offset, const MPI_Comm& comm, Int32 my_node_rank)
  : m_node_window(node_window)
  , m_offset(offset)
  , m_comm(comm)
  , m_my_rank(my_node_rank)
  {}

  ~MpiAiONodeWindow() override = default;

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

    int error = MPI_Win_shared_query(*_win(), rank, &size_win, &size_type, &ptr_win);

    return ((size_win - m_offset) / sizeof(Type));
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

    int error = MPI_Win_shared_query(*_win(), rank, &size_win, &size_type, &ptr_win);

    Integer nb_elem = static_cast<Integer>((size_win - m_offset) / sizeof(Type));
    Type* data = reinterpret_cast<Type*>(reinterpret_cast<char*>(ptr_win) + m_offset);

    return ArrayView<Type>(nb_elem, data);
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

    int error = MPI_Win_shared_query(*_win(), rank, &size_win, &size_type, &ptr_win);

    Integer nb_elem = static_cast<Integer>((size_win - m_offset) / sizeof(Type));
    Type* data = reinterpret_cast<Type*>(reinterpret_cast<char*>(ptr_win) + m_offset);

    return ConstArrayView<Type>(nb_elem, data);
  }

  Type* data() override
  {
    return m_node_window;
  }

  Type* dataOtherRank(int rank) override
  {
    MPI_Aint size_win;
    int size_type;
    Type* ptr_win = nullptr;

    int error = MPI_Win_shared_query(*_win(), rank, &size_win, &size_type, &ptr_win);

    return reinterpret_cast<Type*>(reinterpret_cast<char*>(ptr_win) + m_offset);
  }

 private:

  MPI_Win* _win() const
  {
    return reinterpret_cast<MPI_Win*>(reinterpret_cast<char*>(m_node_window) - m_offset);
  }

 private:

  Type* m_node_window;
  MPI_Aint m_offset;
  MPI_Comm m_comm;
  Int32 m_my_rank;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

