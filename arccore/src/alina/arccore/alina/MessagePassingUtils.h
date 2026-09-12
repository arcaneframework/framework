// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessagePassingUtils.h                                       (C) 2000-2026 */
/*                                                                           */
/* Various utilities to handle message passing.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_MESSAGEPASSINGUTILS_H
#define ARCCORE_ALINA_MESSAGEPASSINGUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * This file is based on the work on AMGCL library (version march 2026)
 * which can be found at https://github.com/ddemidov/amgcl.
 *
 * Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>
 * SPDX-License-Identifier: MIT
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/FixedArray.h"
#include "arccore/common/Array.h"

#include "arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h"
#include "arccore/message_passing/Messages.h"
#include "arccore/message_passing/PointToPointMessageInfo.h"

#include "arccore/alina/ValueTypeInterface.h"
#include "arccore/alina/AlinaUtils.h"

#include <vector>
#include <numeric>
#include <complex>
#include <type_traits>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Convenience wrapper around MPI_Init/MPI_Finalize.
struct mpi_init
{
  mpi_init(int* argc, char*** argv)
  {
    MPI_Init(argc, argv);
  }

  ~mpi_init()
  {
    MPI_Finalize();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Convenience wrapper around MPI_Init_threads/MPI_Finalize.
struct mpi_init_thread
{
  mpi_init_thread(int* argc, char*** argv)
  {
    int _;
    MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &_);
  }

  ~mpi_init_thread()
  {
    MPI_Finalize();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Convenience wrapper around MPI_Comm.
 */
struct ARCCORE_ALINA_EXPORT AlinaCommunicator
{
 private:

  MPI_Comm m_mpi_communicator = MPI_COMM_NULL;

 public:

  int rank = 0;
  int size = 0;
  Ref<IMessagePassingMng> m_message_passing_mng;

  AlinaCommunicator() = default;

  explicit AlinaCommunicator(MPI_Comm comm);

  explicit AlinaCommunicator(IMessagePassingMng* mpm_comm);

  MPI_Comm mpiCommunicator() const { return m_mpi_communicator; }
  IMessagePassingMng* messagePassingMng() const { return m_message_passing_mng.get(); }

  /// Exclusive sum over mpi communicator
  template <typename T>
  UniqueArray<T> exclusive_sum(T n) const
  {
    // TODO: Use scan.
    UniqueArray<T> v(size + 1);
    v[0] = 0;
    T v0 = n;
    ConstArrayView<T> v0_view(1, &v0);
    ArrayView<T> out_view(static_cast<Int32>(v.size()), &v[1]);
    mpAllGather(m_message_passing_mng.get(), v0_view, out_view);
    std::partial_sum(v.begin(), v.end(), v.begin());
    return v;
  }

  std::complex<long double> reduceSum(const std::complex<long double>& lval) const
  {
    return _reduceSumForComplex(lval);
  }
  std::complex<double> reduceSum(const std::complex<double>& lval) const
  {
    return _reduceSumForComplex(lval);
  }
  std::complex<float> reduceSum(const std::complex<float>& lval) const
  {
    return _reduceSumForComplex(lval);
  }

  template <typename T> T reduceSum(const T& lval) const
  {
    return mpAllReduce(m_message_passing_mng.get(), MessagePassing::eReduceType::ReduceSum, lval);
  }

  void waitAll(ArrayView<MessagePassing::Request> requests) const
  {
    mpWaitAll(m_message_passing_mng.get(), requests);
  }
  void wait(MessagePassing::Request request) const
  {
    ArrayView<MessagePassing::Request> requests(1, &request);
    mpWaitAll(m_message_passing_mng.get(), requests);
  }

  /*!
   * \brief Communicator-wise condition checking.
   *
   * Checks conditions at each process in the communicator;
   *
   * If the condition is false on any of the participating processes, outputs the
   * provided message together with the ranks of the offending process.
   * After that each process in the communicator throws.
   */
  void check(bool cond, const String& message);

  void barrier() { mpBarrier(m_message_passing_mng.get()); }

  template <typename T> MessagePassing::Request
  doIReceive(T* buf, int count, int source, int tag) const
  {
    using namespace Arcane::MessagePassing;
    Span<T> s(buf, count);
    Span<unsigned char> schar(reinterpret_cast<unsigned char*>(s.data()), s.sizeBytes());
    PointToPointMessageInfo msg_info(MessageRank{ source }, MessageTag{ tag }, eBlockingType::NonBlocking);
    return mpReceive(m_message_passing_mng.get(), schar, msg_info);
  }

  template <typename T> void
  doReceive(T* buf, int count, int source, int tag) const
  {
    using namespace Arcane::MessagePassing;
    Span<T> s(buf, count);
    Span<unsigned char> schar(reinterpret_cast<unsigned char*>(s.data()), s.sizeBytes());
    PointToPointMessageInfo msg_info(MessageRank{ source }, MessageTag{ tag }, eBlockingType::Blocking);
    mpReceive(m_message_passing_mng.get(), schar, msg_info);
  }

  template <typename T> MessagePassing::Request
  doISend(const T* buf, int count, int dest, int tag) const
  {
    using namespace Arcane::MessagePassing;
    Span<const T> s(buf, count);
    Span<const unsigned char> schar(reinterpret_cast<const unsigned char*>(s.data()), s.sizeBytes());
    PointToPointMessageInfo msg_info(MessageRank{ dest }, MessageTag{ tag }, eBlockingType::NonBlocking);
    return mpSend(m_message_passing_mng.get(), schar, msg_info);
  }

  template <typename T> void
  doSend(const T* buf, int count, int dest, int tag) const
  {
    using namespace Arcane::MessagePassing;
    Span<const T> s(buf, count);
    Span<const unsigned char> schar(reinterpret_cast<const unsigned char*>(s.data()), s.sizeBytes());
    PointToPointMessageInfo msg_info(MessageRank{ dest }, MessageTag{ tag }, eBlockingType::Blocking);
    mpSend(m_message_passing_mng.get(), schar, msg_info);
  }

 private:

  template <typename T> std::complex<T>
  _reduceSumForComplex(const std::complex<T>& lval) const
  {
    // Specialisation for 'std::complex<T>' as 2 T.
    FixedArray<T, 2> values = { { lval.real(), lval.imag() } };
    mpAllReduce(m_message_passing_mng.get(), MessagePassing::eReduceType::ReduceSum, values.view());
    return std::complex<T>(values[0], values[1]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
