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

/// Converts C type to MPI datatype.
template <class T, class Enable = void>
struct mpi_datatype_impl
{
  static MPI_Datatype get()
  {
    static const MPI_Datatype t = create();
    return t;
  }

  static MPI_Datatype create()
  {
    typedef typename math::scalar_of<T>::type S;
    MPI_Datatype t;
    int n = sizeof(T) / sizeof(S);
    MPI_Type_contiguous(n, mpi_datatype_impl<S>::get(), &t);
    MPI_Type_commit(&t);
    return t;
  }
};

template <>
struct mpi_datatype_impl<float>
{
  static MPI_Datatype get() { return MPI_FLOAT; }
};

template <>
struct mpi_datatype_impl<double>
{
  static MPI_Datatype get() { return MPI_DOUBLE; }
};

template <>
struct mpi_datatype_impl<long double>
{
  static MPI_Datatype get() { return MPI_LONG_DOUBLE; }
};

template <>
struct mpi_datatype_impl<int>
{
  static MPI_Datatype get() { return MPI_INT; }
};

template <>
struct mpi_datatype_impl<unsigned>
{
  static MPI_Datatype get() { return MPI_UNSIGNED; }
};

template <>
struct mpi_datatype_impl<long long>
{
  static MPI_Datatype get() { return MPI_LONG_LONG_INT; }
};

template <>
struct mpi_datatype_impl<unsigned long long>
{
  static MPI_Datatype get() { return MPI_UNSIGNED_LONG_LONG; }
};

#if (MPI_VERSION > 2) || (MPI_VERSION == 2 && MPI_SUBVERSION >= 2)
template <>
struct mpi_datatype_impl<std::complex<double>>
{
  static MPI_Datatype get() { return MPI_CXX_DOUBLE_COMPLEX; }
};

template <>
struct mpi_datatype_impl<std::complex<float>>
{
  static MPI_Datatype get() { return MPI_CXX_FLOAT_COMPLEX; }
};
#endif

template <typename T>
struct mpi_datatype_impl<T,
                         typename std::enable_if<
                         std::is_same<T, ptrdiff_t>::value &&
                         !std::is_same<ptrdiff_t, long long>::value &&
                         !std::is_same<ptrdiff_t, int>::value>::type> : std::conditional<sizeof(ptrdiff_t) == sizeof(int), mpi_datatype_impl<int>, mpi_datatype_impl<long long>>::type
{};

template <typename T>
struct mpi_datatype_impl<T,
                         typename std::enable_if<
                         std::is_same<T, size_t>::value &&
                         !std::is_same<size_t, unsigned long long>::value &&
                         !std::is_same<ptrdiff_t, unsigned int>::value>::type>
: std::conditional<
  sizeof(size_t) == sizeof(unsigned), mpi_datatype_impl<unsigned>, mpi_datatype_impl<unsigned long long>>::type
{};

template <>
struct mpi_datatype_impl<char>
{
  static MPI_Datatype get() { return MPI_CHAR; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Wrapper to obtain the equivalent MPI datatype for a datatype.
 */
template <typename T>
MPI_Datatype mpi_datatype()
{
  return mpi_datatype_impl<T>::get();
}

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
struct mpi_communicator
{
  MPI_Comm comm = MPI_COMM_NULL;
  int rank = 0;
  int size = 0;
  Ref<IMessagePassingMng> m_message_passing_mng;

  mpi_communicator() = default;

  explicit mpi_communicator(MPI_Comm comm)
  : comm(comm)
  {
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    m_message_passing_mng = MessagePassing::Mpi::StandaloneMpiMessagePassingMng::createRef(comm);
  };

  operator MPI_Comm() const
  {
    return comm;
  }

  /// Exclusive sum over mpi communicator
  template <typename T>
  std::vector<T> exclusive_sum(T n) const
  {
    // TODO: Utiliser scan.
    std::vector<T> v(size + 1);
    v[0] = 0;
    MPI_Allgather(&n, 1, mpi_datatype<T>(), &v[1], 1, mpi_datatype<T>(), comm);
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
  template <class Condition, class Message>
  void check(const Condition& cond, const Message& message)
  {
    int lc = static_cast<int>(cond);
    int gc = _reduce(MPI_PROD, lc);

    if (!gc) {
      std::vector<int> c(size);
      MPI_Gather(&lc, 1, MPI_INT, &c[0], size, MPI_INT, 0, comm);
      if (rank == 0) {
        std::cerr << "Failed assumption: " << message << std::endl;
        std::cerr << "Offending processes:";
        for (int i = 0; i < size; ++i)
          if (!c[i])
            std::cerr << " " << i;
        std::cerr << std::endl;
      }
      MPI_Barrier(comm);
      ARCCORE_FATAL("CheckError in MessagePassingUtils: {0}", message);
    }
  }

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

  template <typename T> T _reduce(MPI_Op op, const T& lval) const
  {
    const int elems = math::static_rows<T>::value * math::static_cols<T>::value;
    T gval;

    MPI_Allreduce((void*)&lval, &gval, elems, mpi_datatype<T>(), op, comm);
    return gval;
  }

  template <typename T> std::complex<T>
  _reduceSumForComplex(const std::complex<T>& lval) const
  {
    // Specialisation for 'std::complex<float>' as 2 float.
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
