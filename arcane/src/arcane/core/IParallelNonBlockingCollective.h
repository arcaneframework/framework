// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelNonBlockingCollective.h                            (C) 2000-2025 */
/*                                                                           */
/* Interface for non-blocking collective parallel operations.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPARALLELNONBLOCKINGCOLLECTIVE_H
#define ARCANE_CORE_IPARALLELNONBLOCKINGCOLLECTIVE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include "arcane/core/Parallel.h"
#include "arcane/core/VariableTypedef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * NOTE:
 * The goal is that IParallelNonBlockingCollective possesses the same
 * collective methods as IParallelMng. However, some collective methods in
 * IParallleMng actually call multiple collective operations in their
 * implementation. It is therefore not possible to transform this directly
 * into collective operations. To implement this with MPI, it would be
 * necessary to be able to associate a callback with each request (this
 * callback would be called when the request is finished) which would allow
 * operations to continue. But this is not currently available (perhaps this
 * is possible with generalized requests). For now, we remove these calls
 * from the interface by protecting them with a define _NEED_ADVANCED_NBC.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Parallel
 * \brief Interface for non-blocking collective parallel operations.
 */
class ARCANE_CORE_EXPORT IParallelNonBlockingCollective
{
 public:

  virtual ~IParallelNonBlockingCollective() = default; //!< Releases resources.

 public:

  typedef Parallel::Request Request;
  typedef Parallel::eReduceType eReduceType;

 public:

  //! Constructs the instance.
  virtual void build() = 0;

 public:

  //! Associated parallelism manager.
  virtual IParallelMng* parallelMng() const = 0;

 public:

  //! @name allGather
  //@{
  /*!
   * \brief Performs a gather on all processors.
   * This is a collective operation. The array \a send_buf
   * must have the same size, denoted \a n, for all processors and
   * the array \a recv_buf must have a size equal to the number
   * of processors multiplied by \a n.
   */
  virtual Request allGather(ConstArrayView<char> send_buf, ArrayView<char> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<unsigned char> send_buf, ArrayView<unsigned char> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<signed char> send_buf, ArrayView<signed char> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<int> send_buf, ArrayView<int> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<unsigned int> send_buf, ArrayView<unsigned int> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<short> send_buf, ArrayView<short> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<unsigned short> send_buf, ArrayView<unsigned short> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<long> send_buf, ArrayView<long> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<unsigned long> send_buf, ArrayView<unsigned long> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<long long> send_buf, ArrayView<long long> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<unsigned long long> send_buf, ArrayView<unsigned long long> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<float> send_buf, ArrayView<float> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<double> send_buf, ArrayView<double> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<long double> send_buf, ArrayView<long double> recv_buf) = 0;
#ifdef ARCANE_REAL_NOT_BUILTIN
  virtual Request allGather(ConstArrayView<Real> send_buf, ArrayView<Real> recv_buf) = 0;
#endif
  virtual Request allGather(ConstArrayView<Real2> send_buf, ArrayView<Real2> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<Real3> send_buf, ArrayView<Real3> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<Real2x2> send_buf, ArrayView<Real2x2> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<Real3x3> send_buf, ArrayView<Real3x3> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<HPReal> send_buf, ArrayView<HPReal> recv_buf) = 0;
  //virtual Request allGather(ISerializer* send_serializer,ISerializer* recv_serializer) =0;
  //@}

  //! @name gather
  //@{
  /*!
   * \brief Performs a gather on one processor.
   * This is a collective operation. The array \a send_buf
   * must have the same size, denoted \a n, for all processors and
   * the array \a recv_buf for processor \a rank must have a size equal to the number
   * of processors multiplied by \a n. This array \a recv_buf is unused for
   * ranks other than \a rank.
   */
  virtual Request gather(ConstArrayView<char> send_buf, ArrayView<char> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<unsigned char> send_buf, ArrayView<unsigned char> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<signed char> send_buf, ArrayView<signed char> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<int> send_buf, ArrayView<int> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<unsigned int> send_buf, ArrayView<unsigned int> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<short> send_buf, ArrayView<short> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<unsigned short> send_buf, ArrayView<unsigned short> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<long> send_buf, ArrayView<long> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<unsigned long> send_buf, ArrayView<unsigned long> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<long long> send_buf, ArrayView<long long> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<unsigned long long> send_buf, ArrayView<unsigned long long> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<float> send_buf, ArrayView<float> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<double> send_buf, ArrayView<double> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<long double> send_buf, ArrayView<long double> recv_buf, Integer rank) = 0;
#ifdef ARCANE_REAL_NOT_BUILTIN
  virtual Request gather(ConstArrayView<Real> send_buf, ArrayView<Real> recv_buf, Integer rank) = 0;
#endif
  virtual Request gather(ConstArrayView<Real2> send_buf, ArrayView<Real2> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<Real3> send_buf, ArrayView<Real3> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<Real2x2> send_buf, ArrayView<Real2x2> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<Real3x3> send_buf, ArrayView<Real3x3> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<HPReal> send_buf, ArrayView<HPReal> recv_buf, Integer rank) = 0;
  //virtual void gather(ISerializer* send_serializer,ISerializer* recv_serializer,Integer rank) =0;
  //@}

  //! @name allGather variable
  //@{

#if _NEED_ADVANCED_NBC
  /*!
   * \brief Performs a gather on all processors.
   *
   * This is a collective operation. The number of elements in the array
   * \a send_buf can be different for each processor. The array
   * \a recv_buf contains the concatenation of the arrays \a send_buf
   * from each processor as output. This array \a recv_buf may be resized
   * for the processor of rank \a rank.
   */
  virtual Request gatherVariable(ConstArrayView<char> send_buf,
                                 Array<char>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<signed char> send_buf,
                                 Array<signed char>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<unsigned char> send_buf,
                                 Array<unsigned char>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<int> send_buf,
                                 Array<int>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<unsigned int> send_buf,
                                 Array<unsigned int>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<short> send_buf,
                                 Array<short>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<unsigned short> send_buf,
                                 Array<unsigned short>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<long> send_buf,
                                 Array<long>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<unsigned long> send_buf,
                                 Array<unsigned long>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<long long> send_buf,
                                 Array<long long>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<unsigned long long> send_buf,
                                 Array<unsigned long long>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<float> send_buf,
                                 Array<float>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<double> send_buf,
                                 Array<double>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<long double> send_buf,
                                 Array<long double>& recv_buf, Integer rank) = 0;
#ifdef ARCANE_REAL_NOT_BUILTIN
  virtual Request gatherVariable(ConstArrayView<Real> send_buf,
                                 Array<Real>& recv_buf, Integer rank) = 0;
#endif
  virtual Request gatherVariable(ConstArrayView<Real2> send_buf,
                                 Array<Real2>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<Real3> send_buf,
                                 Array<Real3>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<Real2x2> send_buf,
                                 Array<Real2x2>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<Real3x3> send_buf,
                                 Array<Real3x3>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<HPReal> send_buf,
                                 Array<HPReal>& recv_buf, Integer rank) = 0;
  //@}
#endif

  //! @name allGather variable
  //@{

#if _NEED_ADVANCED_NBC
  /*!
   * \brief Performs a gather on all processors.
   *
   * This is a collective operation. The number of elements in the array
   * \a send_buf can be different for each processor. The array
   * \a recv_buf contains the concatenation of the arrays \a send_buf
   * from each processor as output. This array \a recv_buf may be resized.
   */
  virtual Request allGatherVariable(ConstArrayView<char> send_buf,
                                    Array<char>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<signed char> send_buf,
                                    Array<signed char>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<unsigned char> send_buf,
                                    Array<unsigned char>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<int> send_buf,
                                    Array<int>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<unsigned int> send_buf,
                                    Array<unsigned int>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<short> send_buf,
                                    Array<short>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<unsigned short> send_buf,
                                    Array<unsigned short>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<long> send_buf,
                                    Array<long>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<unsigned long> send_buf,
                                    Array<unsigned long>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<long long> send_buf,
                                    Array<long long>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<unsigned long long> send_buf,
                                    Array<unsigned long long>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<float> send_buf,
                                    Array<float>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<double> send_buf,
                                    Array<double>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<long double> send_buf,
                                    Array<long double>& recv_buf) = 0;
#ifdef ARCANE_REAL_NOT_BUILTIN
  virtual Request allGatherVariable(ConstArrayView<Real> send_buf,
                                    Array<Real>& recv_buf) = 0;
#endif
  virtual Request allGatherVariable(ConstArrayView<Real2> send_buf,
                                    Array<Real2>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<Real3> send_buf,
                                    Array<Real3>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<Real2x2> send_buf,
                                    Array<Real2x2>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<Real3x3> send_buf,
                                    Array<Real3x3>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<HPReal> send_buf,
                                    Array<HPReal>& recv_buf) = 0;
  //@}
#endif

#if _NEED_ADVANCED_NBC
  //! @name scalar reduction operations
  //@{
  /*!
   * \brief Splits an array across multiple processors.
   */
  virtual Request scatterVariable(ConstArrayView<char> send_buf,
                                  ArrayView<char> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<signed char> send_buf,
                                  ArrayView<signed char> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<unsigned char> send_buf,
                                  ArrayView<unsigned char> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<int> send_buf,
                                  ArrayView<int> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<unsigned int> send_buf,
                                  ArrayView<unsigned int> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<long> send_buf,
                                  ArrayView<long> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<unsigned long> send_buf,
                                  ArrayView<unsigned long> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<long long> send_buf,
                                  ArrayView<long long> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<unsigned long long> send_buf,
                                  ArrayView<unsigned long long> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<float> send_buf,
                                  ArrayView<float> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<double> send_buf,
                                  ArrayView<double> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<long double> send_buf,
                                  ArrayView<long double> recv_buf, Integer root) = 0;
#ifdef ARCANE_REAL_NOT_BUILTIN
  virtual Request scatterVariable(ConstArrayView<Real> send_buf,
                                  ArrayView<Real> recv_buf, Integer root) = 0;
#endif
  virtual Request scatterVariable(ConstArrayView<Real2> send_buf,
                                  ArrayView<Real2> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<Real3> send_buf,
                                  ArrayView<Real3> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<Real2x2> send_buf,
                                  ArrayView<Real2x2> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<Real3x3> send_buf,
                                  ArrayView<Real3x3> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<HPReal> send_buf,
                                  ArrayView<HPReal> recv_buf, Integer root) = 0;
  //@}
#endif

  //! @name Array reduction operations
  //@{
  /*!
   * \brief Performs the reduction of type \a rt on the array \a send_buf and
   * stores the result in \a recv_buf.
   */
  virtual Request allReduce(eReduceType rt, ConstArrayView<char> send_buf, ArrayView<char> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<signed char> send_buf, ArrayView<signed char> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<unsigned char> send_buf, ArrayView<unsigned char> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<short> send_buf, ArrayView<short> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<unsigned short> send_buf, ArrayView<unsigned short> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<int> send_buf, ArrayView<int> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<unsigned int> send_buf, ArrayView<unsigned int> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<long> send_buf, ArrayView<long> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<unsigned long> send_buf, ArrayView<unsigned long> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<long long> send_buf, ArrayView<long long> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<unsigned long long> send_buf, ArrayView<unsigned long long> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<float> send_buf, ArrayView<float> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<double> send_buf, ArrayView<double> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<long double> send_buf, ArrayView<long double> recv_buf) = 0;
#ifdef ARCANE_REAL_NOT_BUILTIN
  virtual Request allReduce(eReduceType rt, ConstArrayView<Real> send_buf, ArrayView<Real> recv_buf) = 0;
#endif
  virtual Request allReduce(eReduceType rt, ConstArrayView<Real2> send_buf, ArrayView<Real2> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<Real3> send_buf, ArrayView<Real3> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<Real2x2> send_buf, ArrayView<Real2x2> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<Real3x3> send_buf, ArrayView<Real3x3> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<HPReal> send_buf, ArrayView<HPReal> recv_buf) = 0;
  //@}

  /*!
   * @name Broadcast operations
   *
   * \brief Sends an array of values to all subdomains.
   *
   * This operation sends the value array \a send_buf to all
   * subdomains. The array used is the one whose rank (commRank) is \a rank.
   * All participating subdomains must call this method with
   * the same parameter \a rank and have an array \a send_buf
   * containing the same number of elements.
   */
  //@{
  virtual Request broadcast(ArrayView<char> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<signed char> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<unsigned char> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<short> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<unsigned short> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<int> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<unsigned int> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<long> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<unsigned long> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<long long> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<unsigned long long> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<float> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<double> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<long double> send_buf, Integer rank) = 0;
#ifdef ARCANE_REAL_NOT_BUILTIN
  virtual Request broadcast(ArrayView<Real> send_buf, Integer rank) = 0;
#endif
  virtual Request broadcast(ArrayView<Real2> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<Real3> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<Real2x2> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<Real3x3> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<HPReal> send_buf, Integer rank) = 0;
  //virtual Request broadcastString(String& str,Integer rank) =0;

  //virtual Request broadcastSerializer(ISerializer* values,Integer rank) =0;
  /*! \brief Performs a broadcast of a memory region.
   *
   * The processor performing the broadcast is given by \id. The array
   * sent is then given by \a bytes. The receiving processors receive
   * the array in \a bytes. This array is allocated automatically; receiving
   * processors do not need to know the number of bytes to be sent.
   *
   */
  //virtual Request broadcastMemoryBuffer(ByteArray& bytes,Integer rank) =0;
  //@}

  virtual Request allToAll(ConstArrayView<char> send_buf, ArrayView<char> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<signed char> send_buf, ArrayView<signed char> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<unsigned char> send_buf, ArrayView<unsigned char> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<int> send_buf, ArrayView<int> recv_buf, Integer count) = 0;
  virtual Request allToAll(ConstArrayView<unsigned int> send_buf, ArrayView<unsigned int> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<short> send_buf, ArrayView<short> recv_buf, Integer count) = 0;
  virtual Request allToAll(ConstArrayView<unsigned short> send_buf, ArrayView<unsigned short> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<long> send_buf, ArrayView<long> recv_buf, Integer count) = 0;
  virtual Request allToAll(ConstArrayView<unsigned long> send_buf, ArrayView<unsigned long> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<long long> send_buf, ArrayView<long long> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<unsigned long long> send_buf,
                           ArrayView<unsigned long long> recv_buf, Integer count) = 0;
  virtual Request allToAll(ConstArrayView<float> send_buf, ArrayView<float> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<double> send_buf, ArrayView<double> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<long double> send_buf, ArrayView<long double> recv_buf,
                           Integer count) = 0;
#ifdef ARCANE_REAL_NOT_BUILTIN
  virtual Request allToAll(ConstArrayView<Real> send_buf, ArrayView<Real> recv_buf,
                           Integer count) = 0;
#endif
  virtual Request allToAll(ConstArrayView<Real2> send_buf, ArrayView<Real2> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<Real3> send_buf, ArrayView<Real3> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<Real2x2> send_buf, ArrayView<Real2x2> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<Real3x3> send_buf, ArrayView<Real3x3> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<HPReal> send_buf, ArrayView<HPReal> recv_buf,
                           Integer count) = 0;

  /*! @name allToAll variable
   *
   * \brief Performs a variable allToAll.
   *
   */
  //@{
  virtual Request allToAllVariable(ConstArrayView<char> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<char> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<signed char> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<signed char> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<unsigned char> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<unsigned char> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<int> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<int> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<unsigned int> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<unsigned int> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<short> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<short> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<unsigned short> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<unsigned short> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<long> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<long> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<unsigned long> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<unsigned long> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<long long> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<long long> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<unsigned long long> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<unsigned long long> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<float> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<float> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<double> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<double> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<long double> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<long double> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
#ifdef ARCANE_REAL_NOT_BUILTIN
  virtual Request allToAllVariable(ConstArrayView<Real> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<Real> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
#endif
  virtual Request allToAllVariable(ConstArrayView<Real2> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<Real2> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<Real3> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<Real3> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<Real2x2> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<Real2x2> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<Real3x3> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<Real3x3> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<HPReal> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<HPReal> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  //@}

  //! @name Synchronization and asynchronous operations
  //@{
  //! Performs a barrier
  virtual Request barrier() = 0;
  //@}

  /*!
   * \brief Indicates if the implementation allows reductions on derived types.
   *
   * OpenMPI versions up to and including 1.8.4 seem to have a bug (which results in a crash)
   * with non-blocking reductions when the reduction operator is redefined. This is the case with
   * derived types such as Real3, Real2, ...
   */
  virtual bool hasValidReduceForDerivedType() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
