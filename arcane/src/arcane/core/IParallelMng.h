// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelMng.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface of the parallelism manager on a subdomain.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPARALLELMNG_H
#define ARCANE_CORE_IPARALLELMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include "arcane/core/Parallel.h"
#include "arcane/core/VariableTypedef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Parallel
 * \brief Interface of the parallelism manager for a subdomain.
 *
 * This manager provides an interface to access
 * all functionalities related to parallelism.
 *
 * Several possible implementations exist:
 * - sequential mode.
 * - parallel mode via MPI
 * - parallel mode via threads.
 * - mixed MPI/threads parallel mode.
 * The choice of implementation is made when launching the application.
 *
 * When an operation is collective, all associated managers must
 * participate.
 *
 * It is possible to create another manager from an instance
 * containing a subset of ranks via createSubParallelMng().
 *
 */
class ARCANE_CORE_EXPORT IParallelMng
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();
  // Class to access _internalUtilsFactory()
  friend class ParallelMngUtilsAccessor;

 public:

  // NOTE: Temporarily keeping this public destructor while
  // createParallelMng() methods exist for compatibility with the existing code
  virtual ~IParallelMng() = default; //!< Releases resources.

 public:

  typedef Parallel::Request Request;
  using PointToPointMessageInfo = Parallel::PointToPointMessageInfo;
  using MessageId = Parallel::MessageId;
  using MessageSourceInfo = Parallel::MessageSourceInfo;
  typedef Parallel::eReduceType eReduceType;
  typedef Parallel::IStat IStat;

 public:

  //! Constructs the instance.
  virtual void build() = 0;

 public:

  /*!
   * \brief Returns true if the execution is parallel.
   *
   * The execution is parallel if the instance implements
   * a message exchange mechanism such as MPI.
   */
  virtual bool isParallel() const = 0;

 private:

  // NOTE: Temporarily leaving these two methods to maintain binary compatibility.

  //! Subdomain number associated with this manager.
  virtual ARCANE_DEPRECATED Integer subDomainId() const final { return commRank(); }

  //! Total number of subdomains.
  virtual ARCANE_DEPRECATED Integer nbSubDomain() const final { return commSize(); }

 public:

  //! Rank of this instance in the communicator
  virtual Int32 commRank() const = 0;

  //! Number of instances in the communicator
  virtual Int32 commSize() const = 0;

  /*!
   * \brief Address of the MPI communicator associated with this manager.
   *
   * The communicator is only valid if MPI is used. Otherwise, the returned address
   * is 0. The returned value is of type (MPI_Comm*).
   */
  virtual void* getMPICommunicator() = 0;

  /*!
   * \brief Address of the MPI communicator associated with this manager.
   *
   * \deprecated Use getMPICommunicator() instead.
   */
  virtual ARCANE_DEPRECATED_120 void* mpiCommunicator();

  /*!
   * \brief MPI communicator associated with this manager
   *
   * The communicator is only valid if MPI is used. It is possible
   * to test its validity by calling the Communicator::isValid() method.
   * If it is valid, it is possible to retrieve its value via a cast:
   * \code
   * IParallelMng* pm = ...;
   * MPI_Comm c = static_cast<MPI_Comm>(pm->communicator());
   * \endcode
   */
  virtual Parallel::Communicator communicator() const = 0;

  /**
   * \brief MPI communicator derived from the communicator \a communicator()
   * gathering all processes of the compute node.
   *
   * The communicator is only valid if MPI is used. It is possible
   * to test its validity by calling the Communicator::isValid() method.
   * If it is valid, it is possible to retrieve its value via a cast:
   * \code
   * IParallelMng* pm = ...;
   * MPI_Comm mc = static_cast<MPI_Comm>(pm->machineCommunicator());
   * \endcode
   */
  virtual Parallel::Communicator machineCommunicator() const { return {}; }

  /*!
   * \brief Indicates if the implementation uses threads.
   *
   * The implementation uses threads either in pure thread mode,
   * or in mixed MPI/thread mode.
   */
  virtual bool isThreadImplementation() const = 0;

  /*!
   * \brief Indicates if the implementation uses hybrid mode.
   *
   * The implementation uses mixed MPI/thread mode.
   */
  virtual bool isHybridImplementation() const = 0;

  //! Sets the statistics manager
  virtual void setTimeStats(ITimeStats* time_stats) = 0;

  //! Associated statistics manager (can be null)
  virtual ITimeStats* timeStats() const = 0;

  //! Trace manager
  virtual ITraceMng* traceMng() const = 0;

  //! Thread manager
  virtual IThreadMng* threadMng() const = 0;

  //! Timer manager
  virtual ITimerMng* timerMng() const = 0;

  //! I/O manager
  virtual IIOMng* ioMng() const = 0;

  //! Parallelism manager over all allocated resources
  virtual IParallelMng* worldParallelMng() const = 0;

  //! Initializes the parallelism manager
  virtual void initialize() = 0;

  //! Arccore temporal statistics collector (can be null)
  virtual ITimeMetricCollector* timeMetricCollector() const = 0;

 public:
 public:

  //! \a true if the instance is a master I/O manager.
  virtual bool isMasterIO() const = 0;

  /*!
    \brief Rank of the instance managing I/O (for which isMasterIO() is true)
    *
    * In the current implementation, this is always the rank 0 processor.
    */
  virtual Integer masterIORank() const = 0;

  //! @name allGather
  //@{
  /*!
   * \brief Performs an all-gather operation across all processors.
   * This is a collective operation. The array \a send_buf
   * must have the same size, denoted \a n, for all processors, and
   * the array \a recv_buf must have a size equal to the number
   * of processors multiplied by \a n.
   */
  virtual void allGather(ConstArrayView<char> send_buf, ArrayView<char> recv_buf) = 0;
  virtual void allGather(ConstArrayView<unsigned char> send_buf, ArrayView<unsigned char> recv_buf) = 0;
  virtual void allGather(ConstArrayView<signed char> send_buf, ArrayView<signed char> recv_buf) = 0;
  virtual void allGather(ConstArrayView<short> send_buf, ArrayView<short> recv_buf) = 0;
  virtual void allGather(ConstArrayView<unsigned short> send_buf, ArrayView<unsigned short> recv_buf) = 0;
  virtual void allGather(ConstArrayView<int> send_buf, ArrayView<int> recv_buf) = 0;
  virtual void allGather(ConstArrayView<unsigned int> send_buf, ArrayView<unsigned int> recv_buf) = 0;
  virtual void allGather(ConstArrayView<long> send_buf, ArrayView<long> recv_buf) = 0;
  virtual void allGather(ConstArrayView<unsigned long> send_buf, ArrayView<unsigned long> recv_buf) = 0;
  virtual void allGather(ConstArrayView<long long> send_buf, ArrayView<long long> recv_buf) = 0;
  virtual void allGather(ConstArrayView<unsigned long long> send_buf, ArrayView<unsigned long long> recv_buf) = 0;
  virtual void allGather(ConstArrayView<float> send_buf, ArrayView<float> recv_buf) = 0;
  virtual void allGather(ConstArrayView<double> send_buf, ArrayView<double> recv_buf) = 0;
  virtual void allGather(ConstArrayView<long double> send_buf, ArrayView<long double> recv_buf) = 0;
  virtual void allGather(ConstArrayView<APReal> send_buf, ArrayView<APReal> recv_buf) = 0;
  virtual void allGather(ConstArrayView<Real2> send_buf, ArrayView<Real2> recv_buf) = 0;
  virtual void allGather(ConstArrayView<Real3> send_buf, ArrayView<Real3> recv_buf) = 0;
  virtual void allGather(ConstArrayView<Real2x2> send_buf, ArrayView<Real2x2> recv_buf) = 0;
  virtual void allGather(ConstArrayView<Real3x3> send_buf, ArrayView<Real3x3> recv_buf) = 0;
  virtual void allGather(ConstArrayView<HPReal> send_buf, ArrayView<HPReal> recv_buf) = 0;
  virtual void allGather(ISerializer* send_serializer, ISerializer* recv_serializer) = 0;
  //@}

  //! @name gather
  //@{
  /*!
   * \brief Performs a gather operation onto a processor.
   * This is a collective operation. The array \a send_buf
   * must have the same size, denoted \a n, for all processors, and
   * the array \a recv_buf for processor \a rank must have a size equal to the number
   * of processors multiplied by \a n. This array \a recv_buf is unused for
   * other ranks than \a rank.
   */
  virtual void gather(ConstArrayView<char> send_buf, ArrayView<char> recv_buf, Int32 rank) = 0;
  virtual void gather(ConstArrayView<unsigned char> send_buf, ArrayView<unsigned char> recv_buf, Int32 rank) = 0;
  virtual void gather(ConstArrayView<signed char> send_buf, ArrayView<signed char> recv_buf, Int32 rank) = 0;
  virtual void gather(ConstArrayView<short> send_buf, ArrayView<short> recv_buf, Int32 rank) = 0;
  virtual void gather(ConstArrayView<unsigned short> send_buf, ArrayView<unsigned short> recv_buf, Int32 rank) = 0;
  virtual void gather(ConstArrayView<int> send_buf, ArrayView<int> recv_buf, Int32 rank) = 0;
  virtual void gather(ConstArrayView<unsigned int> send_buf, ArrayView<unsigned int> recv_buf, Int32 rank) = 0;
  virtual void gather(ConstArrayView<long> send_buf, ArrayView<long> recv_buf, Int32 rank) = 0;
  virtual void gather(ConstArrayView<unsigned long> send_buf, ArrayView<unsigned long> recv_buf, Int32 rank) = 0;
  virtual void gather(ConstArrayView<long long> send_buf, ArrayView<long long> recv_buf, Int32 rank) = 0;
  virtual void gather(ConstArrayView<unsigned long long> send_buf, ArrayView<unsigned long long> recv_buf, Int32 rank) = 0;
  virtual void gather(ConstArrayView<float> send_buf, ArrayView<float> recv_buf, Int32 rank) = 0;
  virtual void gather(ConstArrayView<double> send_buf, ArrayView<double> recv_buf, Int32 rank) = 0;
  virtual void gather(ConstArrayView<long double> send_buf, ArrayView<long double> recv_buf, Int32 rank) = 0;
  virtual void gather(ConstArrayView<APReal> send_buf, ArrayView<APReal> recv_buf, Int32 rank) = 0;
  virtual void gather(ConstArrayView<Real2> send_buf, ArrayView<Real2> recv_buf, Int32 rank) = 0;
  virtual void gather(ConstArrayView<Real3> send_buf, ArrayView<Real3> recv_buf, Int32 rank) = 0;
  virtual void gather(ConstArrayView<Real2x2> send_buf, ArrayView<Real2x2> recv_buf, Int32 rank) = 0;
  virtual void gather(ConstArrayView<Real3x3> send_buf, ArrayView<Real3x3> recv_buf, Int32 rank) = 0;
  virtual void gather(ConstArrayView<HPReal> send_buf, ArrayView<HPReal> recv_buf, Int32 rank) = 0;
  //@}

  //! @name allGather variable
  //@{

  /*!
   * \brief Performs an all-gather operation across all processors.
   *
   * This is a collective operation. The number of elements in the array
   * \a send_buf may be different for each processor. The array
   * \a recv_buf contains the concatenation of the \a send_buf
   * arrays from each processor. This array \a recv_buf may be resized
   * for the processor of rank \a rank.
   */
  virtual void gatherVariable(ConstArrayView<char> send_buf,
                              Array<char>& recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<signed char> send_buf,
                              Array<signed char>& recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<unsigned char> send_buf,
                              Array<unsigned char>& recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<short> send_buf,
                              Array<short>& recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<unsigned short> send_buf,
                              Array<unsigned short>& recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<int> send_buf,
                              Array<int>& recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<unsigned int> send_buf,
                              Array<unsigned int>& recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<long> send_buf,
                              Array<long>& recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<unsigned long> send_buf,
                              Array<unsigned long>& recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<long long> send_buf,
                              Array<long long>& recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<unsigned long long> send_buf,
                              Array<unsigned long long>& recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<float> send_buf,
                              Array<float>& recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<double> send_buf,
                              Array<double>& recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<long double> send_buf,
                              Array<long double>& recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<APReal> send_buf,
                              Array<APReal>& recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<Real2> send_buf,
                              Array<Real2>& recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<Real3> send_buf,
                              Array<Real3>& recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<Real2x2> send_buf,
                              Array<Real2x2>& recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<Real3x3> send_buf,
                              Array<Real3x3>& recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<HPReal> send_buf,
                              Array<HPReal>& recv_buf, Int32 rank) = 0;
  //@}

  //! @name allGather variable
  //@{

  /*!
   * \brief Performs an all-gather operation across all processors.
   *
   * This is a collective operation. The number of elements in the array
   * \a send_buf may be different for each processor. The array
   * \a recv_buf contains the concatenation of the \a send_buf
   * arrays from each processor. This array \a recv_buf may be resized.
   */
  virtual void allGatherVariable(ConstArrayView<char> send_buf,
                                 Array<char>& recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<signed char> send_buf,
                                 Array<signed char>& recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<unsigned char> send_buf,
                                 Array<unsigned char>& recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<short> send_buf,
                                 Array<short>& recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<unsigned short> send_buf,
                                 Array<unsigned short>& recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<int> send_buf,
                                 Array<int>& recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<unsigned int> send_buf,
                                 Array<unsigned int>& recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<long> send_buf,
                                 Array<long>& recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<unsigned long> send_buf,
                                 Array<unsigned long>& recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<long long> send_buf,
                                 Array<long long>& recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<unsigned long long> send_buf,
                                 Array<unsigned long long>& recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<float> send_buf,
                                 Array<float>& recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<double> send_buf,
                                 Array<double>& recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<long double> send_buf,
                                 Array<long double>& recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<APReal> send_buf,
                                 Array<APReal>& recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<Real2> send_buf,
                                 Array<Real2>& recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<Real3> send_buf,
                                 Array<Real3>& recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<Real2x2> send_buf,
                                 Array<Real2x2>& recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<Real3x3> send_buf,
                                 Array<Real3x3>& recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<HPReal> send_buf,
                                 Array<HPReal>& recv_buf) = 0;
  //@}

  //! @name scalar reduction operations
  //@{
  /*!
   * \brief Scatters an array across multiple processors.
   */
  virtual void scatterVariable(ConstArrayView<char> send_buf,
                               ArrayView<char> recv_buf, Integer root) = 0;
  virtual void scatterVariable(ConstArrayView<signed char> send_buf,
                               ArrayView<signed char> recv_buf, Integer root) = 0;
  virtual void scatterVariable(ConstArrayView<unsigned char> send_buf,
                               ArrayView<unsigned char> recv_buf, Integer root) = 0;
  virtual void scatterVariable(ConstArrayView<short> send_buf,
                               ArrayView<short> recv_buf, Integer root) = 0;
  virtual void scatterVariable(ConstArrayView<unsigned short> send_buf,
                               ArrayView<unsigned short> recv_buf, Integer root) = 0;
  virtual void scatterVariable(ConstArrayView<int> send_buf,
                               ArrayView<int> recv_buf, Integer root) = 0;
  virtual void scatterVariable(ConstArrayView<unsigned int> send_buf,
                               ArrayView<unsigned int> recv_buf, Integer root) = 0;
  virtual void scatterVariable(ConstArrayView<long> send_buf,
                               ArrayView<long> recv_buf, Integer root) = 0;
  virtual void scatterVariable(ConstArrayView<unsigned long> send_buf,
                               ArrayView<unsigned long> recv_buf, Integer root) = 0;
  virtual void scatterVariable(ConstArrayView<long long> send_buf,
                               ArrayView<long long> recv_buf, Integer root) = 0;
  virtual void scatterVariable(ConstArrayView<unsigned long long> send_buf,
                               ArrayView<unsigned long long> recv_buf, Integer root) = 0;
  virtual void scatterVariable(ConstArrayView<float> send_buf,
                               ArrayView<float> recv_buf, Integer root) = 0;
  virtual void scatterVariable(ConstArrayView<double> send_buf,
                               ArrayView<double> recv_buf, Integer root) = 0;
  virtual void scatterVariable(ConstArrayView<long double> send_buf,
                               ArrayView<long double> recv_buf, Integer root) = 0;
  virtual void scatterVariable(ConstArrayView<APReal> send_buf,
                               ArrayView<APReal> recv_buf, Integer root) = 0;
  virtual void scatterVariable(ConstArrayView<Real2> send_buf,
                               ArrayView<Real2> recv_buf, Integer root) = 0;
  virtual void scatterVariable(ConstArrayView<Real3> send_buf,
                               ArrayView<Real3> recv_buf, Integer root) = 0;
  virtual void scatterVariable(ConstArrayView<Real2x2> send_buf,
                               ArrayView<Real2x2> recv_buf, Integer root) = 0;
  virtual void scatterVariable(ConstArrayView<Real3x3> send_buf,
                               ArrayView<Real3x3> recv_buf, Integer root) = 0;
  virtual void scatterVariable(ConstArrayView<HPReal> send_buf,
                               ArrayView<HPReal> recv_buf, Integer root) = 0;
  //@}

  //! @name scalar reduction operations
  //@{
  /*!
   * \brief Performs a reduction of type \a rt on the real \a v and returns the value
   */
  virtual char reduce(eReduceType rt, char v) = 0;
  virtual signed char reduce(eReduceType rt, signed char v) = 0;
  virtual unsigned char reduce(eReduceType rt, unsigned char v) = 0;
  virtual short reduce(eReduceType rt, short v) = 0;
  virtual unsigned short reduce(eReduceType rt, unsigned short v) = 0;
  virtual int reduce(eReduceType rt, int v) = 0;
  virtual unsigned int reduce(eReduceType rt, unsigned int v) = 0;
  virtual long reduce(eReduceType rt, long v) = 0;
  virtual unsigned long reduce(eReduceType rt, unsigned long v) = 0;
  virtual long long reduce(eReduceType rt, long long v) = 0;
  virtual unsigned long long reduce(eReduceType rt, unsigned long long v) = 0;
  virtual float reduce(eReduceType rt, float v) = 0;
  virtual double reduce(eReduceType rt, double v) = 0;
  virtual long double reduce(eReduceType rt, long double v) = 0;
  virtual APReal reduce(eReduceType rt, APReal v) = 0;
  virtual Real2 reduce(eReduceType rt, Real2 v) = 0;
  virtual Real3 reduce(eReduceType rt, Real3 v) = 0;
  virtual Real2x2 reduce(eReduceType rt, Real2x2 v) = 0;
  virtual Real3x3 reduce(eReduceType rt, Real3x3 v) = 0;
  virtual HPReal reduce(eReduceType rt, HPReal v) = 0;
  //@}

  //! @name scalar reduction operations
  //@{
  /*!
   * \brief Calculates the sum, min, and max of a value in one operation.
   *
   * Calculates the minimum, maximum, and sum of the value \a val.
   * \param val value used for the calculation
   * \param[out] min_val minimum value
   * \param[out] max_val maximum value
   * \param[out] sum_val sum of values
   * \param[out] min_rank rank of the processor having the minimum value
   * \param[out] max_rank rank of the processor having the maximum value
   */
  virtual void computeMinMaxSum(char val, char& min_val,
                                char& max_val, char& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  virtual void computeMinMaxSum(signed char val, signed char& min_val,
                                signed char& max_val, signed char& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  virtual void computeMinMaxSum(unsigned char val, unsigned char& min_val,
                                unsigned char& max_val, unsigned char& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  virtual void computeMinMaxSum(short val, short& min_val,
                                short& max_val, short& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  virtual void computeMinMaxSum(unsigned short val, unsigned short& min_val,
                                unsigned short& max_val, unsigned short& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  virtual void computeMinMaxSum(int val, int& min_val,
                                int& max_val, int& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  virtual void computeMinMaxSum(unsigned int val, unsigned int& min_val,
                                unsigned int& max_val, unsigned int& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  virtual void computeMinMaxSum(long val, long& min_val,
                                long& max_val, long& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  virtual void computeMinMaxSum(unsigned long val, unsigned long& min_val,
                                unsigned long& max_val, unsigned long& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  virtual void computeMinMaxSum(long long val, long long& min_val,
                                long long& max_val, long long& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  virtual void computeMinMaxSum(unsigned long long val, unsigned long long& min_val,
                                unsigned long long& max_val, unsigned long long& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  virtual void computeMinMaxSum(float val, float& min_val,
                                float& max_val, float& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  virtual void computeMinMaxSum(double val, double& min_val,
                                double& max_val, double& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  virtual void computeMinMaxSum(long double val, long double& min_val,
                                long double& max_val, long double& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  virtual void computeMinMaxSum(APReal val, APReal& min_val,
                                APReal& max_val, APReal& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  virtual void computeMinMaxSum(Real2 val, Real2& min_val,
                                Real2& max_val, Real2& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  virtual void computeMinMaxSum(Real3 val, Real3& min_val,
                                Real3& max_val, Real3& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  virtual void computeMinMaxSum(Real2x2 val, Real2x2& min_val,
                                Real2x2& max_val, Real2x2& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  virtual void computeMinMaxSum(Real3x3 val, Real3x3& min_val,
                                Real3x3& max_val, Real3x3& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  virtual void computeMinMaxSum(HPReal val, HPReal& min_val,
                                HPReal& max_val, HPReal& sum_val,
                                Int32& min_rank, Int32& max_rank) = 0;
  //@}

  //! @name vector reduction operations
  //@{
  /*!
   * \brief Calculates the sum, min, and max of a value in one operation.
   *
   * Calculates the minimum, maximum, and sum of the value \a val.
   * \param val value used for the calculation
   * \param[out] min_val minimum value
   * \param[out] max_val maximum value
   * \param[out] sum_val sum of values
   * \param[out] min_rank rank of the processor having the minimum value
   * \param[out] max_rank rank of the processor having the maximum value
   */
  virtual void computeMinMaxSum(ConstArrayView<char> values, ArrayView<char> min_values,
                                ArrayView<char> max_values, ArrayView<char> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  virtual void computeMinMaxSum(ConstArrayView<signed char> values, ArrayView<signed char> min_values,
                                ArrayView<signed char> max_values, ArrayView<signed char> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  virtual void computeMinMaxSum(ConstArrayView<unsigned char> values, ArrayView<unsigned char> min_values,
                                ArrayView<unsigned char> max_values, ArrayView<unsigned char> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  virtual void computeMinMaxSum(ConstArrayView<short> values, ArrayView<short> min_values,
                                ArrayView<short> max_values, ArrayView<short> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  virtual void computeMinMaxSum(ConstArrayView<unsigned short> values, ArrayView<unsigned short> min_values,
                                ArrayView<unsigned short> max_values, ArrayView<unsigned short> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  virtual void computeMinMaxSum(ConstArrayView<int> values, ArrayView<int> min_values,
                                ArrayView<int> max_values, ArrayView<int> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  virtual void computeMinMaxSum(ConstArrayView<unsigned int> values, ArrayView<unsigned int> min_values,
                                ArrayView<unsigned int> max_values, ArrayView<unsigned int> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  virtual void computeMinMaxSum(ConstArrayView<long> values, ArrayView<long> min_values,
                                ArrayView<long> max_values, ArrayView<long> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  virtual void computeMinMaxSum(ConstArrayView<unsigned long> values, ArrayView<unsigned long> min_values,
                                ArrayView<unsigned long> max_values, ArrayView<unsigned long> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  virtual void computeMinMaxSum(ConstArrayView<long long> values, ArrayView<long long> min_values,
                                ArrayView<long long> max_values, ArrayView<long long> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  virtual void computeMinMaxSum(ConstArrayView<unsigned long long> values, ArrayView<unsigned long long> min_values,
                                ArrayView<unsigned long long> max_values, ArrayView<unsigned long long> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  virtual void computeMinMaxSum(ConstArrayView<float> values, ArrayView<float> min_values,
                                ArrayView<float> max_values, ArrayView<float> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  virtual void computeMinMaxSum(ConstArrayView<double> values, ArrayView<double> min_values,
                                ArrayView<double> max_values, ArrayView<double> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  virtual void computeMinMaxSum(ConstArrayView<long double> values, ArrayView<long double> min_values,
                                ArrayView<long double> max_values, ArrayView<long double> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  virtual void computeMinMaxSum(ConstArrayView<APReal> values, ArrayView<APReal> min_values,
                                ArrayView<APReal> max_values, ArrayView<APReal> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  virtual void computeMinMaxSum(ConstArrayView<Real2> values, ArrayView<Real2> min_values,
                                ArrayView<Real2> max_values, ArrayView<Real2> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  virtual void computeMinMaxSum(ConstArrayView<Real3> values, ArrayView<Real3> min_values,
                                ArrayView<Real3> max_values, ArrayView<Real3> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  virtual void computeMinMaxSum(ConstArrayView<Real2x2> values, ArrayView<Real2x2> min_values,
                                ArrayView<Real2x2> max_values, ArrayView<Real2x2> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  virtual void computeMinMaxSum(ConstArrayView<Real3x3> values, ArrayView<Real3x3> min_values,
                                ArrayView<Real3x3> max_values, ArrayView<Real3x3> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  virtual void computeMinMaxSum(ConstArrayView<HPReal> values, ArrayView<HPReal> min_values,
                                ArrayView<HPReal> max_values, ArrayView<HPReal> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) = 0;
  //@}

  //! @name array reduction operations
  //@{
  /*!
   * \brief Performs the reduction of type \a rt on array \a v.
   */
  virtual void reduce(eReduceType rt, ArrayView<char> v) = 0;
  virtual void reduce(eReduceType rt, ArrayView<signed char> v) = 0;
  virtual void reduce(eReduceType rt, ArrayView<unsigned char> v) = 0;
  virtual void reduce(eReduceType rt, ArrayView<short> v) = 0;
  virtual void reduce(eReduceType rt, ArrayView<unsigned short> v) = 0;
  virtual void reduce(eReduceType rt, ArrayView<int> v) = 0;
  virtual void reduce(eReduceType rt, ArrayView<unsigned int> v) = 0;
  virtual void reduce(eReduceType rt, ArrayView<long> v) = 0;
  virtual void reduce(eReduceType rt, ArrayView<unsigned long> v) = 0;
  virtual void reduce(eReduceType rt, ArrayView<long long> v) = 0;
  virtual void reduce(eReduceType rt, ArrayView<unsigned long long> v) = 0;
  virtual void reduce(eReduceType rt, ArrayView<float> v) = 0;
  virtual void reduce(eReduceType rt, ArrayView<double> v) = 0;
  virtual void reduce(eReduceType rt, ArrayView<long double> v) = 0;
  virtual void reduce(eReduceType rt, ArrayView<APReal> v) = 0;
  virtual void reduce(eReduceType rt, ArrayView<Real2> v) = 0;
  virtual void reduce(eReduceType rt, ArrayView<Real3> v) = 0;
  virtual void reduce(eReduceType rt, ArrayView<Real2x2> v) = 0;
  virtual void reduce(eReduceType rt, ArrayView<Real3x3> v) = 0;
  virtual void reduce(eReduceType rt, ArrayView<HPReal> v) = 0;
  //@}

  /*!
   * @name broadcast operations
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
  virtual void broadcast(ArrayView<char> send_buf, Int32 rank) = 0;
  virtual void broadcast(ArrayView<signed char> send_buf, Int32 rank) = 0;
  virtual void broadcast(ArrayView<unsigned char> send_buf, Int32 rank) = 0;
  virtual void broadcast(ArrayView<short> send_buf, Int32 rank) = 0;
  virtual void broadcast(ArrayView<unsigned short> send_buf, Int32 rank) = 0;
  virtual void broadcast(ArrayView<int> send_buf, Int32 rank) = 0;
  virtual void broadcast(ArrayView<unsigned int> send_buf, Int32 rank) = 0;
  virtual void broadcast(ArrayView<long> send_buf, Int32 rank) = 0;
  virtual void broadcast(ArrayView<unsigned long> send_buf, Int32 rank) = 0;
  virtual void broadcast(ArrayView<long long> send_buf, Int32 rank) = 0;
  virtual void broadcast(ArrayView<unsigned long long> send_buf, Int32 rank) = 0;
  virtual void broadcast(ArrayView<float> send_buf, Int32 rank) = 0;
  virtual void broadcast(ArrayView<double> send_buf, Int32 rank) = 0;
  virtual void broadcast(ArrayView<long double> send_buf, Int32 rank) = 0;
  virtual void broadcast(ArrayView<APReal> send_buf, Int32 rank) = 0;
  virtual void broadcast(ArrayView<Real2> send_buf, Int32 rank) = 0;
  virtual void broadcast(ArrayView<Real3> send_buf, Int32 rank) = 0;
  virtual void broadcast(ArrayView<Real2x2> send_buf, Int32 rank) = 0;
  virtual void broadcast(ArrayView<Real3x3> send_buf, Int32 rank) = 0;
  virtual void broadcast(ArrayView<HPReal> send_buf, Int32 rank) = 0;
  virtual void broadcastString(String& str, Int32 rank) = 0;

  virtual void broadcastSerializer(ISerializer* values, Int32 rank) = 0;
  /*! \brief Performs a broadcast of a memory region.
   *
   * The processor performing the broadcast is given by \id. The array
   * sent is then given by \a bytes. The processors receiving
   * the array in \a bytes. This array is allocated automatically, the processors
   * receiving do not need to know the number of bytes to be sent.
   *
   */
  virtual void broadcastMemoryBuffer(ByteArray& bytes, Int32 rank) = 0;
  //@}

  /*!
   * @name message sending operations
   *
   * \brief Blocking send of an array of values to a subdomain.
   *
   * Sends the values of array \a values to subdomain \a rank.
   * The subdomain must perform a corresponding reception (the subdomain
   * number must be that of this handler and the type and the
   * size of the array must correspond) with the function recvValues().
   * The send is blocking.
   */
  //@{
  virtual void send(ConstArrayView<char> values, Int32 rank) = 0;
  virtual void send(ConstArrayView<signed char> values, Int32 rank) = 0;
  virtual void send(ConstArrayView<unsigned char> values, Int32 rank) = 0;
  virtual void send(ConstArrayView<short> values, Int32 rank) = 0;
  virtual void send(ConstArrayView<unsigned short> values, Int32 rank) = 0;
  virtual void send(ConstArrayView<int> values, Int32 rank) = 0;
  virtual void send(ConstArrayView<unsigned int> values, Int32 rank) = 0;
  virtual void send(ConstArrayView<long> values, Int32 rank) = 0;
  virtual void send(ConstArrayView<unsigned long> values, Int32 rank) = 0;
  virtual void send(ConstArrayView<long long> values, Int32 rank) = 0;
  virtual void send(ConstArrayView<unsigned long long> values, Int32 rank) = 0;
  virtual void send(ConstArrayView<float> values, Int32 rank) = 0;
  virtual void send(ConstArrayView<double> values, Int32 rank) = 0;
  virtual void send(ConstArrayView<long double> values, Int32 rank) = 0;
  virtual void send(ConstArrayView<APReal> values, Int32 rank) = 0;
  virtual void send(ConstArrayView<Real2> values, Int32 rank) = 0;
  virtual void send(ConstArrayView<Real3> values, Int32 rank) = 0;
  virtual void send(ConstArrayView<Real2x2> values, Int32 rank) = 0;
  virtual void send(ConstArrayView<Real3x3> values, Int32 rank) = 0;
  virtual void send(ConstArrayView<HPReal> values, Int32 rank) = 0;

  virtual void sendSerializer(ISerializer* values, Int32 rank) = 0;
  /*!
   * The returned request must be used in waitAllRequests() or
   * freed by calling freeRequests().
   */
  ARCCORE_DEPRECATED_2019("Use createSendSerializer(Int32 rank) instead")
  virtual Parallel::Request sendSerializer(ISerializer* values, Int32 rank, ByteArray& bytes) = 0;

  /*!
   * \brief Creates a non-blocking message to send serialized data to rank \a rank.
   *
   * The message is processed only when processMessages() is called.
   */
  virtual ISerializeMessage* createSendSerializer(Int32 rank) = 0;
  //@}

  //! @name message receiving operations.
  //@{
  /*! Receives array \a values from rank \a rank */
  virtual void recv(ArrayView<char> values, Int32 rank) = 0;
  virtual void recv(ArrayView<signed char> values, Int32 rank) = 0;
  virtual void recv(ArrayView<unsigned char> values, Int32 rank) = 0;
  virtual void recv(ArrayView<short> values, Int32 rank) = 0;
  virtual void recv(ArrayView<unsigned short> values, Int32 rank) = 0;
  virtual void recv(ArrayView<int> values, Int32 rank) = 0;
  virtual void recv(ArrayView<unsigned int> values, Int32 rank) = 0;
  virtual void recv(ArrayView<long> values, Int32 rank) = 0;
  virtual void recv(ArrayView<unsigned long> values, Int32 rank) = 0;
  virtual void recv(ArrayView<long long> values, Int32 rank) = 0;
  virtual void recv(ArrayView<unsigned long long> values, Int32 rank) = 0;
  virtual void recv(ArrayView<float> values, Int32 rank) = 0;
  virtual void recv(ArrayView<double> values, Int32 rank) = 0;
  virtual void recv(ArrayView<long double> values, Int32 rank) = 0;
  virtual void recv(ArrayView<APReal> values, Int32 rank) = 0;
  virtual void recv(ArrayView<Real2> values, Int32 rank) = 0;
  virtual void recv(ArrayView<Real3> values, Int32 rank) = 0;
  virtual void recv(ArrayView<Real2x2> values, Int32 rank) = 0;
  virtual void recv(ArrayView<Real3x3> values, Int32 rank) = 0;
  virtual void recv(ArrayView<HPReal> values, Int32 rank) = 0;
  virtual void recvSerializer(ISerializer* values, Int32 rank) = 0;
  //@}

  /*!
   * \brief Creates a non-blocking message to receive serialized data from rank \a rank.
   *
   * The message is processed only when processMessages() is called.
   */
  virtual ISerializeMessage* createReceiveSerializer(Int32 rank) = 0;

  /*!
   * \brief Executes the operations of messages \a messages
   */
  virtual void processMessages(ConstArrayView<ISerializeMessage*> messages) = 0;

  /*!
   * \brief Executes the operations of messages \a messages
   */
  virtual void processMessages(ConstArrayView<Ref<ISerializeMessage>> messages) = 0;

  /*!
   * \brief Frees the requests.
   */
  virtual void freeRequests(ArrayView<Parallel::Request> requests) = 0;

  /*! @name non-blocking message sending operations
   *
   * \brief Sends an array of values to rank \a rank.
   *
   * Sends the values of array \a values to the instance of rank \a rank.
   * The recipient must perform a corresponding reception (whose
   * rank must be that of this handler and the type and the
   * size of the array must correspond) with the function recvValues().
   * The send is blocking if \a is_blocking is true, non-blocking if it is false.
   * In the latter case, the returned request must be used in waitAllRequests()
   * or freed by freeRequests().
   */
  //@{
  virtual Request send(ConstArrayView<char> values, Int32 rank, bool is_blocking) = 0;
  virtual Request send(ConstArrayView<signed char> values, Int32 rank, bool is_blocking) = 0;
  virtual Request send(ConstArrayView<unsigned char> values, Int32 rank, bool is_blocking) = 0;
  virtual Request send(ConstArrayView<short> values, Int32 rank, bool is_blocking) = 0;
  virtual Request send(ConstArrayView<unsigned short> values, Int32 rank, bool is_blocking) = 0;
  virtual Request send(ConstArrayView<int> values, Int32 rank, bool is_blocking) = 0;
  virtual Request send(ConstArrayView<unsigned int> values, Int32 rank, bool is_blocking) = 0;
  virtual Request send(ConstArrayView<long> values, Int32 rank, bool is_blocking) = 0;
  virtual Request send(ConstArrayView<unsigned long> values, Int32 rank, bool is_blocking) = 0;
  virtual Request send(ConstArrayView<long long> values, Int32 rank, bool is_blocking) = 0;
  virtual Request send(ConstArrayView<unsigned long long> values, Int32 rank, bool is_blocking) = 0;
  virtual Request send(ConstArrayView<float> values, Int32 rank, bool is_blocking) = 0;
  virtual Request send(ConstArrayView<double> values, Int32 rank, bool is_blocking) = 0;
  virtual Request send(ConstArrayView<long double> values, Int32 rank, bool is_blocking) = 0;
  virtual Request send(ConstArrayView<APReal> values, Int32 rank, bool is_blocking) = 0;
  virtual Request send(ConstArrayView<Real2> values, Int32 rank, bool is_blocking) = 0;
  virtual Request send(ConstArrayView<Real3> values, Int32 rank, bool is_blocking) = 0;
  virtual Request send(ConstArrayView<Real2x2> values, Int32 rank, bool is_blocking) = 0;
  virtual Request send(ConstArrayView<Real3x3> values, Int32 rank, bool is_blocking) = 0;
  virtual Request send(ConstArrayView<HPReal> values, Int32 rank, bool is_blocking) = 0;
  //@}

  //! @name non-blocking message receiving operations.
  //@{
  /*! Receives array \a values from subdomain \a rank */
  virtual Request recv(ArrayView<char> values, Int32 rank, bool is_blocking) = 0;
  virtual Request recv(ArrayView<signed char> values, Int32 rank, bool is_blocking) = 0;
  virtual Request recv(ArrayView<unsigned char> values, Int32 rank, bool is_blocking) = 0;
  virtual Request recv(ArrayView<short> values, Int32 rank, bool is_blocking) = 0;
  virtual Request recv(ArrayView<unsigned short> values, Int32 rank, bool is_blocking) = 0;
  virtual Request recv(ArrayView<int> values, Int32 rank, bool is_blocking) = 0;
  virtual Request recv(ArrayView<unsigned int> values, Int32 rank, bool is_blocking) = 0;
  virtual Request recv(ArrayView<long> values, Int32 rank, bool is_blocking) = 0;
  virtual Request recv(ArrayView<unsigned long> values, Int32 rank, bool is_blocking) = 0;
  virtual Request recv(ArrayView<long long> values, Int32 rank, bool is_blocking) = 0;
  virtual Request recv(ArrayView<unsigned long long> values, Int32 rank, bool is_blocking) = 0;
  virtual Request recv(ArrayView<float> values, Int32 rank, bool is_blocking) = 0;
  virtual Request recv(ArrayView<double> values, Int32 rank, bool is_blocking) = 0;
  virtual Request recv(ArrayView<long double> values, Int32 rank, bool is_blocking) = 0;
  virtual Request recv(ArrayView<APReal> values, Int32 rank, bool is_blocking) = 0;
  virtual Request recv(ArrayView<Real2> values, Int32 rank, bool is_blocking) = 0;
  virtual Request recv(ArrayView<Real3> values, Int32 rank, bool is_blocking) = 0;
  virtual Request recv(ArrayView<Real2x2> values, Int32 rank, bool is_blocking) = 0;
  virtual Request recv(ArrayView<Real3x3> values, Int32 rank, bool is_blocking) = 0;
  virtual Request recv(ArrayView<HPReal> values, Int32 rank, bool is_blocking) = 0;
  //@}

  //! @name generic message receiving operations
  //@{
  /*! Receives message \a message, array \a values */
  virtual Request receive(Span<char> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<signed char> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<unsigned char> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<short> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<unsigned short> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<int> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<unsigned int> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<long> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<unsigned long> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<long long> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<unsigned long long> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<float> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<double> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<long double> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<APReal> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<Real2> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<Real3> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<Real2x2> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<Real3x3> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<HPReal> values, const PointToPointMessageInfo& message) = 0;
  virtual Request receiveSerializer(ISerializer* values, const PointToPointMessageInfo& message) = 0;
  //@}

  //! @name generic message sending operations
  //@{
  /*! Sends message \a message with the values of array \a values */
  virtual Request send(Span<const char> values, const PointToPointMessageInfo& message) = 0;
  virtual Request send(Span<const signed char> values, const PointToPointMessageInfo& message) = 0;
  virtual Request send(Span<const unsigned char> values, const PointToPointMessageInfo& message) = 0;
  virtual Request send(Span<const short> values, const PointToPointMessageInfo& message) = 0;
  virtual Request send(Span<const unsigned short> values, const PointToPointMessageInfo& message) = 0;
  virtual Request send(Span<const int> values, const PointToPointMessageInfo& message) = 0;
  virtual Request send(Span<const unsigned int> values, const PointToPointMessageInfo& message) = 0;
  virtual Request send(Span<const long> values, const PointToPointMessageInfo& message) = 0;
  virtual Request send(Span<const unsigned long> values, const PointToPointMessageInfo& message) = 0;
  virtual Request send(Span<const long long> values, const PointToPointMessageInfo& message) = 0;
  virtual Request send(Span<const unsigned long long> values, const PointToPointMessageInfo& message) = 0;
  virtual Request send(Span<const float> values, const PointToPointMessageInfo& message) = 0;
  virtual Request send(Span<const double> values, const PointToPointMessageInfo& message) = 0;
  virtual Request send(Span<const long double> values, const PointToPointMessageInfo& message) = 0;
  virtual Request send(Span<const APReal> values, const PointToPointMessageInfo& message) = 0;
  virtual Request send(Span<const Real2> values, const PointToPointMessageInfo& message) = 0;
  virtual Request send(Span<const Real3> values, const PointToPointMessageInfo& message) = 0;
  virtual Request send(Span<const Real2x2> values, const PointToPointMessageInfo& message) = 0;
  virtual Request send(Span<const Real3x3> values, const PointToPointMessageInfo& message) = 0;
  virtual Request send(Span<const HPReal> values, const PointToPointMessageInfo& message) = 0;
  virtual Request sendSerializer(const ISerializer* values, const PointToPointMessageInfo& message) = 0;
  //@}

  /*!
   * \brief Probes if messages are available.
   *
   * \sa Arccore::MessagePassing::mpProbe().
   */
  virtual MessageId probe(const PointToPointMessageInfo& message) = 0;

  /*!
   * \brief Probes if messages are available.
   *
   * \sa Arccore::MessagePassing::mpLegacyProbe().
   */
  virtual MessageSourceInfo legacyProbe(const PointToPointMessageInfo& message) = 0;

  virtual void sendRecv(ConstArrayView<char> send_buf,
                        ArrayView<char> recv_buf, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<signed char> send_buf,
                        ArrayView<signed char> recv_buf, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<unsigned char> send_buf,
                        ArrayView<unsigned char> recv_buf, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<short> send_buf,
                        ArrayView<short> recv_buf, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<unsigned short> send_buf,
                        ArrayView<unsigned short> recv_buf, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<int> send_buf,
                        ArrayView<int> recv_buf, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<unsigned int> send_buf,
                        ArrayView<unsigned int> recv_buf, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<long> send_buf,
                        ArrayView<long> recv_buf, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<unsigned long> send_buf,
                        ArrayView<unsigned long> recv_buf, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<long long> send_buf,
                        ArrayView<long long> recv_buf, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<unsigned long long> send_buf,
                        ArrayView<unsigned long long> recv_buf, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<float> send_buf,
                        ArrayView<float> recv_buf, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<double> send_buf,
                        ArrayView<double> recv_buf, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<long double> send_buf,
                        ArrayView<long double> recv_buf, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<APReal> send_buf,
                        ArrayView<APReal> recv_buf, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<Real2> send_buf,
                        ArrayView<Real2> recv_buf, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<Real3> send_buf,
                        ArrayView<Real3> recv_buf, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<Real2x2> send_buf,
                        ArrayView<Real2x2> recv_buf, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<Real3x3> send_buf,
                        ArrayView<Real3x3> recv_buf, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<HPReal> send_buf,
                        ArrayView<HPReal> recv_buf, Int32 rank) = 0;

  virtual void allToAll(ConstArrayView<char> send_buf, ArrayView<char> recv_buf,
                        Integer count) = 0;
  virtual void allToAll(ConstArrayView<signed char> send_buf, ArrayView<signed char> recv_buf,
                        Integer count) = 0;
  virtual void allToAll(ConstArrayView<unsigned char> send_buf, ArrayView<unsigned char> recv_buf,
                        Integer count) = 0;
  virtual void allToAll(ConstArrayView<short> send_buf, ArrayView<short> recv_buf, Integer count) = 0;
  virtual void allToAll(ConstArrayView<unsigned short> send_buf, ArrayView<unsigned short> recv_buf,
                        Integer count) = 0;
  virtual void allToAll(ConstArrayView<int> send_buf, ArrayView<int> recv_buf, Integer count) = 0;
  virtual void allToAll(ConstArrayView<unsigned int> send_buf, ArrayView<unsigned int> recv_buf,
                        Integer count) = 0;
  virtual void allToAll(ConstArrayView<long> send_buf, ArrayView<long> recv_buf, Integer count) = 0;
  virtual void allToAll(ConstArrayView<unsigned long> send_buf, ArrayView<unsigned long> recv_buf,
                        Integer count) = 0;
  virtual void allToAll(ConstArrayView<long long> send_buf, ArrayView<long long> recv_buf,
                        Integer count) = 0;
  virtual void allToAll(ConstArrayView<unsigned long long> send_buf,
                        ArrayView<unsigned long long> recv_buf, Integer count) = 0;
  virtual void allToAll(ConstArrayView<float> send_buf, ArrayView<float> recv_buf,
                        Integer count) = 0;
  virtual void allToAll(ConstArrayView<double> send_buf, ArrayView<double> recv_buf,
                        Integer count) = 0;
  virtual void allToAll(ConstArrayView<long double> send_buf, ArrayView<long double> recv_buf,
                        Integer count) = 0;
  virtual void allToAll(ConstArrayView<APReal> send_buf, ArrayView<APReal> recv_buf,
                        Integer count) = 0;
  virtual void allToAll(ConstArrayView<Real2> send_buf, ArrayView<Real2> recv_buf,
                        Integer count) = 0;
  virtual void allToAll(ConstArrayView<Real3> send_buf, ArrayView<Real3> recv_buf,
                        Integer count) = 0;
  virtual void allToAll(ConstArrayView<Real2x2> send_buf, ArrayView<Real2x2> recv_buf,
                        Integer count) = 0;
  virtual void allToAll(ConstArrayView<Real3x3> send_buf, ArrayView<Real3x3> recv_buf,
                        Integer count) = 0;
  virtual void allToAll(ConstArrayView<HPReal> send_buf, ArrayView<HPReal> recv_buf,
                        Integer count) = 0;

  /*! @name allToAll variable
   *
   * \brief Performs a variable allToAll operation
   */
  //@{
  virtual void allToAllVariable(ConstArrayView<char> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<char> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual void allToAllVariable(ConstArrayView<signed char> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<signed char> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual void allToAllVariable(ConstArrayView<unsigned char> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<unsigned char> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual void allToAllVariable(ConstArrayView<short> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<short> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual void allToAllVariable(ConstArrayView<unsigned short> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<unsigned short> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual void allToAllVariable(ConstArrayView<int> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<int> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual void allToAllVariable(ConstArrayView<unsigned int> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<unsigned int> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual void allToAllVariable(ConstArrayView<long> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<long> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual void allToAllVariable(ConstArrayView<unsigned long> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<unsigned long> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual void allToAllVariable(ConstArrayView<long long> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<long long> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual void allToAllVariable(ConstArrayView<unsigned long long> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<unsigned long long> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual void allToAllVariable(ConstArrayView<float> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<float> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual void allToAllVariable(ConstArrayView<double> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<double> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual void allToAllVariable(ConstArrayView<long double> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<long double> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual void allToAllVariable(ConstArrayView<APReal> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<APReal> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual void allToAllVariable(ConstArrayView<Real2> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<Real2> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual void allToAllVariable(ConstArrayView<Real3> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<Real3> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual void allToAllVariable(ConstArrayView<Real2x2> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<Real2x2> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual void allToAllVariable(ConstArrayView<Real3x3> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<Real3x3> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual void allToAllVariable(ConstArrayView<HPReal> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<HPReal> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  //@}

  /*! @name scan
   *
   * \brief Performs an algorithm equivalent to MPI_Scan in semantics
   */
  //@{
  //! Applies a prefix-sum algorithm on the values of \a v using the \a rt operation.
  virtual void scan(eReduceType rt, ArrayView<char> v) = 0;
  virtual void scan(eReduceType rt, ArrayView<signed char> v) = 0;
  virtual void scan(eReduceType rt, ArrayView<unsigned char> v) = 0;
  virtual void scan(eReduceType rt, ArrayView<short> v) = 0;
  virtual void scan(eReduceType rt, ArrayView<unsigned short> v) = 0;
  virtual void scan(eReduceType rt, ArrayView<int> v) = 0;
  virtual void scan(eReduceType rt, ArrayView<unsigned int> v) = 0;
  virtual void scan(eReduceType rt, ArrayView<long> v) = 0;
  virtual void scan(eReduceType rt, ArrayView<unsigned long> v) = 0;
  virtual void scan(eReduceType rt, ArrayView<long long> v) = 0;
  virtual void scan(eReduceType rt, ArrayView<unsigned long long> v) = 0;
  virtual void scan(eReduceType rt, ArrayView<float> v) = 0;
  virtual void scan(eReduceType rt, ArrayView<double> v) = 0;
  virtual void scan(eReduceType rt, ArrayView<long double> v) = 0;
  virtual void scan(eReduceType rt, ArrayView<APReal> v) = 0;
  virtual void scan(eReduceType rt, ArrayView<Real2> v) = 0;
  virtual void scan(eReduceType rt, ArrayView<Real3> v) = 0;
  virtual void scan(eReduceType rt, ArrayView<Real2x2> v) = 0;
  virtual void scan(eReduceType rt, ArrayView<Real3x3> v) = 0;
  virtual void scan(eReduceType rt, ArrayView<HPReal> v) = 0;
  //@}

  /*!
   * \brief Creates a list to manage 'ISerializeMessage'.
   *
   * \deprecated Use createSerializeMessageListRef() instead.
   */
  ARCCORE_DEPRECATED_2020("Use createSerializeMessageListRef() instead")
  virtual ISerializeMessageList* createSerializeMessageList() = 0;

  //! Creates a list to manage 'ISerializeMessage'
  virtual Ref<ISerializeMessageList> createSerializeMessageListRef() = 0;

  //! @name synchronization operations and asynchronous operations
  //@{
  //! Performs a barrier
  virtual void barrier() = 0;

  //! Blocks while waiting for the \a rvalues requests to complete
  virtual void waitAllRequests(ArrayView<Request> rvalues) = 0;

  /*!
  * \brief Blocks while waiting for one of the \a rvalues requests to complete.
  *
  * Returns an array of indices of completed requests.
  */
  virtual UniqueArray<Integer> waitSomeRequests(ArrayView<Request> rvalues) = 0;

  /*!
  * \brief Tests if one of the \a rvalues requests is complete.
  *
  * Returns an array of indices of completed requests.
  */
  virtual UniqueArray<Integer> testSomeRequests(ArrayView<Request> rvalues) = 0;

  //@}

  //! @name various operations
  //@{

  /*!
   * \brief Returns a sequential parallelism manager.
   *
   * This instance retains ownership of the returned instance, which must not
   * be destroyed. The lifetime of the returned instance is
   * the same as this instance.
   */
  virtual IParallelMng* sequentialParallelMng() = 0;
  virtual Ref<IParallelMng> sequentialParallelMngRef() = 0;

  //@}

  /*!
   * \brief Returns an operation to retrieve the values of a variable
   * on the entities of another subdomain.
   *
   * The returned instance must be destroyed by the delete operator.
   */
  [[deprecated("Y2021: Use Arcane::ParallelMngUtils;:createGetVariablesValuesOperationRef() instead")]]
  virtual IGetVariablesValuesParallelOperation* createGetVariablesValuesOperation() = 0;

  /*!
   * \brief Returns an operation to transfer values
   * between subdomains.
   *
   * The returned instance must be destroyed by the delete operator.
   */
  [[deprecated("Y2021: Use Arcane::ParallelMngUtils;:createTransferValuesOperationRef() instead")]]
  virtual ITransferValuesParallelOperation* createTransferValuesOperation() = 0;

  /*!
   * \brief Returns an interface for transferring messages
   * between processors.
   *
   * The returned instance must be destroyed by the delete operator.
   */
  [[deprecated("Y2021: Use Arcane::ParallelMngUtils;:createExchangerRef() instead")]]
  virtual IParallelExchanger* createExchanger() = 0;

  /*!
   * \brief Returns an interface for synchronizing
   * variables on the group of the \a family
   *
   * The returned instance must be destroyed by the delete operator.
   */
  [[deprecated("Y2021: Use Arcane::ParallelMngUtils;:createSynchronizerRef() instead")]]
  virtual IVariableSynchronizer* createSynchronizer(IItemFamily* family) = 0;

  /*!
   * \brief Returns an interface for synchronizing
   * variables on the \a group.
   *
   * The returned instance must be destroyed by the delete operator.
   */
  [[deprecated("Y2021: Use Arcane::ParallelMngUtils;:createSynchronizerRef() instead")]]
  virtual IVariableSynchronizer* createSynchronizer(const ItemGroup& group) = 0;

  /*!
   * \brief Creates an instance containing information about the rank topology of this manager.
   *
   * This operation is collective.
   *
   * The returned instance must be destroyed by the delete operator.
   */
  [[deprecated("Y2021: Use Arcane::ParallelMngUtils;:createTopologyRef() instead")]]
  virtual IParallelTopology* createTopology() = 0;

  /*!
   * \brief Replication information.
   *
   * The returned pointer is never null and remains the property of this
   * instance.
   */
  virtual IParallelReplication* replication() const = 0;

  /*!
   * \internal
   * \brief Sets the Replication Information.
   *
   * This method is internal to Arcane and should only be called during initialization.
   */
  virtual void setReplication(IParallelReplication* v) = 0;

  /*!
   * \brief Creates a new parallelism manager for a subset
   * of ranks.
   *
   * \deprecated Use createSubParallelMngRef() instead
   */
  ARCCORE_DEPRECATED_2020("Use createSubParallelMngRef() instead")
  virtual IParallelMng* createSubParallelMng(Int32ConstArrayView kept_ranks) = 0;

  /*!
   * \brief Creates a new parallelism manager for a subset
   * of ranks.
   *
   * This operation is collective.
   *
   * This operation allows creating a new manager containing
   * only the \a kept_ranks of this manager.
   *
   * If the rank calling this operation is not in \a kept_ranks,
   * it returns 0.
   *
   * The returned instance must be destroyed by the delete operator.
   */
  virtual Ref<IParallelMng> createSubParallelMngRef(Int32ConstArrayView kept_ranks) = 0;

  /*!
   * \brief Creates a request list for this manager.
   */
  virtual Ref<Parallel::IRequestList> createRequestListRef() = 0;

  //! Statistics manager
  virtual IStat* stat() = 0;

  //! Prints statistics related to this parallelism manager
  virtual void printStats() = 0;

  //! Interface for non-blocking collective operations.
  virtual IParallelNonBlockingCollective* nonBlockingCollective() const = 0;

  //! Associated %Arccore message passing manager
  virtual IMessagePassingMng* messagePassingMng() const = 0;

 public:

  //! Internal Arcane API
  virtual IParallelMngInternal* _internalApi() = 0;

 private:

  /*!
   * \internal
   * \brief Factory for utility functions.
   */
  virtual Ref<IParallelMngUtilsFactory> _internalUtilsFactory() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface for an 'IParallelMng' container.
 *
 * An IParallelMng container manages a set
 * of IParallelMng instances from the same communicator in shared memory mode.
 *
 * \note Do not use outside of Arcane. Unstabilized API.
 */
class ARCANE_CORE_EXPORT IParallelMngContainer
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 protected:

  virtual ~IParallelMngContainer() = default;

 public:

  //! Creates the IParallelMng for the local rank \a local_rank
  virtual Ref<IParallelMng> _createParallelMng(Int32 local_rank, ITraceMng* tm) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface for an 'IParallelMng' container factory.
 * \note Do not use outside of Arcane. Unstabilized API.
 */
class ARCANE_CORE_EXPORT IParallelMngContainerFactory
{
 public:

  virtual ~IParallelMngContainerFactory() = default;

 public:

  /*!
   * \brief Creates a container for \a nb_local_rank local ranks and
   * with \a communicator as the communicator.
   *
   * The MPI communicator \a communicator can be null in sequential or
   * shared memory mode. The number of local ranks is 1 in sequential or pure MPI mode.
   *
   * The second communicator \a machine_communicator is only useful in hybrid mode.
   * In other modes, it can be null.
   */
  virtual Ref<IParallelMngContainer>
  _createParallelMngBuilder(Int32 nb_local_rank, Parallel::Communicator communicator,
                            Parallel::Communicator machine_communicator) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
