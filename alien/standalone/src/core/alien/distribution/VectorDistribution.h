// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

/*!
 * \file VectorDistribution.h
 * \brief VectorDistribution.h
 */

#pragma once

#include <arccore/message_passing/MessagePassingGlobal.h>
#include "alien/data/ISpace.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup distribution
 * \brief Computes a vector distribution
 *
 * Computes or use a pre-existing block row distribution for vectors
 */
class ALIEN_EXPORT VectorDistribution
{
 public:
  //! Constructor
  VectorDistribution();

  /*!
   * \brief Constructor
   * \param[in] space The space of the vector
   * \param[in] parallel_mng The parallel manager
   */
  VectorDistribution(
  const ISpace& space, Arccore::MessagePassing::IMessagePassingMng* parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] space The space of the vector
   * \param[in] parallel_mng The parallel manager
   */
  VectorDistribution(const ISpace& space,
                     std::shared_ptr<Arccore::MessagePassing::IMessagePassingMng> parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] global_size The global size of the vector
   * \param[in] parallel_mng The parallel manager
   */
  VectorDistribution(Arccore::Integer global_size,
                     Arccore::MessagePassing::IMessagePassingMng* parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] global_size The global size of the vector
   * \param[in] parallel_mng The parallel manager
   */
  // FIXME: not implemented !
  VectorDistribution(Arccore::Integer global_size,
                     std::shared_ptr<Arccore::MessagePassing::IMessagePassingMng> parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] space The space of the vector
   * \param[in] local_size The local size of the vector
   * \param[in] parallel_mng The parallel manager
   */
  VectorDistribution(const ISpace& space, Arccore::Integer local_size,
                     Arccore::MessagePassing::IMessagePassingMng* parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] space The space of the vector
   * \param[in] local_size The local size of the vector
   * \param[in] parallel_mng The parallel manager
   */
  VectorDistribution(const ISpace& space, Arccore::Integer local_size,
                     std::shared_ptr<Arccore::MessagePassing::IMessagePassingMng> parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] global_size The global size of the vector
   * \param[in] local_size The local size of the vector
   * \param[in] parallel_mng The parallel manager
   */
  VectorDistribution(Arccore::Integer global_size, Arccore::Integer local_size,
                     Arccore::MessagePassing::IMessagePassingMng* parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] global_size The global size of the vector
   * \param[in] local_size The local size of the vector
   * \param[in] parallel_mng The parallel manager
   */
  VectorDistribution(Arccore::Integer global_size, Arccore::Integer local_size,
                     std::shared_ptr<Arccore::MessagePassing::IMessagePassingMng> parallel_mng);

  /*!
   * \brief Copy constructor
   * \parm[in] dist The distribution to copy
   */
  VectorDistribution(const VectorDistribution& dist);

  /*!
   * \brief Rvalue constructor
   * \param[in] dist The distribution to take
   */
  VectorDistribution(VectorDistribution&& dist);

  //! Free resources
  ~VectorDistribution();

  /*!
   * \brief Equal operator
   * \param[in] dist The distribution to copy
   * \returns The copied distribution
   */
  VectorDistribution& operator=(const VectorDistribution& dist);

  /*!
   * \brief Equal operator
   * \param[in] dist The distribution to copy
   * \returns The copied distribution
   */
  VectorDistribution& operator=(VectorDistribution&& dist);

  /*!
   * \brief Comparison operator
   * \param[in] dist The distribution to compare
   * \returns Whether or not the distribution are identical
   */
  bool operator==(const VectorDistribution& dist) const;

  /*!
   * \brief Whether or not the run is parallel
   * \returns Whether or not the run is parallel
   */
  bool isParallel() const;

  /*!
   * \brief Get the parallel manager
   * \returns The parallel manager
   */
  std::shared_ptr<Arccore::MessagePassing::IMessagePassingMng> sharedParallelMng() const;

  /*!
   * \brief Get the parallel manager
   * \returns The parallel manager
   */
  Arccore::MessagePassing::IMessagePassingMng* parallelMng() const;

  /*!
   * \brief Get the space
   * \returns The vector space
   */
  const ISpace& space() const;

  /*!
   * \brief Get the local size
   * \returns The local size
   */
  Arccore::Integer localSize() const;

  /*!
   * \brief Get the global size
   * \returns The global size
   */
  Arccore::Integer globalSize() const;

  /*!
   * \brief Get the offset
   * \return The offset
   */
  Arccore::Integer offset() const;

  /*!
   * \brief Get the offset for a specific proc
   * \param[in] p The requested proc
   * \returns The offset for the specific proc
   */
  Arccore::Integer offset(Arccore::Integer p) const;

  /*!
   * \brief Get all the offsets
   * \returns Offsets array
   */

  Arccore::ConstArrayView<Integer> offsets() const;

  /*!
   * \brief Get the owner of an entry
   * \param[in] i The global id of the element
   * \returns The proc owner
   */
  Arccore::Integer owner(Arccore::Integer i) const;

  /*!
   * \brief Clone the distribution
   * \returns A clone of the distribution
   */
  std::shared_ptr<VectorDistribution> clone() const;

  /*!
   * \brief Get the local id of an elements with its global id
   * \param[in] i The global id of the element
   * \returns The local id of the element
   */
  Arccore::Integer globalToLocal(Arccore::Integer i) const;

 private:
  /*!
   * \brief Get the local id of a non local element with its global id
   * \param[in] i The global id of the non local element
   * \param[in] p The proc which owns the element
   * \returns The local id of the element
   */
  Arccore::Integer globalToLocal(Arccore::Integer i, Arccore::Integer p) const;

  /*!
   * \brief Get the global id of an elements with its local id
   * \param[in] i The local id of the element
   * \returns The global id of the element
   */
  Arccore::Integer localToGlobal(Arccore::Integer i) const;

  /*!
   * \brief Get the global id of a non local element with its local id
   * \param[in] i The local id of the non local element
   * \param[in] p The proc which owns the element
   * \returns The global id of the element
   */
  Arccore::Integer localToGlobal(Arccore::Integer i, Arccore::Integer p) const;

  // due to 'private' of VectorDistribution methods
  // will be removed
  friend class MatrixDistribution;

 private:
  struct Internal;
  //! Internal implementation of the vector distribution
  std::shared_ptr<Internal> m_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Print the distribution
extern ALIEN_EXPORT std::ostream& operator<<(
std::ostream& nout, const VectorDistribution& dist);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
