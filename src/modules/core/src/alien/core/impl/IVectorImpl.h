/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file IVectorImpl.h
 * \brief IVectorImpl.h
 */

#pragma once

#include <alien/core/block/Block.h>
#include <alien/core/block/VBlockSizes.h>
#include <alien/distribution/VectorDistribution.h>
#include <alien/utils/ObjectWithTrace.h>
#include <alien/utils/Precomp.h>
#include <alien/utils/time_stamp/Timestamp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

typedef Arccore::String BackEndId;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MultiVectorImpl;
class ISpace;
class Block;
class VBlock;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup impl
 * \brief Interface to handle abstract vectors implementation
 */
class ALIEN_EXPORT IVectorImpl : public Timestamp
, public ObjectWithTrace
{
 public:
  //! Type of the values stored in the vectors
  typedef Arccore::Real ValueType;

 public:
  /*!
   * \brief Constructor
   * \param[in] multi_impl Pointer to the multivector handler, null if not associated
   * \param[in] backend Name of the underneath backend, empty string if not associated
   */
  explicit IVectorImpl(const MultiVectorImpl* multi_impl, BackEndId backend = "");

  //! Free resources
  virtual ~IVectorImpl() override
  {
    delete m_vblock_sizes;
    m_vblock_sizes = nullptr;
  }

  IVectorImpl(const IVectorImpl& src) = delete;
  IVectorImpl(IVectorImpl&& src) = delete;
  IVectorImpl& operator=(const IVectorImpl& src) = delete;
  IVectorImpl& operator=(IVectorImpl&& src) = delete;

 public:
  //! Wipe out internal data.
  virtual void clear() {}

  /*!
   * \brief Initialize vector datas
   * \param[in] dist The vector distribution
   * \param[in] do_alloc Allocate memory or not
   *
   * \todo Fix this method : could be removed during the process of solver refactoring
   */
  virtual void init(ALIEN_UNUSED_PARAM const VectorDistribution& dist, ALIEN_UNUSED_PARAM bool do_alloc) {}

  /*!
   * \brief Get the vector space
   * \returns The space associated to the vector
   */
  virtual const ISpace& space() const;

  /*!
   * Get the vector backend id
   * \returns The backend if associated to the vector
   */
  virtual BackEndId backend() const { return m_backend; }

  /*!
   * \brief Get the distribution of the vector
   * \returns The vector distribution
   */
  virtual const VectorDistribution& distribution() const;

  /*!
   * \brief Get block datas of the vector
   *
   * Get the block datas of the vector. This method should be used only with uniform
   * blocks vectors, otherwise il will return a nullptr
   *
   * \returns Block data for block vector, nullptr for scalar and variable block vector
   */
  virtual const Block* block() const;

  /*!
   * \brief Get block datas of the vector
   *
   * Get the block datas of the vector. This method should be used only with variable
   * blocks vectors, otherwise il will return a nullptr
   *
   * \returns Block data for variable block vector, nullptr for scalar and block vector
   */
  virtual const VBlock* vblock() const;

  /*!
   * \brief Get the "scalarized" local size
   * \see VBlockSizes for "scalarized" definition
   * \returns The actual local size
   */
  virtual Arccore::Integer scalarizedLocalSize() const;

  /*!
   * \brief Get the "scalarized" global size
   * \see VBlockSizes for "scalarized" definition
   * \returns The actual global size
   */
  virtual Arccore::Integer scalarizedGlobalSize() const;

  /*!
   * \brief Get the "scalarized" offset
   * \see VBlockSizes for "scalarized" definition
   * \returns The actual offset
   */
  virtual Arccore::Integer scalarizedOffset() const;

 protected:
  //! Pointer on vectors implementations
  const MultiVectorImpl* m_multi_impl;
  //! Backend id
  BackEndId m_backend;
  //! Variable blocks size data
  mutable VBlockSizes* m_vblock_sizes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
