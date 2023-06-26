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
 * \file MultiVectorImpl.h
 * \brief MultiVectorImpl.h
 */

#pragma once

#include <iostream>
#include <map>

#include <alien/core/backend/BackEnd.h>
#include <alien/core/backend/VectorConverterRegisterer.h>
#include <alien/core/block/Block.h>
#include <alien/core/block/VBlock.h>
#include <alien/core/impl/IVectorImpl.h>
#include <alien/data/Space.h>
#include <alien/distribution/VectorDistribution.h>
#include <alien/utils/Precomp.h>
#include <alien/utils/UserFeatureMng.h>
#include <alien/utils/time_stamp/TimestampMng.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_EXPORT MultiVectorImpl : public TimestampMng
, public UserFeatureMng
, public ObjectWithTrace
{
  /*!
   * \ingroup impl
   * \brief Multi vectors representation container
   *
   * This class allows to store and access several implementations of the same vector.
   * It also stores all shared information between those implementation such as
   * distribution, or block information. It provided two accessors, one to access the
   * vector in read-only, that do not increase the timestamp and one in read-write mode,
   * that can increase the timestamp. The former implementation might therefore become the
   * up to date vector.
   *
   * While requesting a vector in a specific format, if the up to date vector is not in
   * that specific format, the up to vector matrix will be converted, provided a converter
   * exists.
   */
 public:
  //! Default constructor
  MultiVectorImpl();

  /*!
   * \brief Constructor
   * \param[in] space The space of the vector
   * \param[in] dist The distribution of the vector
   */
  MultiVectorImpl(
  std::shared_ptr<ISpace> space, std::shared_ptr<VectorDistribution> dist);

  //! Free resources
  ~MultiVectorImpl() override;

 protected:
  /*!
   * \brief Copy constructor
   * \param[in] impl The MultiVectorImpl to copy
   */
  MultiVectorImpl(const MultiVectorImpl& impl);

 public:
  MultiVectorImpl(MultiVectorImpl&& impl) = delete;
  void operator=(const MultiVectorImpl&) = delete;
  void operator=(MultiVectorImpl&&) = delete;

 public:
  /*!
   * \brief Set uniform block information
   * \param[in] blocks The block data
   */
  void setBlockInfos(const Block* blocks);

  /*!
   * \brief Set variable block information
   * \param[in] blocks The block data
   */
  void setBlockInfos(const VBlock* blocks);

  /*!
   * \brief Set uniform block information
   * \param[in] block_size The size of the blocks
   */
  void setBlockInfos(Arccore::Integer block_size);

  //! Free resources
  void free();

  //! Clear resources
  void clear();

  /*!
   * \brief Get the space associated with the vector
   * \returns The space
   */
  const ISpace& space() const { return *m_space.get(); }

  /*!
   * \brief Get the vector distribution
   * \returns The vector distribution
   */
  const VectorDistribution& distribution() const { return *m_distribution.get(); }

  /*!
   * \brief Get uniform block datas
   * \returns The vector block data, or a nullptr if the vector is scalar or has variable
   * blocks
   */
  const Block* block() const;

  /*!
   * \brief Get variable block datas
   * \returns The vector block data or a nullptr if the vector is scalar or has uniform
   * blocks
   */
  const VBlock* vblock() const;

 public:
  /*!
   * \brief Clone this object
   * \return A clone of this object
   */
  MultiVectorImpl* clone() const;

  /*!
   * \brief Get a specific vector implementation
   *
   * Might induce a conversion, depending on the up to date and requested vector
   * implementation
   *
   * \returns The up to date vector in the requested implementation
   */
  template <typename tag>
  const typename AlgebraTraits<tag>::vector_type& get() const;

  /*!
   * \brief Get a specific vector implementation
   *
   * Might induce a conversion, depending on the up to date and requested vector
   * implementation
   *
   * \param[in] update_stamp Whether or not the timestamp should be increased or not
   * \returns The up to date vector in the requested implementation
   */
  template <typename tag>
  typename AlgebraTraits<tag>::vector_type& get(bool update_stamp);

  //! Release a vector implementation
  template <typename tag>
  void release() const;

 private:
  /*!
   * \brief Get a specific vector implementation
   * \param[in] backend The id of the specific implementation
   * \returns The vector in the requested format
   */
  template <typename vector_type>
  IVectorImpl*& getImpl(BackEndId backend) const;

  /*!
   * \brief Update a vector implementation
   * \param[in] target The targeted implementation
   */
  void updateImpl(IVectorImpl* target) const;

 protected:
  /*!
   * \brief Insert a vector implementation in the multi vector container
   * \param[in] backend The implementation backend id
   * \param[in] v The vector to insert
   */
  template <typename vector_type>
  void insert(const BackEndId& backend, vector_type* v);

 private:
  //! The vector space
  std::shared_ptr<ISpace> m_space;
  //! The vector distribution
  std::shared_ptr<VectorDistribution> m_distribution;
  //! The type of the vector container
  typedef std::map<BackEndId, IVectorImpl*> MultiVectorImplMap;
  //! The vectors container
  mutable MultiVectorImplMap m_impls2;
  //! The uniform block datas
  std::shared_ptr<Block> m_block;
  //! The variable block datas
  std::shared_ptr<VBlock> m_variable_block;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename tag>
const typename AlgebraTraits<tag>::vector_type&
MultiVectorImpl::get() const
{
  typedef typename AlgebraTraits<tag>::vector_type vector_type;
  IVectorImpl*& impl2 = getImpl<vector_type>(AlgebraTraits<tag>::name());
  ALIEN_ASSERT(
  (impl2->backend() == AlgebraTraits<tag>::name()), ("Inconsistent backend"));
  updateImpl(impl2);
  return *dynamic_cast<vector_type*>(impl2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename tag>
typename AlgebraTraits<tag>::vector_type&
MultiVectorImpl::get(const bool update_stamp)
{
  typedef typename AlgebraTraits<tag>::vector_type vector_type;
  IVectorImpl*& impl2 = getImpl<vector_type>(AlgebraTraits<tag>::name());
  ALIEN_ASSERT(
  (impl2->backend() == AlgebraTraits<tag>::name()), ("Inconsistent backend"));
  updateImpl(impl2);
  if (update_stamp) {
    impl2->updateTimestamp();
  }
  return *dynamic_cast<vector_type*>(impl2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename tag>
void MultiVectorImpl::release() const
{
  auto finder = m_impls2.find(AlgebraTraits<tag>::name());
  if (finder == m_impls2.end())
    return; // already freed
  delete finder->second, finder->second = NULL;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename vector_type>
IVectorImpl*&
MultiVectorImpl::getImpl(BackEndId backend) const
{
  auto inserter = m_impls2.insert(MultiVectorImplMap::value_type(backend, NULL));
  IVectorImpl*& impl2 = inserter.first->second;
  if (impl2 == NULL) {
    auto new_impl = new vector_type(this); // constructeur associ� � un multi-impl
    new_impl->init(*m_distribution.get(), true);
    impl2 = new_impl;
  }
  return impl2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename vector_type>
void MultiVectorImpl::insert(const BackEndId& backend, vector_type* v)
{
  if (m_impls2.find(backend) != m_impls2.end()) {
    alien_fatal([&] { cout() << "try to insert already inserted value"; });
  }
  m_impls2[backend] = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
