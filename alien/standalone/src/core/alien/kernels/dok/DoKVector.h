/*
 * Copyright 2021 IFPEN-CEA
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
 *  SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <alien/core/impl/IVectorImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>

#include <alien/kernels/dok/DoKDistributor.h>

#include <alien/kernels/redistributor/Redistributor.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

class DoKDistributor;
class DoKToSimpleCSRVectorConverter;

/*!
 * Vector stored as Dictionary Of Keys
 *
 * This allows to create "sparse" vector, as `id` are not required to be consecutive nor to have any meaning.
 */
class ALIEN_EXPORT DoKVector : public IVectorImpl
{
 public:
  typedef Real ValueType;

 public:
  explicit DoKVector(const MultiVectorImpl* multi_impl = nullptr)
  : IVectorImpl(multi_impl, "DoK")
  , m_data()
  {}

  DoKVector(const DoKVector&) = delete;

  ~DoKVector() override = default;

  void clear() override
  {
    m_data.clear();
  }

  /// Contribute to a non zero, identified by its index
  /// \param index can be local or remote
  /// \param value
  /// \return updated local value.
  std::optional<ValueType> contribute(Arccore::Int32 index, ValueType value)
  {
    m_data[index] += value;
    return m_data[index];
  }

  /// Set a non zero, identified by its index. Should not be used on remote values as we do not have global ordering between calls.
  /// \param index can be local or remote
  /// \param value
  /// \return updated local value.
  std::optional<ValueType> set(Arccore::Int32 index, ValueType value)
  {
    //     if (distribution().owner(index) != distribution().parallelMng()->commRank()) {
    //       return std::nullopt;
    //     }
    m_data[index] = value;
    return value;
  }

  //! Dispatch matrix elements
  void assemble() { _distribute(); }

  void reserve(Arccore::Int32 size)
  {
    m_data.reserve(size);
  }

 private:
  void _distribute();
  friend DoKDistributor;
  friend DoKToSimpleCSRVectorConverter;

 private:
  std::unordered_map<Arccore::Int32, DoKVector::ValueType> m_data;
};

} // namespace Alien
