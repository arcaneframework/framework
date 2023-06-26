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
 * \file IVectorConverter.h
 * \brief IVectorConverter.h
 */

#pragma once

#include <alien/core/backend/BackEnd.h>
#include <alien/core/impl/IVectorImpl.h>
#include <alien/utils/ObjectWithTrace.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup core
 * \brief Vectors converter
 *
 * Interface to convert a vector in another format, provided a converter between those two
 * formats has been registered.
 * If there is no direct converter between the source and target format, a third
 * intermediate format will be used if available.
 *
 * \todo IVectorImpl.h is needed only for the template definition. Could be removed
 */
class IVectorConverter : public ObjectWithTrace
{
 public:
  //! Free resources
  virtual ~IVectorConverter() {}

 public:
  /*!
   * \brief Get the source backend id
   * \returns The source backend id
   */
  virtual BackEndId sourceBackend() const = 0;

  /*!
   * \brief Get the target backend id
   * \returns The target backend id
   */
  virtual BackEndId targetBackend() const = 0;

  /*!
   * \brief Convert a vector from one format to another
   * \param[in] sourceImpl Implementation of the source vector
   * \param[in,out] targetImpl Implementation of the target vector
   */
  virtual void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const = 0;

 protected:
  /*!
   * \brief Get the target backend id
   * \returns The target backend id
   */
  template <typename T>
  BackEndId backendId() const { return AlgebraTraits<T>::name(); }

  /*!
   * \brief Cast a vector implementation in its actual type
   * \param[in] impl Vector implementation
   * \param[in] backend Backend id
   * \returns The actual vector implementation
   *
   * \todo Check backend using T
   */
  template <typename T>
  static T& cast(IVectorImpl* impl, BackEndId backend)
  {
    ALIEN_ASSERT((impl != NULL), ("Null implementation"));
    ALIEN_ASSERT((impl->backend() == backend), ("Bad backend"));
    T* t = dynamic_cast<T*>(impl);
    ALIEN_ASSERT((t != NULL), ("Bad dynamic cast"));
    return *t;
  }

  /*!
   * \brief Const cast a vector implementation in its actual type
   * \param[in] impl Vector implemenantation
   * \param[in] backend Backend id
   * \returns The actual vector implementation
   *
   * \todo Check backend using T
   */
  template <typename T>
  static const T& cast(const IVectorImpl* impl, BackEndId backend)
  {
    ALIEN_ASSERT((impl != NULL), ("Null implementation"));
    ALIEN_ASSERT((impl->backend() == backend), ("Bad backend"));
    const T* t = dynamic_cast<const T*>(impl);
    ALIEN_ASSERT((t != NULL), ("Bad dynamic cast"));
    return *t;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
