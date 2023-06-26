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
 * \file IMatrixConverter.h
 * \brief IMatrixConverter.h
 */

#pragma once

#include <alien/core/backend/BackEnd.h>
#include <alien/core/impl/IMatrixImpl.h>
#include <alien/utils/ObjectWithTrace.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup core
 * \brief Matrices converter
 *
 * Interface to convert a matrix in another format, provided a converter between those two
 * formats has been registered.
 * If there is no direct converter between the source and target format, a third
 * intermediate format will be used if available.
 *
 * \todo IMatrixImpl.h is needed only for the template definition. Could be removed
 */
class IMatrixConverter : public Alien::ObjectWithTrace
{
 public:
  //! Type of matrix implementation
  typedef Alien::IMatrixImpl IMatrixImpl;
  //! Type of matrix backend
  typedef Alien::BackEndId BackEndId;

 public:
  //! Free resources
  virtual ~IMatrixConverter() {}

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
   * \brief Convert a matrix from one format to another
   * \param[in] sourceImpl Implementation of the source matrix
   * \param[in,out] targetImpl Implementation of the target matrix
   */
  virtual void convert(const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const = 0;

 public:
  /*!
   * \brief Cast a matrix implementation in its actual type
   * \param[in] impl Matrix implementation
   * \param[in] backend Backend id
   * \returns The actual matrix implementation
   *
   * \todo Check backend using T
   */
  template <typename T>
  static T& cast(IMatrixImpl* impl, BackEndId backend)
  {
    ALIEN_ASSERT((impl != NULL), ("Null implementation"));
    ALIEN_ASSERT((impl->backend() == backend), ("Bad backend"));
    T* t = dynamic_cast<T*>(impl);
    ALIEN_ASSERT((t != NULL), ("Bad dynamic cast"));
    return *t;
  }

  /*!
   * \brief Const cast a matrix implementation in its actual type
   * \param[in] impl Matrix implemenantation
   * \param[in] backend Backend id
   * \returns The actual matrix implementation
   *
   * \todo Check backend using T
   */
  template <typename T>
  static const T& cast(const IMatrixImpl* impl, BackEndId backend)
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
