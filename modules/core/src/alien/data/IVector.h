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
 * \file IVector.h
 * \brief IVector.h
 */

#pragma once

#include <memory>

#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISpace;
class MultiVectorImpl;
struct ICopyOnWriteVector;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup data
 * \brief Interface for all vectors
 */
class ALIEN_EXPORT IVector
{
 protected:
  //! Constructor
  IVector() {}

 private:
  /* Forbid use of copy and move constructors for implementations. */
  IVector(const IVector&) = delete;
  IVector(IVector&&) = delete;
  void operator=(const IVector&) = delete;
  void operator=(IVector&&) = delete;

 public:
  // Free resources
  virtual ~IVector() {}

  //! Visit method
  virtual void visit(ICopyOnWriteVector&) const = 0;

  /*!
   * \brief Get the space associated to the vector
   * \returns The space
   */
  virtual const ISpace& space() const = 0;

  /*!
   * \brief Get the multivector implementation
   * \returns The multivector implementation
   */
  virtual MultiVectorImpl* impl() = 0;

  /*!
   * \brief Get the multivector implementation
   * \returns The multivector implementation
   */
  virtual const MultiVectorImpl* impl() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
