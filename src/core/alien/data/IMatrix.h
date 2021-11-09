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
 * \file IMatrix.h
 * \brief IMatrix.h
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
class MultiMatrixImpl;
struct ICopyOnWriteMatrix;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup data
 * \brief Interface for all matrices
 */
class ALIEN_EXPORT IMatrix
{
 protected:
  //! Constructor
  IMatrix() {}

 private:
  /* Forbid use of copy and move constructors for implementations. */
  IMatrix(const IMatrix&) = delete;
  IMatrix(IMatrix&&) = delete;
  void operator=(const IMatrix&) = delete;
  void operator=(IMatrix&&) = delete;

 public:
  // Free resources
  virtual ~IMatrix() {}

  //! Visit method
  virtual void visit(ICopyOnWriteMatrix&) const = 0;

  /*!
   * \brief Get row space associated to the matrix
   * \returns The row space
   */
  virtual const ISpace& rowSpace() const = 0;

  /*!
   * \brief Get col space associated to the matrix
   * \returns The col space
   */
  virtual const ISpace& colSpace() const = 0;

  /*!
   * \brief Get the multimatrix implementation
   * \returns The multimatrix implementation
   */
  virtual MultiMatrixImpl* impl() = 0;

  /*!
   * \brief Get the multimatrix implementation
   * \returns The multimatrix implementation
   */
  virtual const MultiMatrixImpl* impl() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
