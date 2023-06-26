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
 * \file ILinearAlgebra.h
 * \brief ILinearAlgebra.h
 */

#pragma once

#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVector;
class IMatrix;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup expression
 * \brief Interface for linear algebra
 */
class ILinearAlgebra
{
 public:
  //! Free resources
  virtual ~ILinearAlgebra() {}

 public:
  /*!
   * \brief Computes norm 0 of a vector
   * \param[in] x The vector
   * \returns The norm0 of the vector
   */
  virtual Arccore::Real norm0(const IVector& x) const = 0;

  /*!
   * \brief Computes norm 1 of a vector
   * \param[in] x The vector
   * \returns The norm1 of the vector
   */
  virtual Arccore::Real norm1(const IVector& x) const = 0;

  /*!
   * \brief Computes norm 2 of a vector
   * \param[in] x The vector
   * \returns The norm2 of the vector
   */
  virtual Arccore::Real norm2(const IVector& x) const = 0;

  /*!
   * \brief Computes a matrix vector product
   * \param[in] a The matrix to multiply
   * \param[in] x The vector to multiply
   * \param[in,out] r The vector to store the result
   */
  virtual void mult(const IMatrix& a, const IVector& x, IVector& r) const = 0;

  /*!
   * \brief Computes y += alpa * x
   * \param[in] alpha The real value to scale the vector
   * \param[in] x The vector to scale
   * \param[in, out] y The vector to store the result
   */
  virtual void axpy(Real alpha, const IVector& x, IVector& y) const = 0;

  /*!
   * \brief Copy a vector
   * \param[in] x The vector to copy
   * \param[in,out] r The copied vector
   */
  virtual void copy(const IVector& x, IVector& r) const = 0;

  /*!
   * \brief Computes the dot product of two vectors
   * \param[in] x The first vector
   * \param[in] y The second vector
   * \returns The dot product
   */
  virtual Arccore::Real dot(const IVector& x, const IVector& y) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
