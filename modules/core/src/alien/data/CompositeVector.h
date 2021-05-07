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
 * \file alien/data/CompositeVector.h
 * \brief CompositeVector.h
 */

#pragma once

#include <alien/data/IVector.h>
#include <alien/kernels/composite/CompositeVectorElement.h>
#include <alien/utils/ObjectWithTrace.h>
#include <cstdlib>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace CompositeKernel
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  class Vector;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace CompositeKernel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup data
 * \brief Composite vector for heterogenous vector
 *
 * This class allow to handle vectors made of several vectors
 */
class ALIEN_EXPORT CompositeVector : public IVector
, private ObjectWithTrace
{
 public:
  //! Type of submatrix element
  using Element = CompositeKernel::VectorElement;

 public:
  //! Constructor
  CompositeVector();

  /*!
   * \brief Constructor
   * \param[in] nc The number of subvectors
   */
  CompositeVector(Arccore::Integer nc);

  //! Free resources
  virtual ~CompositeVector() {}

  /*!
   * \brief Visit method
   * \param[in] visit Visit vector method
   */
  void visit(ICopyOnWriteVector&) const;

  /*!
   * \brief Get the  space of the global vector
   * \returns The space
   */
  const ISpace& space() const;

  /*!
   * \brief Resize the number of submatrices
   * \param[in] nc The number of submatrices
   */
  void resize(Arccore::Integer nc);

  /*!
   * \brief Get the number of subvectors
   * \returns The number of subvectors
   */
  Arccore::Integer size() const;

  /*!
   * \brief Get the i-th element
   * \param[in] i The index of the element
   * \returns The i-th element
   */
  Element composite(Arccore::Integer i);

  /*!
   * \brief Get the i-th subvector
   * \param[in] i The index of the subvector
   * \returns The i-th subvector
   */
  IVector& operator[](Arccore::Integer i);

  /*!
   * \brief Get the i-th subvector
   * \param[in] i The index of the subvector
   * \returns The i-th subvector
   */
  const IVector& operator[](Arccore::Integer i) const;

  /*!
   * \brief Add a feature to the composite vector
   * \param[in] feature The feature to add
   */
  void setUserFeature(Arccore::String feature);

  /*!
   * \brief Check if the composite vector has a feature
   * \returns Wheteher or not the composite matrix has the feature
   */
  bool hasUserFeature(Arccore::String feature) const;

 public:
  /*!
   * \brief Get the multivector implementation
   * \returns The multivector implementation
   */
  MultiVectorImpl* impl();

  /*!
   * \brief Get the multivector implementation
   * \returns The multivector implementation
   */
  const MultiVectorImpl* impl() const;

  //! Free resources
  void free();

  //! Clear resources
  void clear();

 private:
  //! The multivector implementation
  std::shared_ptr<MultiVectorImpl> m_impl;
  //! The composite vector
  CompositeKernel::Vector& m_composite_vector;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern ALIEN_EXPORT CompositeVector::Element CompositeElement(
CompositeVector& v, Arccore::Integer i);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
