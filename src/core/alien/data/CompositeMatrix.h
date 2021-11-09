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
 * \file alien/data/CompositeMatrix.h
 * \brief CompositeMatrix.h
 */

#pragma once

#include <alien/data/IMatrix.h>
#include <alien/kernels/composite/CompositeMatrixElement.h>

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

  class Matrix;
  class MultiMatrixImpl;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace CompositeKernel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup data
 * \brief Composite matrix for heterogenous matrices
 *
 * This class allow to handle matrices made of several matrices. It allows to deal with a
 * single matrix composed of severals submatrices. This is especially usefull to apply
 * different solvers on a linear system, or to build more efficiently submatrices due to
 * specific problems implying different matrices data structure.
 */
class ALIEN_EXPORT CompositeMatrix : public IMatrix
{
 public:
  //! Type of submatrix element
  using Element = CompositeKernel::MatrixElement;

 public:
  //! Constructor
  CompositeMatrix();

  /*!
   * \brief Constructor
   * \param[in] nc The number of submatrices
   */
  CompositeMatrix(Arccore::Integer nc);

  //! Free resources
  virtual ~CompositeMatrix() {}

  /*!
   * \brief Visit method
   * \param[in] visit Visit matrix method
   */
  void visit(ICopyOnWriteMatrix& visit) const;

  /*!
   * \brief Resize the number of submatrices
   * \param[in] nc The number of submatrices
   */
  void resize(Arccore::Integer nc);

  /*!
   * \brief Get the number of submatrices
   * \returns The number of submatrices
   */
  Arccore::Integer size() const;

  /*!
   * \brief Get the row space of the global matrix
   * \returns The row space
   */
  const ISpace& rowSpace() const;

  /*!
   * \brief Get the col space of the global matrix
   * \returns The col space
   */
  const ISpace& colSpace() const;

  /*!
   * \brief Get the (i,j) element
   * \param[in] i The row index of the element
   * \param[in] j The col index of the element
   * \returns The (i,j) element
   */
  Element composite(Arccore::Integer i, Arccore::Integer j);

  /*!
   * \brief Get the (i,j) submatrix
   * \param[in] i The row index of the submatrix
   * \param[in] j The col index of the submatrix
   * \returns The (i,j) submatrix
   */
  IMatrix& operator()(Arccore::Integer i, Arccore::Integer j);

  /*!
   * \brief Get the (i,j) submatrix
   * \param[in] i The row index of the submatrix
   * \param[in] j The col index of the submatrix
   * \returns The (i,j) submatrix
   */
  const IMatrix& operator()(Arccore::Integer i, Arccore::Integer j) const;

  /*!
   * \brief Add a feature to the composite matrix
   * \param[in] feature The feature to add
   */
  void setUserFeature(Arccore::String feature);

  /*!
   * \brief Check if the composite matrix has a feature
   * \returns Wheteher or not the composite matrix has the feature
   */
  bool hasUserFeature(Arccore::String feature) const;

 public:
  /*!
   * \brief Get the multimatrix implementation
   * \returns The multimatrix implementation
   */
  MultiMatrixImpl* impl();

  /*!
   * \brief Get the multimatrix implementation
   * \returns The multimatrix implementation
   */
  const MultiMatrixImpl* impl() const;

  //! Free the composite matrix
  void free();

  //! Clear the composite matrix
  void clear();

 private:
  //! The multimatrix implementation
  std::shared_ptr<CompositeKernel::MultiMatrixImpl> m_impl;
  //! The composite matrix
  CompositeKernel::Matrix& m_composite_matrix;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern ALIEN_EXPORT CompositeMatrix::Element CompositeElement(
CompositeMatrix& m, Arccore::Integer i, Arccore::Integer j);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
