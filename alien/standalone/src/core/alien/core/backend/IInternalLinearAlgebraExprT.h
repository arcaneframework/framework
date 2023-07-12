/*

Copyright 2020 IFPEN-CEA

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

/*!
 * \file IInternalLinearAlgebraExprT.h
 * \brief IInternalLinearAlgebraExprT.h
 */
#pragma once

#include <alien/core/backend/IInternalLinearAlgebraT.h>
#include <alien/utils/Precomp.h>
#include <string>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

class Space;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup core
 * \brief Internal linear algebra interface
 *
 * Internal interface for all linear algebra package
 *
 * \tparam M The type of matrix used
 * \tparam V The type of vector used
 */
template <class M, class V>
class IInternalLinearAlgebraExpr
{
 public:
  //! Type of the matrix used
  typedef M Matrix;
  //! Type of the vector used
  typedef V Vector;
  //! Type of the the linear algebra
  typedef IInternalLinearAlgebraExpr<Matrix, Vector>* (*Factory)();

 public:
  //! Free resources
  virtual ~IInternalLinearAlgebraExpr() {}

 public:
  /*!
   * \brief Compute L0 norm of a vector
   * \param[in] x The vector on which norm0 is computed
   * \returns The norm0 of the vector
   */
  virtual Real norm0(const Vector& x) const = 0;

  /*!
   * \brief Compute L1 norm of a vector
   * \param[in] x The vector on which norm0 is computed
   * \returns The norm1 of the vector
   */
  virtual Real norm1(const Vector& x) const = 0;

  /*!
   * \brief Compute L2 norm of a vector
   * \param[in] x The vector on which norm0 is computed
   * \returns The norm2 of the vector
   */
  virtual Real norm2(const Vector& x) const = 0;

  /*!
   * \brief Compute L2 (Frobenous) norm of a matrix
   * \param[in] x The matrix on which norm2 is computed
   * \returns The norm2 of the matrix
   */
  virtual Real norm2(const Matrix& x) const = 0;

  /*!
   * \brief Compute a matrix vector product
   *
   * Compute the matrix-vector product a by x and store it in r : r = a * x
   *
   * \param[in] a The matrix to be multiplied
   * \param[in] x The vector to be multipled
   * \param[in,out] r The resulting vector
   */
  virtual void mult(const Matrix& a, const Vector& x, Vector& r) const = 0;

  /*!
   * \brief Scale a vector by a factor and adds the result to another vector
   *
   * Scale the vector x by the real value alpha and add the result to the vector y : y +=
   * alpha * x
   *
   * \param[in] alpha The real value to scale with
   * \param[in] x The vector to be scaled
   * \param[in,out] y The resulting vector
   */
  virtual void axpy(Real alpha, const Vector& x, Vector& y) const = 0;

  /*!
   * \brief Scale a vector by a factor and adds the result to another vector
   *
   * Scale the vector y by the real value alpha and add the values of x : alpha * y += x
   *
   * \param[in] alpha The real value to scale with
   * \param[in,out] y The vector to be scaled
   * \param[in] x The vector to add
   */
  virtual void aypx(Real alpha, Vector& y, const Vector& x) const = 0;

  /*!
   * \brief Copy a vector in another one
   *
   * \param[in] x The vector to copy
   * \param[in,out] r The copied vector
   */
  virtual void copy(const Vector& x, Vector& r) const = 0;

  /*!
   * \brief Copy a matrix in another one
   *
   * \param[in] x The vector to copy
   * \param[in,out] r The copied vector
   */
  virtual void copy(const Matrix& a, Matrix& r) const = 0;

  /*!
   * \brief Add a matrix to another one
   *
   * \param[in] a The matrix to copy
   *
   * \param[in,out] r The copied vector
   */
  virtual void add(const Matrix& a, Matrix& r) const = 0;

  /*!
   * \brief Compute the dot product of two vectors
   * \param[in] x The first vector
   * \param[in] y The second vector
   * \returns The dot product of x * y
   */
  virtual Real dot(const Vector& x, const Vector& y) const = 0;

  /*!
   * \brief Scale a vector by a factor
   * \param[in] alpha The real value to scale with
   * \param[in,out] x The vector to be scaled
   */
  virtual void scal(Real alpha, Vector& x) const = 0;

  /*!
   * \brief Scale a matrix by a factor
   * \param[in] alpha The real value to scale with
   * \param[in,out] x The vector to be scaled
   */
  virtual void scal(Real alpha, Matrix& a) const = 0;

  /*!
   * \brief Extract the diagonal of a matrix in a vector
   * \param[in] a The matrix to extract the diagonal
   * \param[in,out] x The diagonal elements of the matrix stored in a vector
   */
  virtual void diagonal(const Matrix& a, Vector& x) const = 0;

  /*!
   * \brief Compute the reciprocal of a vector
   * \param[in,out] x The vector to be processed
   */
  virtual void reciprocal(Vector& x) const = 0;

  /*!
   * \brief Compute the point wise multiplication of two vectors and store the result in
   * another one
   * \param[in] x The first vector
   * \param[in] y The second vector
   * \param[in,out] w The resulting vector
   */
  virtual void pointwiseMult(const Vector& x, const Vector& y, Vector& w) const = 0;

  /*!
   * \brief Dumps a matrix to a file
   * \param[in] a The matrix to dump
   * \param[in] filename The name of the file
   * \todo Implement this method
   */
  virtual void dump(Matrix const& a, std::string const& filename) const
  {
    throw NotImplementedException(
    A_FUNCINFO, "IInternalLinearAlgebra::dump not implemented");
  }

  /*!
   * \brief Dumps a vector to a file
   * \param[in] x The vector to dump
   * \param[in] filename The name of the file
   * \todo Implement this method
   */
  virtual void dump(Vector const& x, std::string const& filename) const
  {
    throw NotImplementedException(
    A_FUNCINFO, "IInternalLinearAlgebra::dump not implemented");
  }

  /*!
   * \brief Compute a matrix vector product
   *
   * Compute the matrix-vector product a by x and store it in r : r = a * x
   *
   * \param[in] a The matrix to be multiplied
   * \param[in] x The vector to be multipled
   * \param[in,out] r The resulting vector
   */
  virtual void mult(
  const Matrix& a, const UniqueArray<Real>& x, UniqueArray<Real>& r) const = 0;

  /*!
   * \brief Scale a vector by a factor and adds the result to another vector
   *
   * Scale the vector x by the real value alpha and add the result to the vector y : y +=
   * alpha * x
   *
   * \param[in] alpha The real value to scale with
   * \param[in] x The vector to be scaled
   * \param[in,out] y The resulting vector
   */
  virtual void axpy(Real alpha, const UniqueArray<Real>& x, UniqueArray<Real>& y) const = 0;

  /*!
   * \brief Scale a vector by a factor and adds the result to another vector
   *
   * Scale the vector y by the real value alpha and add the values of x : alpha * y += x
   *
   * \param[in] alpha The real value to scale with
   * \param[in,out] y The vector to be scaled
   * \param[in] x The vector to add
   */
  virtual void aypx(Real alpha, UniqueArray<Real>& y, const UniqueArray<Real>& x) const = 0;

  /*!
   * \brief Copy a vector in another one
   *
   * \param[in] x The vector to copy
   * \param[in,out] r The copied vector
   */
  virtual void copy(const UniqueArray<Real>& x, UniqueArray<Real>& r) const = 0;

  /*!
   * \brief Compute the dot product of two vectors
   * \param[in] local_size The size of the vectors
   * \param[in] x The first vector
   * \param[in] y The second vector
   * \returns The dot product of x * y
   */
  virtual Real dot(Integer local_size, const UniqueArray<Real>& x,
                   const UniqueArray<Real>& y) const = 0;

  /*!
   * \brief Scale a vector by a factor
   * \param[in] alpha The real value to scale with
   * \param[in,out] x The vector to be scaled
   */
  virtual void scal(Real alpha, UniqueArray<Real>& x) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
