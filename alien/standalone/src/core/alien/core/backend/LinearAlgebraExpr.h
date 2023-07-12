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
 * \file LinearAlgebraExpr.h
 * \brief LinearAlgebraExpr.h
 */
#pragma once

#include <alien/utils/Precomp.h>

#include <memory>

#include <alien/core/backend/BackEnd.h>
#include <alien/expression/solver/ILinearAlgebra.h>

#include <alien/core/backend/IInternalLinearAlgebraExprT.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Space;
class IVector;
class IMatrix;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup core
 * \brief Linear algebra interface
 *
 * Interface for all linear algebra package
 *
 * \tparam Tag The tag of the type of matrix used
 * \tparam TagV The tag of the type of vector used
 */
template <class Tag, class TagV = Tag>
class LinearAlgebraExpr
{
 public:
  /*!
   * \brief Creates a linear algebra
   *
   * Creates a linear algebra using traits and the linear algebra factory
   *
   * \tparam T Variadics type of linear algebra
   * \param[in] args Linear algebras
   */
  template <typename... T>
  LinearAlgebraExpr(T... args)
  : m_algebra(AlgebraTraits<Tag>::algebra_expr_factory(args...))
  {}

  //! Free resources
  virtual ~LinearAlgebraExpr();

  /*!
   * \brief Compute L0 norm of a vector
   * \param[in] x The vector on which norm0 is computed
   * \returns The norm0 of the vector
   */
  Real norm0(const IVector& x) const;

  /*!
   * \brief Compute L1 norm of a vector
   * \param[in] x The vector on which norm0 is computed
   * \returns The norm1 of the vector
   */
  Real norm1(const IVector& x) const;

  /*!
   * \brief Compute L2 norm of a vector
   * \param[in] x The vector on which norm0 is computed
   * \returns The norm2 of the vector
   */
  Real norm2(const IVector& x) const;

  /*!
   * \brief Compute L2 (Frobenous) norm of a matrix
   * \param[in] x The matrix on which norm2 is computed
   * \returns The norm2 of the matrix
   */
  Real norm2(const IMatrix& x) const;

  /*!
   * \brief Compute a matrix vector product
   *
   * Compute the matrix-vector product a by x and store it in r : r = a * x
   *
   * \param[in] a The matrix to be multiplied
   * \param[in] x The vector to be multipled
   * \param[in,out] r The resulting vector
   */
  void mult(const IMatrix& a, const IVector& x, IVector& r) const;

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
  void axpy(Real alpha, const IVector& x, IVector& y) const;

  /*!
   * \brief Scale a vector by a factor and adds the result to another vector
   *
   * Scale the vector y by the real value alpha and add the values of x : alpha * y += x
   *
   * \param[in] alpha The real value to scale with
   * \param[in,out] y The vector to be scaled
   * \param[in] x The vector to add
   */
  void aypx(Real alpha, IVector& y, const IVector& x) const;

  /*!
   * \brief Copy a vector in another one
   *
   * \param[in] x The vector to copy
   * \param[in,out] r The copied vector
   */
  void copy(const IVector& x, IVector& r) const;

  /*!
   * \brief Copy a matrix in another one
   *
   * \param[in] x The matrix to copy
   * \param[in,out] r The copied matrix
   */
  void copy(const IMatrix& x, IMatrix& r) const;

  /*!
   * \brief Add two  matrices A and B
   *
   *
   * \param[in] a The matrix A
   * \param[in] b The matrix B
   * \param[in,out] c The resulting matrix
   */
  void add(const IMatrix& a, IMatrix& b) const;

  /*!
   * \brief Compute the dot product of two vectors
   * \param[in] x The first vector
   * \param[in] y The second vector
   * \returns The dot product of x * y
   */
  Real dot(const IVector& x, const IVector& y) const;

  /*!
   * \brief Scale a vector by a factor
   * \param[in] alpha The real value to scale with
   * \param[in,out] x The vector to be scaled
   */
  void scal(Real alpha, IVector& x) const;

  /*!
   * \brief Scale a matrix by a factor
   * \param[in] alpha The real value to scale with
   * \param[in,out] x The vector to be scaled
   */
  void scal(Real alpha, IMatrix& a) const;

  /*!
   * \brief Extract the diagonal of a matrix in a vector
   * \param[in] a The matrix to extract the diagonal
   * \param[in,out] x The diagonal elements of the matrix stored in a vector
   */
  void diagonal(const IMatrix& a, IVector& x) const;

  /*!
   * \brief Compute the reciprocal of a vector
   * \param[in,out] x The vector to be processed
   */
  void reciprocal(IVector& x) const;

  /*!
   * \brief Compute the point wise multiplication of two vectors and store the result in
   * another one
   * \param[in] x The first vector
   * \param[in] y The second vector
   * \param[in,out] w The resulting vector
   */
  void pointwiseMult(const IVector& x, const IVector& y, IVector& w) const;

  /*!
   * \brief Compute a matrix vector product
   *
   * Compute the matrix-vector product a by x and store it in r : r = a * x
   *
   * \param[in] a The matrix to be multiplied
   * \param[in] x The vector to be multipled
   * \param[in,out] r The resulting vector
   */
  void mult(const IMatrix& a, const UniqueArray<Real>& x, UniqueArray<Real>& r) const;

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
  void axpy(Real alpha, const UniqueArray<Real>& x, UniqueArray<Real>& r) const;

  /*!
   * \brief Scale a vector by a factor and adds the result to another vector
   *
   * Scale the vector y by the real value alpha and add the values of x : alpha * y += x
   *
   * \param[in] alpha The real value to scale with
   * \param[in,out] y The vector to be scaled
   * \param[in] x The vector to add
   */
  void aypx(Real alpha, UniqueArray<Real>& y, const UniqueArray<Real>& x) const;

  /*!
   * \brief Copy a vector in another one
   *
   * \param[in] x The vector to copy
   * \param[in,out] r The copied vector
   */
  void copy(const UniqueArray<Real>& x, UniqueArray<Real>& r) const;

  /*!
   * \brief Compute the dot product of two vectors
   * \param[in] local_size The size of the vectors
   * \param[in] x The first vector
   * \param[in] y The second vector
   * \returns The dot product of x * y
   */
  Real dot(
  Integer local_size, const UniqueArray<Real>& x, const UniqueArray<Real>& y) const;

  /*!
   * \brief Scale a vector by a factor
   * \param[in] alpha The real value to scale with
   * \param[in,out] x The vector to be scaled
   */
  void scal(Real alpha, UniqueArray<Real>& x) const;

  /*!
   * \brief Dumps a matrix to a file
   * \param[in] a The matrix to dump
   * \param[in] filename The name of the file
   */
  void dump(IMatrix const& a, std::string const& filename) const;

  /*!
   * \brief Dumps a vector to a file
   * \param[in] x The vector to dump
   * \param[in] filename The name of the file
   */
  void dump(IVector const& x, std::string const& filename) const;

 private:
  //! The type of the linear algebra
  typedef typename AlgebraTraits<Tag>::algebra_type KernelAlgebra;
  typedef typename AlgebraTraits<Tag>::algebra_expr_type KernelAlgebraExpr;
  //! The linear algebra kernel
  std::unique_ptr<KernelAlgebraExpr> m_algebra;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
