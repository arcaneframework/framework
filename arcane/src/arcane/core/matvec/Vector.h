// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Vector.h                                                    (C) 2000-2026 */
/*                                                                           */
/* Linear algebra vector.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATVEC_VECTOR_H
#define ARCANE_CORE_MATVEC_VECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/Numeric.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MatVec
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VectorImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Linear algebra vector.
 *
 * This class has a reference semantics.
 */
class ARCANE_CORE_EXPORT Vector
{
 public:

  //! Creates an empty vector
  Vector();

  /*!
   * \brief Created to store \a size elements.
   *
   * The vector is not initialized and its values are arbitrary.
   */
  explicit Vector(Integer size);

  /*!
   * \brief Created to store \a size elements.
   *
   * The vector is initialized with the values of \a init_value.
   */
  Vector(Integer size, Real init_value);

  /*!
   * \brief Creates a vector with the elements of \a v.
   */
  explicit Vector(RealUniqueArray v);

  /*!
   * \brief Constructs a vector that references \a rhs.
   */
  Vector(const Vector& rhs);

  //! Changes the vector reference
  const Vector& operator=(const Vector& rhs);

  //! Destroys the reference
  ~Vector();

 public:

  //! Number of elements in the vector
  Integer size() const;

  /*!
   * \brief Vector values.
   * \warning the returned view is invalidated as soon as the vector
   * is resized.
   */
  RealArrayView values();

  /*!
   * \brief Vector values
   * \warning the returned view is invalidated as soon as the vector
   * is resized.
   */
  RealConstArrayView values() const;

  //! Prints the vector values
  void dump(std::ostream& o) const;

  //! Clones this vector
  Vector clone();

  /*!
   * \brief Copies the elements of \a rhs into this vector.
   *
   * The vector may optionally be resized.
   */
  void copy(const Vector& rhs);

  /*!
   * \brief Changes the number of elements in the vector.
   *
   * If the number of elements increases, the new elements are not
   * initialized.
   */
  void resize(Integer new_size);

  /*!
   * \brief Changes the number of elements in the vector.
   *
   * If the number of elements increases, the new elements are
   * initialized with the value \a init_value.
   */
  void resize(Integer new_size, Real init_value);

  Real normInf();

  //! Initializes a vector using a Hypre format file.
  static Vector readHypre(const String& file_name);

 private:

  //! Internal representation of the group.
  VectorImpl* m_impl = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MatVec

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
