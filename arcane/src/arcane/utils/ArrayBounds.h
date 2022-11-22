// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayBounds.h                                               (C) 2000-2022 */
/*                                                                           */
/* Gestion des itérations sur les tableaux N-dimensions                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ARRAYBOUNDS_H
#define ARCANE_UTILS_ARRAYBOUNDS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayExtents.h"

/*
 * ATTENTION:
 *
 * Toutes les classes de ce fichier sont expérimentales et l'API n'est pas
 * figée. A NE PAS UTILISER EN DEHORS DE ARCANE.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ExtentType>
class ArrayBoundsBase
: private ArrayExtents<ExtentType>
{
 public:

  using BaseClass = ArrayExtents<ExtentType>;
  using BaseClass::asStdArray;
  using BaseClass::constExtent;
  using BaseClass::getIndices;
  using IndexType = typename BaseClass::IndexType;
  using ArrayExtentType = Arcane::ArrayExtents<ExtentType>;

 public:

  constexpr ArrayBoundsBase()
  : m_nb_element(0)
  {}
  constexpr explicit ArrayBoundsBase(const BaseClass& rhs)
  : ArrayExtents<ExtentType>(rhs)
  {
    m_nb_element = this->totalNbElement();
  }

 public:

  constexpr ARCCORE_HOST_DEVICE Int64 nbElement() const { return m_nb_element; }

 private:

  Int64 m_nb_element;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <int X0>
class ArrayBounds<ExtentsV<X0>>
: public ArrayBoundsBase<ExtentsV<X0>>
{
  using ExtentsType = ExtentsV<X0>;
  using BaseClass = ArrayBoundsBase<ExtentsType>;

 public:

  // Note: le constructeur ne doit pas être explicite pour permettre la conversion
  // à partir d'un entier.
  constexpr ArrayBounds(Int32 dim1)
  : BaseClass(ArrayExtents<ExtentsType>(dim1))
  {
  }

  constexpr explicit ArrayBounds(const ArrayExtents<ExtentsType>& v)
  : BaseClass(v)
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<int X0,int X1>
class ArrayBounds<ExtentsV<X0,X1>>
: public ArrayBoundsBase<ExtentsV<X0,X1>>
{
  using ExtentsType = ExtentsV<X0,X1>;
  using BaseClass = ArrayBoundsBase<ExtentsType>;

 public:

  constexpr ArrayBounds(Int32 dim1, Int32 dim2)
  : BaseClass(ArrayExtents<ExtentsType>(dim1, dim2))
  {
  }

  constexpr explicit ArrayBounds(const ArrayExtents<ExtentsType>& v)
  : BaseClass(v)
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<int X0,int X1,int X2>
class ArrayBounds<ExtentsV<X0,X1,X2>>
: public ArrayBoundsBase<ExtentsV<X0,X1,X2>>
{
  using ExtentsType = ExtentsV<X0,X1,X2>;
  using BaseClass = ArrayBoundsBase<ExtentsType>;

 public:

  constexpr ArrayBounds(Int32 dim1, Int32 dim2, Int32 dim3)
  : BaseClass(ArrayExtents<ExtentsType>(dim1, dim2, dim3))
  {
  }

  constexpr explicit ArrayBounds(const ArrayExtents<ExtentsType>& v)
  : BaseClass(v)
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<int X0,int X1,int X2,int X3>
class ArrayBounds<ExtentsV<X0,X1,X2,X3>>
: public ArrayBoundsBase<ExtentsV<X0,X1,X2,X3>>
{
  using ExtentsType = ExtentsV<X0,X1,X2,X3>;
  using BaseClass = ArrayBoundsBase<ExtentsType>;

 public:

  constexpr ArrayBounds(Int32 dim1, Int32 dim2, Int32 dim3, Int32 dim4)
  : BaseClass(ArrayExtents<ExtentsType>(dim1, dim2, dim3, dim4))
  {
  }

  constexpr explicit ArrayBounds(const ArrayExtents<ExtentsType>& v)
  : BaseClass(v)
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
