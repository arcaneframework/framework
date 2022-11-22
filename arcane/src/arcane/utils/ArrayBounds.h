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

namespace impl
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ExtentType>
class ArrayBoundsBase
: private ArrayExtents<ExtentType>
{
 public:

  using BaseClass = ArrayExtents<ExtentType>;
  using BaseClass::getIndices;
  using BaseClass::asStdArray;
  using BaseClass::constExtent;
  using IndexType = typename BaseClass::IndexType;

 public:

  constexpr ArrayBoundsBase() : m_nb_element(0) {}
  constexpr explicit ArrayBoundsBase(const ArrayExtents<ExtentType>& rhs)
  : ArrayExtents<ExtentType>(rhs)
  {
    m_nb_element = this->totalNbElement();
  }

 public:

  ARCCORE_HOST_DEVICE constexpr Int64 nbElement() const { return m_nb_element; }

 private:

  Int64 m_nb_element;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayBounds<MDDim1>
: public ArrayBoundsBase<MDDim1>
{
 public:
  // Note: le constructeur ne doit pas être explicite pour permettre la conversion
  // à partir d'un entier.
  constexpr ArrayBounds(Int32 dim1)
  : ArrayBoundsBase<MDDim1>(ArrayExtents<MDDim1>(dim1))
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayBounds<MDDim2>
: public ArrayBoundsBase<MDDim2>
{
 public:
  constexpr ArrayBounds(Int32 dim1,Int32 dim2)
  : ArrayBoundsBase<MDDim2>(ArrayExtents<MDDim2>(dim1,dim2))
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayBounds<MDDim3>
: public ArrayBoundsBase<MDDim3>
{
 public:
  constexpr ArrayBounds(Int32 dim1,Int32 dim2,Int32 dim3)
  : ArrayBoundsBase<MDDim3>(ArrayExtents<MDDim3>(dim1,dim2,dim3))
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayBounds<MDDim4>
: public ArrayBoundsBase<MDDim4>
{
 public:
  constexpr ArrayBounds(Int32 dim1,Int32 dim2,Int32 dim3,Int32 dim4)
  : ArrayBoundsBase<MDDim4>(ArrayExtents<MDDim4>(dim1,dim2,dim3,dim4))
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
