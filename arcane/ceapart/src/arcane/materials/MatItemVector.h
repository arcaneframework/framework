// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatItemVector.h                                             (C) 2000-2016 */
/*                                                                           */
/* Vecteur sur les entités d'un matériau.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MATITEMVECTOR_H
#define ARCANE_MATERIALS_MATITEMVECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/ComponentItemVector.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshMaterial;
class MatCellEnumerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vecteur sur les entités d'un matériau.
 *
 * \warning Ce vecteur n'est valide que tant que le matériau et le groupe support
 * ne change pas.
 */
class ARCANE_MATERIALS_EXPORT MatCellVector
: public ComponentItemVector
{
 public:

  typedef MatCellEnumerator EnumeratorType;

 public:

  //! Construit un vecteur contenant les entités de \a group pour le matériau \a material.
  MatCellVector(const CellGroup& group,IMeshMaterial* material);
  //! Construit un vecteur contenant les entités de \a view pour le matériau \a material.
  MatCellVector(CellVectorView view,IMeshMaterial* material);
  //! Constructeur par recopie. L'instance fera référence à \a rhs
  MatCellVector(const MatCellVector& rhs) = default;
  //! Constructeur de recopie. Cette instance est une copie de \a rhs.
  MatCellVector(MatItemVectorView rhs) : ComponentItemVector(rhs){}

 public:
  
  //! Conversion vers une vue sur ce vecteur
  operator MatCellVectorView() const
  { return view(); }

  //! Vue sur ce vecteur
  MatCellVectorView view() const
  {
    return MatCellVectorView(_component(),matvarIndexes(),itemsInternalView());
  }

  //! Matériau associé
  IMeshMaterial* material() const;

  //! Clone ce vecteur
  MatCellVector clone() const { return MatCellVector(view()); }

 private:

  void _build(CellVectorView view);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

