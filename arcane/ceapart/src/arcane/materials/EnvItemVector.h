// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EnvItemVector.h                                             (C) 2000-2016 */
/*                                                                           */
/* Vecteur sur les entités d'un milieu.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_ENVITEMVECTOR_H
#define ARCANE_MATERIALS_ENVITEMVECTOR_H
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

class IMeshEnvironment;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vecteur sur les entités d'un milieu.
 *
 * \warning Ce vecteur n'est valide que tant que le milieu et le groupe support
 * ne change pas.
 */
class ARCANE_MATERIALS_EXPORT EnvCellVector
: public ComponentItemVector
{
 public:

  //! Construit un vecteur contenant les entités de \a group pour le milieu \a environment
  EnvCellVector(const CellGroup& group,IMeshEnvironment* environment);
  //! Construit un vecteur contenant les entités de \a view pour le milieu \a environment
  EnvCellVector(CellVectorView view,IMeshEnvironment* environment);
  //! Constructeur par recopie. L'instance fera référence à \a rhs
  EnvCellVector(const EnvCellVector& rhs) = default;
  //! Constructeur de recopie. Cette instance est une copie de \a rhs.
  EnvCellVector(EnvItemVectorView rhs) : ComponentItemVector(rhs){}

 public:

  //! Conversion vers une vue sur ce vecteur
  operator EnvCellVectorView() const
  { return view(); }

  //! Vue sur ce vecteur
  EnvCellVectorView view() const
  {
    return EnvCellVectorView(_component(),matvarIndexes(),itemsInternalView());
  }

  //! Milieu associé
  IMeshEnvironment* environment() const;

  //! Clone ce vecteur
  EnvCellVector clone() const { return EnvCellVector(view()); }

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

