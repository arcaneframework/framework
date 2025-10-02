// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EnvItemVector.h                                             (C) 2000-2025 */
/*                                                                           */
/* Vecteur sur les entités d'un milieu.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_ENVITEMVECTOR_H
#define ARCANE_CORE_MATERIALS_ENVITEMVECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/ComponentItemVector.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vecteur sur les entités d'un milieu.
 *
 * \warning Ce vecteur n'est valide que tant que le milieu ne change pas.
 */
class ARCANE_CORE_EXPORT EnvCellVector
: public ComponentItemVector
{
 public:

  //! Construit un vecteur contenant les entités de \a group pour le milieu \a environment
  EnvCellVector(const CellGroup& group, IMeshEnvironment* environment);
  //! Construit un vecteur contenant les entités de \a view pour le milieu \a environment
  EnvCellVector(CellVectorView view, IMeshEnvironment* environment);
  //! Construit un vecteur contenant les entités \a local_ids pour le milieu \a environment
  EnvCellVector(SmallSpan<const Int32> local_ids, IMeshEnvironment* environment);
  //! Construit un vecteur sur les entités du milieu \a environment.
  EnvCellVector(const ConstituentItemVectorBuildInfo& build_info, IMeshEnvironment* environment);
  //! Constructeur par recopie. L'instance fera référence à \a rhs
  EnvCellVector(const EnvCellVector& rhs) = default;
  //! Constructeur de recopie. Cette instance est une copie de \a rhs.
  EnvCellVector(EnvItemVectorView rhs)
  : ComponentItemVector(rhs)
  {}

 public:

  //! Conversion vers une vue sur ce vecteur
  operator EnvCellVectorView() const
  {
    return view();
  }

  //! Vue sur ce vecteur
  EnvCellVectorView view() const
  {
    return { _component(), _matvarIndexes(), _constituentItemListView(), _localIds() };
  }

  //! Milieu associé
  IMeshEnvironment* environment() const;

  //! Clone ce vecteur
  EnvCellVector clone() const { return { view() }; }

 private:

  void _build(SmallSpan<const Int32> view);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

