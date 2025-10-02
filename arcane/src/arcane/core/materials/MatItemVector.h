// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatItemVector.h                                             (C) 2000-2025 */
/*                                                                           */
/* Vecteur sur les entités d'un matériau.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_MATITEMVECTOR_H
#define ARCANE_CORE_MATERIALS_MATITEMVECTOR_H
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
 * \brief Vecteur sur les entités d'un matériau.
 *
 * \warning Ce vecteur n'est valide que tant que le matériau ne change pas.
 */
class ARCANE_CORE_EXPORT MatCellVector
: public ComponentItemVector
{
 public:

  typedef MatCellEnumerator EnumeratorType;

 public:

  //! Construit un vecteur contenant les entités de \a group pour le matériau \a material.
  MatCellVector(const CellGroup& group, IMeshMaterial* material);
  //! Construit un vecteur contenant les entités de \a view pour le matériau \a material.
  MatCellVector(CellVectorView view, IMeshMaterial* material);
  //! Construit un vecteur contenant les entités \a local_ids pour le matériau \a material.
  MatCellVector(SmallSpan<const Int32> local_ids, IMeshMaterial* material);
  //! Construit un vecteur sur les entités du matériau \a material.
  MatCellVector(const ConstituentItemVectorBuildInfo& build_info, IMeshMaterial* material);
  //! Constructeur par recopie. L'instance fera référence à \a rhs
  MatCellVector(const MatCellVector& rhs) = default;
  //! Constructeur de recopie. Cette instance est une copie de \a rhs.
  MatCellVector(MatItemVectorView rhs)
  : ComponentItemVector(rhs)
  {}

 public:

  //! Conversion vers une vue sur ce vecteur
  operator MatCellVectorView() const
  {
    return view();
  }

  //! Vue sur ce vecteur
  MatCellVectorView view() const
  {
    return { _component(), _matvarIndexes(), _constituentItemListView(), _localIds() };
  }

  //! Matériau associé
  IMeshMaterial* material() const;

  //! Clone ce vecteur
  MatCellVector clone() const { return { view() }; }

 private:

  void _build(SmallSpan<const Int32> view);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

