// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshComponentPartData.h                                     (C) 2000-2017 */
/*                                                                           */
/* Données séparées en parties pure et impures d'un constituant .            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHCOMPONENTPARTDATA_H
#define ARCANE_MATERIALS_MESHCOMPONENTPARTDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/utils/UniqueArray.h"

#include "arcane/materials/MatVarIndex.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Données d'une partie (pure ou partielle) d'un constituant.
 *
 * Cette classe est interne à Arcane.
 */
class MeshComponentPartData
: public TraceAccessor
{
 public:

  MeshComponentPartData(IMeshComponent* component);
  MeshComponentPartData(const MeshComponentPartData& rhs) = default;
  virtual ~MeshComponentPartData();

 public:

 public:

  void setComponentItemInternalView(ConstArrayView<ComponentItemInternal*> v)
  {
    m_items_internal = v;
  }

  void setFromMatVarIndexes(ConstArrayView<MatVarIndex> matvar_indexes);

  Int32 impureVarIdx() const { return m_impure_var_idx; }

  IMeshComponent* component() const { return m_component; }

  ConstArrayView<ComponentItemInternal*> itemsInternal() const { return m_items_internal; }

  void checkValid() const;

  //! Vue sur la partie pure
  ComponentPurePartItemVectorView pureView() const;

  //! Vue sur la partie impure
  ComponentImpurePartItemVectorView impureView() const;

  //! Vue sur la partie \a part
  ComponentPartItemVectorView partView(eMatPart part) const;

 public:

  Int32ConstArrayView valueIndexes(eMatPart k) const
  {
    return m_value_indexes[(Int32)k];
  }

  Int32ConstArrayView itemIndexes(eMatPart k) const
  {
    return m_items_internal_indexes[(Int32)k];
  }

  //! Il faut appeler notifyValueIndexesChanged() après modification du tableau.
  Int32Array& _mutableValueIndexes(eMatPart k)
  {
    return m_value_indexes[(Int32)k];
  }

  void _notifyValueIndexesChanged();

 private:

  //! Gestionnaire de constituants
  IMeshComponent* m_component;

  //! Indice du constituant pour l'accès aux valeurs partielles.
  Int32 m_impure_var_idx;

  //! Liste des valueIndex() de chaque partie
  UniqueArray<Int32> m_value_indexes[2];

  //! Liste des indices dans \a m_items_internal de chaque maille matériau.
  UniqueArray<Int32> m_items_internal_indexes[2];

  //! Liste des ComponentItemInternal* pour ce constituant.
  ConstArrayView<ComponentItemInternal*> m_items_internal;

 private:

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
