// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshComponentPartData.h                                     (C) 2000-2022 */
/*                                                                           */
/* Données séparées en parties pure et impures d'un constituant .            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_MESHCOMPONENTPARTDATA_H
#define ARCANE_CORE_MATERIALS_MESHCOMPONENTPARTDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/UniqueArray.h"

#include "arcane/core/materials/MatVarIndex.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Données d'une partie (pure ou partielle) d'un constituant.
 *
 * Cette classe est interne à Arcane.
 */
class ARCANE_CORE_EXPORT MeshComponentPartData
: public TraceAccessor
{
  friend class MeshComponentData;
  friend class ComponentItemVector;

 public:

  explicit MeshComponentPartData(IMeshComponent* component);
  MeshComponentPartData(const MeshComponentPartData& rhs) = default;
  ~MeshComponentPartData() override;

 public:

  Int32 impureVarIdx() const { return m_impure_var_idx; }

  IMeshComponent* component() const { return m_component; }

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

 private:

  void _setComponentItemInternalView(ConstArrayView<ComponentItemInternal*> v)
  {
    m_items_internal = v;
  }

  void _setFromMatVarIndexes(ConstArrayView<MatVarIndex> matvar_indexes);

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

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
