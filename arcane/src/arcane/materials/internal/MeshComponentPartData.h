// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshComponentPartData.h                                     (C) 2000-2024 */
/*                                                                           */
/* Données séparées en parties pures et impures d'un constituant.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_MESHCOMPONENTPARTDATA_H
#define ARCANE_MATERIALS_INTERNAL_MESHCOMPONENTPARTDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/FixedArray.h"

#include "arcane/core/materials/MatVarIndex.h"
#include "arcane/core/materials/ComponentItemInternal.h"

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
class MeshComponentPartData
: public TraceAccessor
{
 public:

  MeshComponentPartData(IMeshComponent* component,const String& debug_name);

 public:

  Int32 impureVarIdx() const { return m_impure_var_idx; }

  IMeshComponent* component() const { return m_component; }

  void checkValid();

  //! Vue sur la partie pure
  ComponentPurePartItemVectorView pureView();

  //! Vue sur la partie impure
  ComponentImpurePartItemVectorView impureView();

  //! Vue sur la partie \a part
  ComponentPartItemVectorView partView(eMatPart part);

  /*
   * \brief Fonctor pour recalculer les parties pures et impures suite à une modification.
   *
   * Si ce fonctor n'est pas positionné, alors il faut mettre à jour manuellement
   * l'instance via l'appel à _setFromMatVarIndexes(). \a func doit rester valide
   * durant toute la durée de vie de cette instance
   */
  void setRecomputeFunctor(IFunctor* func) { m_compute_functor = func; }

  void setNeedRecompute() { m_is_need_recompute = true; }

 public:

  void _setConstituentListView(const ConstituentItemLocalIdListView& v);
  void _setFromMatVarIndexes(ConstArrayView<MatVarIndex> matvar_indexes, RunQueue& queue);
  void _setFromMatVarIndexes(ConstArrayView<MatVarIndex> globals,
                             ConstArrayView<MatVarIndex> multiples);

 private:

  //! Gestionnaire de constituants
  IMeshComponent* m_component = nullptr;

  //! Indice du constituant pour l'accès aux valeurs partielles.
  Int32 m_impure_var_idx = -1;

  //! Liste des valueIndex() de chaque partie
  FixedArray<UniqueArray<Int32>, 2> m_value_indexes;

  //! Liste des indices dans \a m_items_internal de chaque maille matériau.
  FixedArray<UniqueArray<Int32>, 2> m_items_internal_indexes;

  //! Liste des ComponentItem pour ce constituant.
  ConstituentItemLocalIdListView m_constituent_list_view;

  IFunctor* m_compute_functor = nullptr;
  bool m_is_need_recompute = false;

 public:

  // Cette fonction est privée mais doit être rendue publique pour compiler avec CUDA.
  void _notifyValueIndexesChanged(RunQueue* queue);

 private:

  void _checkNeedRecompute();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
