// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshComponentPartData.h                                     (C) 2000-2024 */
/*                                                                           */
/* Data separated into pure and impure parts of a constituent.               */
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
 * \brief Data of a part (pure or partial) of a constituent.
 *
 * This class is internal to Arcane.
 */
class MeshComponentPartData
: public TraceAccessor
{
 public:

  MeshComponentPartData(IMeshComponent* component, const String& debug_name);

 public:

  Int32 impureVarIdx() const { return m_impure_var_idx; }

  IMeshComponent* component() const { return m_component; }

  void checkValid();

  //! View of the pure part
  ComponentPurePartItemVectorView pureView();

  //! View of the impure part
  ComponentImpurePartItemVectorView impureView();

  //! View of the \a part
  ComponentPartItemVectorView partView(eMatPart part);

  /*
   * \brief Functor to recalculate the pure and impure parts following a modification.
   *
   * If this functor is not set, then the instance must be updated manually
   * via the call to _setFromMatVarIndexes(). \a func must remain valid
   * during the entire lifetime of this instance
   */
  void setRecomputeFunctor(IFunctor* func) { m_compute_functor = func; }

  void setNeedRecompute() { m_is_need_recompute = true; }

 public:

  void _setConstituentListView(const ConstituentItemLocalIdListView& v);
  void _setFromMatVarIndexes(ConstArrayView<MatVarIndex> matvar_indexes, RunQueue& queue);
  void _setFromMatVarIndexes(ConstArrayView<MatVarIndex> globals,
                             ConstArrayView<MatVarIndex> multiples);

 private:

  //! Constituent manager
  IMeshComponent* m_component = nullptr;

  //! Index of the constituent for accessing partial values.
  Int32 m_impure_var_idx = -1;

  //! List of valueIndex() for each part
  FixedArray<UniqueArray<Int32>, 2> m_value_indexes;

  //! List of indices in \a m_items_internal for each material mesh.
  FixedArray<UniqueArray<Int32>, 2> m_items_internal_indexes;

  //! List of ComponentItems for this constituent.
  ConstituentItemLocalIdListView m_constituent_list_view;

  IFunctor* m_compute_functor = nullptr;
  bool m_is_need_recompute = false;

 public:

  // This function is private but must be made public to compile with CUDA.
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
