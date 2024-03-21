// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshComponentPartData.cc                                    (C) 2000-2024 */
/*                                                                           */
/* Données d'une partie (pure ou partielle) d'un constituant.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemPrinter.h"

#include "arcane/core/materials/IMeshComponent.h"
#include "arcane/core/materials/ComponentItemInternal.h"
#include "arcane/core/materials/ComponentPartItemVectorView.h"
#include "arcane/core/materials/internal/IMeshComponentInternal.h"

#include "arcane/materials/internal/MeshComponentPartData.h"

#include "arcane/accelerator/Filter.h"
#include "arcane/accelerator/RunCommandLoop.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshComponentPartData::
MeshComponentPartData(IMeshComponent* component, const String& debug_name)
: TraceAccessor(component->traceMng())
, m_component(component)
, m_impure_var_idx(component->_internalApi()->variableIndexerIndex() + 1)
{
  // Utilise l'allocateur des données pour permettre d'accéder à ces valeurs
  // sur les accélérateurs
  IMemoryAllocator* allocator = platform::getDefaultDataAllocator();
  for (Integer i = 0; i < 2; ++i) {
    m_value_indexes[i] = UniqueArray<Int32>(allocator);
    m_items_internal_indexes[i] = UniqueArray<Int32>(allocator);
  }
  if (!debug_name.empty()) {
    String base_name = String("MeshComponentPartData") + debug_name;
    for (Integer i = 0; i < 2; ++i) {
      m_value_indexes[i].setDebugName(base_name + "ValueIndexes" + String::fromNumber(i));
      m_items_internal_indexes[i].setDebugName(base_name + "ValueIndexes" + String::fromNumber(i));
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentPartData::
_notifyValueIndexesChanged(RunQueue* queue)
{
  if (queue && queue->isAcceleratorPolicy()) {
    SmallSpan<Int32> v0 = m_value_indexes[0].smallSpan();
    SmallSpan<Int32> v1 = m_value_indexes[1].smallSpan();
    Int32 size0 = v0.size();
    Int32 size1 = v1.size();
    Int32 padding_size0 = arcaneSizeWithPadding(size0);
    Int32 padding_size1 = arcaneSizeWithPadding(size1);
    SmallSpan<Int32> padded_v0(v0.data(), padding_size0);
    SmallSpan<Int32> padded_v1(v1.data(), padding_size1);
    if (padding_size0 != size0 || padding_size1 != size1) {
      auto command = makeCommand(queue);
      command << RUNCOMMAND_LOOP1(iter, 1)
      {
        if (size0 > 0) {
          Int32 last_value = v0[size0 - 1];
          for (Int32 k = size0; k < padding_size0; ++k)
            padded_v0[k] = last_value;
        }
        if (size1 > 0) {
          Int32 last_value = v1[size1 - 1];
          for (Int32 k = size1; k < padding_size1; ++k)
            padded_v1[k] = last_value;
        }
      };
    }
  }
  else {
    applySimdPadding(m_value_indexes[0]);
    applySimdPadding(m_value_indexes[1]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentPartData::
_setFromMatVarIndexes(ConstArrayView<MatVarIndex> matvar_indexes, RunQueue& queue)
{
  Int32 nb_index = matvar_indexes.size();

  Int32Array& pure_indexes = m_value_indexes[(Int32)eMatPart::Pure];
  Int32Array& impure_indexes = m_value_indexes[(Int32)eMatPart::Impure];

  Int32Array& pure_internal_indexes = m_items_internal_indexes[(Int32)eMatPart::Pure];
  Int32Array& impure_internal_indexes = m_items_internal_indexes[(Int32)eMatPart::Impure];

  pure_indexes.resize(nb_index);
  pure_internal_indexes.resize(nb_index);

  // TODO: Faire une première passe pour calculer le nombre de valeurs pures
  // et ainsi allouer directement à la bonne taille.
  info(4) << "BEGIN_BUILD_PART_DATA_FOR_COMPONENT c=" << m_component->name();
  Accelerator::GenericFilterer filterer(&queue);

  Int32 nb_impure = 0;
  {
    SmallSpan<Int32> pure_indexes_view(pure_indexes.view());
    SmallSpan<Int32> pure_internal_indexes_view(pure_internal_indexes.view());

    auto is_pure_lambda = [=] ARCCORE_HOST_DEVICE(Int32 i) -> bool {
      return matvar_indexes[i].arrayIndex() == 0;
    };
    auto setter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 i, Int32 output_index) {
      pure_indexes_view[output_index] = matvar_indexes[i].valueIndex();
      pure_internal_indexes_view[output_index] = i;
    };
    filterer.applyWithIndex(nb_index, is_pure_lambda, setter_lambda, A_FUNCINFO);
    Int32 nb_out = filterer.nbOutputElement();
    pure_indexes.resize(nb_out);
    pure_internal_indexes.resize(nb_out);
    nb_impure = nb_index - nb_out;
  }

  impure_indexes.resize(nb_impure);
  impure_internal_indexes.resize(nb_impure);

  {
    SmallSpan<Int32> impure_indexes_view(impure_indexes.view());
    SmallSpan<Int32> impure_internal_indexes_view(impure_internal_indexes.view());
    auto is_impure_lambda = [=] ARCCORE_HOST_DEVICE(Int32 i) -> bool {
      return matvar_indexes[i].arrayIndex() != 0;
    };
    auto setter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 i, Int32 output_index) {
      impure_indexes_view[output_index] = matvar_indexes[i].valueIndex();
      impure_internal_indexes_view[output_index] = i;
    };
    filterer.applyWithIndex(nb_index, is_impure_lambda, setter_lambda, A_FUNCINFO);
    filterer.nbOutputElement();
  }

  info(4) << "BUILD_PART_DATA_FOR_COMPONENT c=" << m_component->name()
          << " nb_pure=" << pure_indexes.size()
          << " nb_impure=" << impure_indexes.size();

  _notifyValueIndexesChanged(&queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentPartData::
checkValid() const
{
  info(4) << "CHECK_VALID_COMPONENT_PART_DATA c=" << m_component->name();
  ValueChecker vc(A_FUNCINFO);
  Integer nb_error = 0;
  for (Integer i = 0; i < 2; ++i) {
    Int32 var_idx = (i == 0) ? 0 : m_impure_var_idx;
    Int32ConstArrayView indexes = m_value_indexes[i];
    Int32ConstArrayView item_indexes = m_items_internal_indexes[i];
    Integer nb_item = indexes.size();
    vc.areEqual(nb_item, item_indexes.size(), "Indexes size");
    for (Integer k = 0; k < nb_item; ++k) {
      MatVarIndex mvi(var_idx, indexes[k]);
      MatVarIndex component_mvi = m_constituent_list_view._matVarIndex(item_indexes[k]);
      if (mvi != component_mvi) {
        info() << "Bad MatVarIndex i=" << i << " k=" << k
               << " mvi=" << mvi << " component_mvi=" << component_mvi;
        ++nb_error;
      }
    }
  }
  if (nb_error != 0)
    ARCANE_FATAL("Bad component part data nb_error={0}", nb_error);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentPurePartItemVectorView MeshComponentPartData::
pureView() const
{
  Int32ConstArrayView value_indexes = valueIndexes(eMatPart::Pure);
  Int32ConstArrayView item_indexes = itemIndexes(eMatPart::Pure);
  return { m_component, value_indexes,
           item_indexes, m_constituent_list_view };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentImpurePartItemVectorView MeshComponentPartData::
impureView() const
{
  Int32ConstArrayView value_indexes = valueIndexes(eMatPart::Impure);
  Int32ConstArrayView item_indexes = itemIndexes(eMatPart::Impure);
  Int32 var_idx = impureVarIdx();
  return { m_component, var_idx, value_indexes,
           item_indexes, m_constituent_list_view };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentPartItemVectorView MeshComponentPartData::
partView(eMatPart part) const
{
  Int32ConstArrayView value_indexes = valueIndexes(part);
  Int32ConstArrayView item_indexes = itemIndexes(part);
  Int32 var_idx = (part == eMatPart::Pure) ? 0 : impureVarIdx();
  return { m_component, var_idx, value_indexes,
           item_indexes, m_constituent_list_view, part };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
