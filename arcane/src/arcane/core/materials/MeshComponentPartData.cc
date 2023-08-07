// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshComponentPartData.cc                                    (C) 2000-2023 */
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
#include "arcane/core/materials/MeshComponentPartData.h"
#include "arcane/core/materials/ComponentItemInternal.h"
#include "arcane/core/materials/ComponentPartItemVectorView.h"
#include "arcane/core/materials/internal/IMeshComponentInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshComponentPartData::
MeshComponentPartData(IMeshComponent* component)
: TraceAccessor(component->traceMng())
, m_component(component)
, m_impure_var_idx(component->_internalApi()->variableIndexerIndex()+1)
{
  // Utilise l'allocateur des données pour permettre d'accéder à ces valeurs
  // sur les accélérateurs
  IMemoryAllocator* allocator = platform::getDefaultDataAllocator();
  for( Integer i=0; i<2; ++i ){
    m_value_indexes[i] = UniqueArray<Int32>(allocator);
    m_items_internal_indexes[i] = UniqueArray<Int32>(allocator);
 }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshComponentPartData::
~MeshComponentPartData()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentPartData::
_notifyValueIndexesChanged()
{
  applySimdPadding(m_value_indexes[0]);
  applySimdPadding(m_value_indexes[1]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentPartData::
_setFromMatVarIndexes(ConstArrayView<MatVarIndex> matvar_indexes)
{
  Int32Array& pure_indexes = m_value_indexes[(Int32)eMatPart::Pure];
  Int32Array& impure_indexes = m_value_indexes[(Int32)eMatPart::Impure];

  Int32Array& pure_internal_indexes = m_items_internal_indexes[(Int32)eMatPart::Pure];
  Int32Array& impure_internal_indexes = m_items_internal_indexes[(Int32)eMatPart::Impure];

  pure_indexes.clear();
  impure_indexes.clear();

  pure_internal_indexes.clear();
  impure_internal_indexes.clear();

  info(4) << "BEGIN_BUILD_PART_DATA_FOR_COMPONENT c=" << m_component->name();

  for( Integer i=0, n=matvar_indexes.size(); i<n; ++i ){
    MatVarIndex mvi = matvar_indexes[i];
    if (mvi.arrayIndex()==0){
      pure_indexes.add(mvi.valueIndex());
      pure_internal_indexes.add(i);
    }
    else{
      impure_indexes.add(mvi.valueIndex());
      impure_internal_indexes.add(i);
    }
  }
  info(4) << "BUILD_PART_DATA_FOR_COMPONENT c=" << m_component->name()
          << " nb_pure=" << pure_indexes.size()
          << " nb_impure=" << impure_indexes.size();

  _notifyValueIndexesChanged();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentPartData::
checkValid() const
{
  info(4) << "CHECK_VALID_COMPONENT_PART_DATA c=" << m_component->name();
  ValueChecker vc(A_FUNCINFO);
  Integer nb_error = 0;
  for( Integer i=0; i<2; ++i ){
    Int32 var_idx = (i==0) ? 0 : m_impure_var_idx;
    Int32ConstArrayView indexes = m_value_indexes[i];
    Int32ConstArrayView item_indexes = m_items_internal_indexes[i];
    Integer nb_item = indexes.size();
    vc.areEqual(nb_item,item_indexes.size(),"Indexes size");
    for( Integer k=0; k<nb_item; ++k ){
      MatVarIndex mvi(var_idx,indexes[k]);
      MatVarIndex component_mvi = m_items_internal[item_indexes[k]]->variableIndex();
      if (mvi!=component_mvi){
        info() << "Bad MatVarIndex i=" << i << " k=" << k
               << " mvi=" << mvi << " component_mvi=" << component_mvi;
        ++nb_error;
      }
    }
  }
  if (nb_error!=0)
    ARCANE_FATAL("Bad component part data nb_error={0}",nb_error);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentPurePartItemVectorView MeshComponentPartData::
pureView() const
{
  Int32ConstArrayView value_indexes = valueIndexes(eMatPart::Pure);
  Int32ConstArrayView item_indexes = itemIndexes(eMatPart::Pure);
  return ComponentPurePartItemVectorView(m_component,value_indexes,
                                         item_indexes,m_items_internal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentImpurePartItemVectorView MeshComponentPartData::
impureView() const
{
  Int32ConstArrayView value_indexes = valueIndexes(eMatPart::Impure);
  Int32ConstArrayView item_indexes = itemIndexes(eMatPart::Impure);
  Int32 var_idx = impureVarIdx();
  return ComponentImpurePartItemVectorView(m_component,var_idx,value_indexes,
                                           item_indexes,m_items_internal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentPartItemVectorView MeshComponentPartData::
partView(eMatPart part) const
{
  Int32ConstArrayView value_indexes = valueIndexes(part);
  Int32ConstArrayView item_indexes = itemIndexes(part);
  Int32 var_idx = (part==eMatPart::Pure) ? 0 : impureVarIdx();
  return ComponentPartItemVectorView(m_component,var_idx,value_indexes,
                                     item_indexes,m_items_internal,part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
