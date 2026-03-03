// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MachineShMemWinVariableBase.h                               (C) 2000-2026 */
/*                                                                           */
/* Allocateur mémoire utilisant la classe MachineShMemWinBase.               */
/*---------------------------------------------------------------------------*/

#include "arcane/core/MachineShMemWinVariable.h"

#include "arcane/core/VariableRefArray.h"
#include "arcane/core/VariableRefArray2.h"
#include "arcane/core/MeshVariable.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/MeshVariableScalarRef.h"
#include "arcane/core/internal/MachineShMemWinVariableBase.h"

#include "arcane/utils/NumericTypes.h"

#include "arccore/base/Span2.h"


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MachineShMemWinVariableCommon::
MachineShMemWinVariableCommon(IVariable* var)
: m_base(makeRef(new MachineShMemWinVariableBase(var)))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MachineShMemWinVariableCommon::
~MachineShMemWinVariableCommon()
= default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> MachineShMemWinVariableCommon::
machineRanks() const
{
  return m_base->machineRanks();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinVariableCommon::
barrier() const
{
  m_base->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
MachineShMemWinVariableArrayT<DataType>::
MachineShMemWinVariableArrayT(VariableRefArrayT<DataType> var)
: MachineShMemWinVariableCommon(var.variable())
{
  updateVariable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
Span<DataType> MachineShMemWinVariableArrayT<DataType>::
view(Int32 rank) const
{
  return asSpan<DataType>(m_base->segmentView(rank));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
void MachineShMemWinVariableArrayT<DataType>::
updateVariable()
{
  m_base->updateVariable(m_base->variable()->nbElement(), sizeof(DataType));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
MachineShMemWinVariableItemT<ItemType, DataType>::
MachineShMemWinVariableItemT(MeshVariableScalarRefT<ItemType, DataType> var)
: MachineShMemWinVariableCommon(var.variable())
{
  updateVariable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
Span<DataType> MachineShMemWinVariableItemT<ItemType, DataType>::
view(Int32 rank) const
{
  return asSpan<DataType>(m_base->segmentView(rank));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
DataType MachineShMemWinVariableItemT<ItemType, DataType>::
operator()(Int32 rank, Int32 notlocal_id)
{
  return this->view(rank)[notlocal_id];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
void MachineShMemWinVariableItemT<ItemType, DataType>::
updateVariable()
{
  m_base->updateVariable(m_base->variable()->nbElement(), sizeof(DataType));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
MachineShMemWinVariableArray2T<DataType>::
MachineShMemWinVariableArray2T(VariableRefArray2T<DataType> var)
: m_base(makeRef(new MachineShMemWinVariable2DBase(var.variable())))
, m_vart(var)
{
  updateVariable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
MachineShMemWinVariableArray2T<DataType>::
~MachineShMemWinVariableArray2T() = default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
ConstArrayView<Int32> MachineShMemWinVariableArray2T<DataType>::
machineRanks() const
{
  return m_base->machineRanks();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
void MachineShMemWinVariableArray2T<DataType>::
barrier() const
{
  m_base->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
Span2<DataType> MachineShMemWinVariableArray2T<DataType>::
view(Int32 rank) const
{
  Span<DataType> span1 = asSpan<DataType>(m_base->segmentView(rank));
  return { span1.data(), m_nb_elem_dim1[rank], m_nb_elem_dim2[rank] };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
void MachineShMemWinVariableArray2T<DataType>::
updateVariable()
{
  Int64 size_dim1 = m_vart.dim1Size();
  Int64 size_dim2 = m_vart.dim2Size();

  m_base->updateVariable(size_dim1, size_dim2, sizeof(DataType));

  m_nb_elem_dim1 = m_base->nbElemDim1();
  m_nb_elem_dim2 = m_base->nbElemDim2();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
MachineShMemWinVariableItemArrayT<ItemType, DataType>::
MachineShMemWinVariableItemArrayT(MeshVariableArrayRefT<ItemType, DataType> var)
: m_base(makeRef(new MachineShMemWinVariableMDBase<1>(var.variable())))
, m_vart(var)
{
  updateVariable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
MachineShMemWinVariableItemArrayT<ItemType, DataType>::
~MachineShMemWinVariableItemArrayT() = default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
ConstArrayView<Int32> MachineShMemWinVariableItemArrayT<ItemType, DataType>::
machineRanks() const
{
  return m_base->machineRanks();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
void MachineShMemWinVariableItemArrayT<ItemType, DataType>::
barrier() const
{
  m_base->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
Span2<DataType> MachineShMemWinVariableItemArrayT<ItemType, DataType>::
view(Int32 rank) const
{
  Span<DataType> span1 = asSpan<DataType>(m_base->segmentView(rank));
  return { span1.data(), m_nb_elem_dim1[rank], m_nb_elem_dim2 };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
Span<DataType> MachineShMemWinVariableItemArrayT<ItemType, DataType>::
operator()(Int32 rank, Int32 notlocal_id)
{
  Span<DataType> span1 = asSpan<DataType>(m_base->segmentView(rank));
  Span2<DataType> span2(span1.data(), m_nb_elem_dim1[rank], m_nb_elem_dim2);

  return span2[notlocal_id];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
void MachineShMemWinVariableItemArrayT<ItemType, DataType>::
updateVariable()
{
  Int64 size_dim1 = m_vart.asArray().dim1Size() * sizeof(DataType);
  m_nb_elem_dim2 = m_vart.asArray().dim2Size() * sizeof(DataType);

  SmallSpan<Int64, 1> size_dim2(&m_nb_elem_dim2);

  m_base->updateVariable(size_dim1, size_dim2, sizeof(DataType));

  m_nb_elem_dim1 = m_base->nbElemDim1();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Macro pour instantier une classe template pour tous les types numériques, tous les items.
#define ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE_WITH_ITEM1(class_name, item) \
  template class ARCANE_TEMPLATE_EXPORT class_name<item, Real>; \
  template class ARCANE_TEMPLATE_EXPORT class_name<item, Real3>; \
  template class ARCANE_TEMPLATE_EXPORT class_name<item, Real3x3>; \
  template class ARCANE_TEMPLATE_EXPORT class_name<item, Real2>; \
  template class ARCANE_TEMPLATE_EXPORT class_name<item, Real2x2>; \
  template class ARCANE_TEMPLATE_EXPORT class_name<item, Int8>; \
  template class ARCANE_TEMPLATE_EXPORT class_name<item, Int16>; \
  template class ARCANE_TEMPLATE_EXPORT class_name<item, Int32>; \
  template class ARCANE_TEMPLATE_EXPORT class_name<item, Int64>;

#define ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE_WITH_ITEM(class_name) \
  ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE_WITH_ITEM1(class_name, Node) \
  ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE_WITH_ITEM1(class_name, Face) \
  ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE_WITH_ITEM1(class_name, Cell) \
  ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE_WITH_ITEM1(class_name, Particle) \
  ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE_WITH_ITEM1(class_name, DoF)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE(MachineShMemWinVariableArrayT);

template class ARCANE_TEMPLATE_EXPORT MachineShMemWinVariableArray2T<Real>;
template class ARCANE_TEMPLATE_EXPORT MachineShMemWinVariableArray2T<Real3>;
template class ARCANE_TEMPLATE_EXPORT MachineShMemWinVariableArray2T<Real3x3>;
template class ARCANE_TEMPLATE_EXPORT MachineShMemWinVariableArray2T<Real2>;
template class ARCANE_TEMPLATE_EXPORT MachineShMemWinVariableArray2T<Real2x2>;
template class ARCANE_TEMPLATE_EXPORT MachineShMemWinVariableArray2T<Int16>;
template class ARCANE_TEMPLATE_EXPORT MachineShMemWinVariableArray2T<Int32>;
template class ARCANE_TEMPLATE_EXPORT MachineShMemWinVariableArray2T<Int64>;
template class ARCANE_TEMPLATE_EXPORT MachineShMemWinVariableArray2T<Byte>;

ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE_WITH_ITEM(MachineShMemWinVariableItemT);
ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE_WITH_ITEM(MachineShMemWinVariableItemArrayT);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
