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
#include "arcane/utils/NumericTypes.h"

#include "arccore/base/Span2.h"

#include "arcane/core/VariableRefArray.h"
#include "arcane/core/VariableRefArray2.h"
#include "arcane/core/MeshVariable.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/MeshVariableScalarRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MachineShMemWinVariableCommon::
MachineShMemWinVariableCommon(IVariable* var, Int64 sizeof_type)
: m_base(var, sizeof_type)
{}

MachineShMemWinVariableCommon::
~MachineShMemWinVariableCommon()
{
}

ConstArrayView<Int32> MachineShMemWinVariableCommon::
machineRanks() const
{
  return m_base.machineRanks();
}

void MachineShMemWinVariableCommon::
barrier() const
{
  m_base.barrier();
}

template <class DataType> MachineShMemWinVariableArrayT<DataType>::
MachineShMemWinVariableArrayT(VariableRefArrayT<DataType> var)
: MachineShMemWinVariableCommon(var.variable(), sizeof(DataType))
{
  updateVariable();
}

template <class DataType> Span<DataType>
MachineShMemWinVariableArrayT<DataType>::
segmentView() const
{
  return asSpan<DataType>(m_base.segmentView());
}

template <class DataType> Span<DataType>
MachineShMemWinVariableArrayT<DataType>::
segmentView(Int32 rank) const
{
  return asSpan<DataType>(m_base.segmentView(rank));
}

template <class DataType>
void MachineShMemWinVariableArrayT<DataType>::
updateVariable()
{
  m_base.updateVariable(m_base.variable()->nbElement());
}

template <class ItemType, class DataType>
MachineShMemWinVariableItemT<ItemType, DataType>::
MachineShMemWinVariableItemT(MeshVariableScalarRefT<ItemType, DataType> var)
: MachineShMemWinVariableCommon(var.variable(), sizeof(DataType))
{
  updateVariable();
}

template <class ItemType, class DataType>
Span<DataType> MachineShMemWinVariableItemT<ItemType, DataType>::
segmentView() const
{
  return asSpan<DataType>(m_base.segmentView());
}

template <class ItemType, class DataType>
Span<DataType> MachineShMemWinVariableItemT<ItemType, DataType>::
segmentView(Int32 rank) const
{
  return asSpan<DataType>(m_base.segmentView(rank));
}

template <class ItemType, class DataType>
DataType MachineShMemWinVariableItemT<ItemType, DataType>::
operator()(Int32 local_id)
{
  return this->segmentView()[local_id];
}

template <class ItemType, class DataType>
DataType MachineShMemWinVariableItemT<ItemType, DataType>::
operator()(Int32 rank, Int32 local_id)
{
  return this->segmentView(rank)[local_id];
}

template <class ItemType, class DataType>
void MachineShMemWinVariableItemT<ItemType, DataType>::
updateVariable()
{
  m_base.updateVariable(m_base.variable()->nbElement());
}

template <class DataType>
MachineShMemWinVariableArray2T<DataType>::
MachineShMemWinVariableArray2T(VariableRefArray2T<DataType> var)
: m_base(var.variable(), sizeof(DataType))
, m_size_dim1(var.dim1Size())
, m_size_dim2(var.dim2Size())
, m_vart(var)
{
  updateVariable();
}

template <class DataType>
MachineShMemWinVariableArray2T<DataType>::
~MachineShMemWinVariableArray2T() = default;

template <class DataType> ConstArrayView<Int32>
MachineShMemWinVariableArray2T<DataType>::
machineRanks() const
{
  return m_base.machineRanks();
}

template <class DataType>
void MachineShMemWinVariableArray2T<DataType>::
barrier() const
{
  m_base.barrier();
}

template <class DataType>
Span2<DataType> MachineShMemWinVariableArray2T<DataType>::
segmentView() const
{
  Span<DataType> span1 = asSpan<DataType>(m_base.segmentView());
  return { span1.data(), m_size_dim1, m_size_dim2 };
}

template <class DataType>
Span2<DataType> MachineShMemWinVariableArray2T<DataType>::
segmentView(Int32 rank) const
{
  Span<DataType> span1 = asSpan<DataType>(m_base.segmentView(rank));
  return { span1.data(), m_size_dim1, m_size_dim2 };
}

template <class DataType>
void MachineShMemWinVariableArray2T<DataType>::
updateVariable()
{
  m_size_dim1 = m_vart.dim1Size();
  m_size_dim2 = m_vart.dim2Size();

  m_base.updateVariable(m_size_dim1, m_size_dim2);
}

template <class ItemType, class DataType>
MachineShMemWinVariableItemArrayT<ItemType, DataType>::
MachineShMemWinVariableItemArrayT(MeshVariableArrayRefT<ItemType, DataType> var)
: m_base(var.variable(), sizeof(DataType))
, m_size_dim1(var.asArray().dim1Size())
, m_size_dim2(var.asArray().dim2Size())
, m_vart(var)
{
  updateVariable();
}

template <class ItemType, class DataType>
MachineShMemWinVariableItemArrayT<ItemType, DataType>::
~MachineShMemWinVariableItemArrayT() = default;

template <class ItemType, class DataType> ConstArrayView<Int32>
MachineShMemWinVariableItemArrayT<ItemType, DataType>::
machineRanks() const
{
  return m_base.machineRanks();
}

template <class ItemType, class DataType>
void MachineShMemWinVariableItemArrayT<ItemType, DataType>::
barrier() const
{
  m_base.barrier();
}

template <class ItemType, class DataType> Span2<DataType>
MachineShMemWinVariableItemArrayT<ItemType, DataType>::
segmentView() const
{
  Span<DataType> span1 = asSpan<DataType>(m_base.segmentView());
  return { span1.data(), m_size_dim1, m_size_dim2 };
}

template <class ItemType, class DataType> Span2<DataType>
MachineShMemWinVariableItemArrayT<ItemType, DataType>::
segmentView(Int32 rank) const
{
  Span<DataType> span1 = asSpan<DataType>(m_base.segmentView(rank));
  return { span1.data(), m_size_dim1, m_size_dim2 };
}

template <class ItemType, class DataType> Span<DataType>
MachineShMemWinVariableItemArrayT<ItemType, DataType>::
segmentView1D() const
{
  return asSpan<DataType>(m_base.segmentView());
}

template <class ItemType, class DataType> Span<DataType>
MachineShMemWinVariableItemArrayT<ItemType, DataType>::
segmentView1D(Int32 rank) const
{
  return asSpan<DataType>(m_base.segmentView(rank));
}

template <class ItemType, class DataType> Span<DataType>
MachineShMemWinVariableItemArrayT<ItemType, DataType>::
operator()(Int32 local_id)
{
  Span<DataType> span1 = asSpan<DataType>(m_base.segmentView());
  Span2<DataType> span2(span1.data(), m_size_dim1, m_size_dim2);

  return span2[local_id];
}

template <class ItemType, class DataType>
Span<DataType> MachineShMemWinVariableItemArrayT<ItemType, DataType>::
operator()(Int32 rank, Int32 local_id)
{
  Span<DataType> span1 = asSpan<DataType>(m_base.segmentView(rank));
  Span2<DataType> span2(span1.data(), m_size_dim1, m_size_dim2);

  return span2[local_id];
}

template <class ItemType, class DataType>
void MachineShMemWinVariableItemArrayT<ItemType, DataType>::
updateVariable()
{
  m_size_dim1 = m_vart.asArray().dim1Size();
  m_size_dim2 = m_vart.asArray().dim2Size();

  SmallSpan<Int64, 1> aaa(&m_size_dim2);

  m_base.updateVariable(m_size_dim1, aaa);
}

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
