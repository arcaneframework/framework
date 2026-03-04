// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MachineShMemWinVariable.cc                                  (C) 2000-2026 */
/*                                                                           */
/* Classes permettant d'exploiter l'objet MachineShMemWinVariable pointé de  */
/* la zone mémoire des variables en mémoire partagée.                        */
/*---------------------------------------------------------------------------*/

#include "arcane/core/MachineShMemWinVariable.h"

#include "arcane/core/VariableRefArray.h"
#include "arcane/core/VariableRefArray2.h"
#include "arcane/core/MeshVariable.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/MeshVariableScalarRef.h"
#include "internal/MachineShMemWinVariableBase.h"

#include "arcane/utils/NumericTypes.h"
#include "arcane/utils/MDSpan.h"

#include "arccore/base/Span2.h"
#include "arccore/base/MDIndex.h"

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
, m_vart(var)
{
  updateVariable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
MachineShMemWinVariableArrayT<DataType>::
~MachineShMemWinVariableArrayT() = default;

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
  m_base->updateVariable(m_vart.asArray().size(), sizeof(DataType));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
MachineShMemWinMeshVariableScalarT<ItemType, DataType>::
MachineShMemWinMeshVariableScalarT(MeshVariableScalarRefT<ItemType, DataType> var)
: MachineShMemWinVariableCommon(var.variable())
, m_vart(var)
{
  updateVariable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
MachineShMemWinMeshVariableScalarT<ItemType, DataType>::
~MachineShMemWinMeshVariableScalarT() = default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
Span<DataType> MachineShMemWinMeshVariableScalarT<ItemType, DataType>::
view(Int32 rank) const
{
  return asSpan<DataType>(m_base->segmentView(rank));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
DataType MachineShMemWinMeshVariableScalarT<ItemType, DataType>::
operator()(Int32 rank, Int32 notlocal_id)
{
  return this->view(rank)[notlocal_id];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
void MachineShMemWinMeshVariableScalarT<ItemType, DataType>::
updateVariable()
{
  m_base->updateVariable(m_vart.asArray().size(), sizeof(DataType));
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
MachineShMemWinMeshVariableArrayT<ItemType, DataType>::
MachineShMemWinMeshVariableArrayT(MeshVariableArrayRefT<ItemType, DataType> var)
: m_base(makeRef(new MachineShMemWinVariableMDBase(var.variable())))
, m_vart(var)
{
  updateVariable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
MachineShMemWinMeshVariableArrayT<ItemType, DataType>::
~MachineShMemWinMeshVariableArrayT() = default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
ConstArrayView<Int32> MachineShMemWinMeshVariableArrayT<ItemType, DataType>::
machineRanks() const
{
  return m_base->machineRanks();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
void MachineShMemWinMeshVariableArrayT<ItemType, DataType>::
barrier() const
{
  m_base->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
Span2<DataType> MachineShMemWinMeshVariableArrayT<ItemType, DataType>::
view(Int32 rank) const
{
  Span<DataType> span1 = asSpan<DataType>(m_base->segmentView(rank));
  return { span1.data(), m_nb_elem_dim1[rank], m_nb_elem_dim2 };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
Span<DataType> MachineShMemWinMeshVariableArrayT<ItemType, DataType>::
operator()(Int32 rank, Int32 notlocal_id)
{
  Span<DataType> span1 = asSpan<DataType>(m_base->segmentView(rank));
  Span2<DataType> span2(span1.data(), m_nb_elem_dim1[rank], m_nb_elem_dim2);

  return span2[notlocal_id];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
void MachineShMemWinMeshVariableArrayT<ItemType, DataType>::
updateVariable()
{
  Int64 size_dim1 = m_vart.asArray().dim1Size();
  m_nb_elem_dim2 = m_vart.asArray().dim2Size();

  m_base->updateVariable(size_dim1, m_nb_elem_dim2, sizeof(DataType));

  m_nb_elem_dim1 = m_base->nbElemDim1();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType, class Extents>
MachineShMemWinMDVariableT<ItemType, DataType, Extents>::
MachineShMemWinMDVariableT(MeshVariableArrayRefT<ItemType, DataType> var)
: m_base(makeRef(new MachineShMemWinVariableMDBase(var.variable())))
, m_vart(var)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType, class Extents>
MachineShMemWinMDVariableT<ItemType, DataType, Extents>::
~MachineShMemWinMDVariableT() = default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType, class Extents>
ConstArrayView<Int32> MachineShMemWinMDVariableT<ItemType, DataType, Extents>::
machineRanks() const
{
  return m_base->machineRanks();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType, class Extents>
void MachineShMemWinMDVariableT<ItemType, DataType, Extents>::
barrier() const
{
  m_base->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType, class Extents>
MDSpan<DataType, typename MDDimType<Extents::rank() + 1>::DimType> MachineShMemWinMDVariableT<ItemType, DataType, Extents>::
view(Int32 rank) const
{
  Span<DataType> span1 = asSpan<DataType>(m_base->segmentView(rank));

  if constexpr (Extents::rank() == 1) {
    std::array<Int32, 2> nb_elem_mdim{ static_cast<Int32>(m_nb_elem_dim1[rank]), m_shape_dim2[0] };
    MDSpan<DataType, MDDim2> mdspan(span1.data(), MDIndex<2>(nb_elem_mdim));
    return mdspan;
  }
  else if constexpr (Extents::rank() == 2) {
    std::array<Int32, 3> nb_elem_mdim{ static_cast<Int32>(m_nb_elem_dim1[rank]), m_shape_dim2[0], m_shape_dim2[1] };
    MDSpan<DataType, MDDim3> mdspan(span1.data(), MDIndex<3>(nb_elem_mdim));
    return mdspan;
  }
  else if constexpr (Extents::rank() == 3) {
    std::array<Int32, 4> nb_elem_mdim{ static_cast<Int32>(m_nb_elem_dim1[rank]), m_shape_dim2[0], m_shape_dim2[1], m_shape_dim2[2] };
    MDSpan<DataType, MDDim4> mdspan(span1.data(), MDIndex<4>(nb_elem_mdim));
    return mdspan;
  }
  ARCANE_FATAL("Unexpected dimension");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType, class Extents>
MDSpan<DataType, Extents> MachineShMemWinMDVariableT<ItemType, DataType, Extents>::
operator()(Int32 rank, Int32 notlocal_id)
{
  Span<DataType> span1 = asSpan<DataType>(m_base->segmentView(rank));
  Span2<DataType> span2(span1.data(), m_nb_elem_dim1[rank], m_nb_elem_dim2);
  MDSpan<DataType, Extents> mdspan(span2[notlocal_id].data(), MDIndex<Extents::rank()>(m_shape_dim2));

  return mdspan;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType, class Extents>
void MachineShMemWinMDVariableT<ItemType, DataType, Extents>::
updateVariable()
{
  Int64 nb_elem_dim1 = m_vart.asArray().dim1Size();
  Int32 nb_elem_dim2 = m_vart.asArray().dim2Size();

  m_base->updateVariable(nb_elem_dim1, nb_elem_dim2, sizeof(DataType));

  SmallSpan<Int32> shape_dim2_view(m_shape_dim2.data(), Extents::rank());
  shape_dim2_view.copy(m_base->arrayShape().dimensions());

  m_nb_elem_dim1 = m_base->nbElemDim1();
  m_nb_elem_dim2 = nb_elem_dim2;
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

ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE_WITH_ITEM(MachineShMemWinMeshVariableScalarT);
ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE_WITH_ITEM(MachineShMemWinMeshVariableArrayT);

template class ARCANE_TEMPLATE_EXPORT MachineShMemWinMDVariableT<Cell, Real, MDDim1>;
template class ARCANE_TEMPLATE_EXPORT MachineShMemWinMDVariableT<Cell, Real, MDDim2>;
template class ARCANE_TEMPLATE_EXPORT MachineShMemWinMDVariableT<Cell, Real, MDDim3>;

template class ARCANE_TEMPLATE_EXPORT MachineShMemWinMeshMDVariableT<Cell, Real, MDDim1>;
template class ARCANE_TEMPLATE_EXPORT MachineShMemWinMeshMDVariableT<Cell, Real, MDDim2>;
template class ARCANE_TEMPLATE_EXPORT MachineShMemWinMeshMDVariableT<Cell, Real, MDDim3>;

template class ARCANE_TEMPLATE_EXPORT MachineShMemWinMeshVectorMDVariableT<Cell, Real, 3, MDDim1>;
template class ARCANE_TEMPLATE_EXPORT MachineShMemWinMeshVectorMDVariableT<Cell, Real, 3, MDDim2>;

template class ARCANE_TEMPLATE_EXPORT MachineShMemWinMeshMatrixMDVariableT<Cell, Real, 4, 5, MDDim1>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
