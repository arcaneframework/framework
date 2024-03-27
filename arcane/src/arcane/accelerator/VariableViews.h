﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableViews.h                                             (C) 2000-2024 */
/*                                                                           */
/* Gestion des vues sur les variables pour les accélérateurs.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_VARIABLEVIEWS_H
#define ARCANE_ACCELERATOR_VARIABLEVIEWS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneCxx20.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/SimdItem.h"
#include "arcane/core/ItemLocalId.h"
#include "arcane/core/VariableTypedef.h"

#include "arcane/accelerator/AcceleratorGlobal.h"
#include "arcane/accelerator/ViewsCommon.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: Faire les vues en ReadWrite pour les accesseurs SIMD

namespace Arcane::Accelerator
{
template <typename DataType> class View1DGetterSetter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des vues sur les variables.
 */
class ARCANE_ACCELERATOR_EXPORT VariableViewBase
{
 public:

  VariableViewBase(const ViewBuildInfo& command, IVariable* var);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour accéder à un tableau 1D d'une vue en lecture/écriture.
 */
template <typename DataType>
class View1DSetter
{
  // Pour accéder à m_data;
  friend class View1DGetterSetter<DataType>;

 public:

  using ValueType = DataType;
  using DataTypeReturnReference = View1DSetter<DataType>;
  explicit ARCCORE_HOST_DEVICE View1DSetter(SmallSpan<DataType> data)
  : m_data(data)
  {}
  ARCCORE_HOST_DEVICE DataViewSetter<DataType> operator[](Int32 index) const
  {
    return DataViewSetter<DataType>(m_data.ptrAt(index));
  }
  ARCCORE_HOST_DEVICE DataViewSetter<DataType> operator()(Int32 index) const
  {
    return DataViewSetter<DataType>(m_data.ptrAt(index));
  }
  DataTypeReturnReference& operator=(const View1DSetter<DataType>& rhs) = delete;

 public:

  ARCCORE_HOST_DEVICE void copy(SmallSpan<const DataType> rhs)
  {
    m_data.copy(rhs);
  }

 private:

  SmallSpan<DataType> m_data;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour accéder à un tableau 1D d'une vue en lecture/écriture.
 */
template <typename DataType>
class View1DGetterSetter
: public View1DSetter<DataType>
{
  using View1DSetter<DataType>::m_data;

 public:

  using ValueType = DataType;
  using DataTypeReturnReference = View1DGetterSetter<DataType>;

 public:

  explicit ARCCORE_HOST_DEVICE View1DGetterSetter(SmallSpan<DataType> data)
  : View1DSetter<DataType>(data)
  {}
  DataTypeReturnReference& operator=(const View1DGetterSetter<DataType>& rhs) = delete;

 public:

  ARCCORE_HOST_DEVICE DataViewGetterSetter<DataType> operator[](Int32 index) const
  {
    return DataViewGetterSetter<DataType>(m_data.ptrAt(index));
  }

 public:

  ARCCORE_HOST_DEVICE SmallSpan<DataType> value() const { return m_data; }
  ARCCORE_HOST_DEVICE operator SmallSpan<DataType>() { return m_data; }
  ARCCORE_HOST_DEVICE operator SmallSpan<const DataType>() const { return m_data; }
  ARCCORE_HOST_DEVICE operator Span<DataType>() { return { m_data.data(), m_data.size() }; }
  ARCCORE_HOST_DEVICE operator Span<const DataType>() const { return { m_data.data(), m_data.size() }; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture sur une variable scalaire du maillage.
 */
template <typename _ItemType, typename _Accessor, typename _IndexerType, bool _HasSimd>
class ItemVariableScalarOutViewBaseT
: public VariableViewBase
{
  using Accessor = _Accessor;

 public:

  using ItemType = _ItemType;
  using IndexerType = _IndexerType;
  using DataType = typename _Accessor::ValueType;
  using DataTypeReturnReference = DataType&;

 public:

  ItemVariableScalarOutViewBaseT(const ViewBuildInfo& command, IVariable* var, SmallSpan<DataType> v)
  : VariableViewBase(command, var)
  , m_values(v.data())
  , m_size(v.size())
  {}

  //! Opérateur d'accès vectoriel avec indirection.
  SimdSetter<DataType> operator[](SimdItemIndexT<ItemType> simd_item) const requires(_HasSimd)
  {
    return SimdSetter<DataType>(m_values, simd_item.simdLocalIds());
  }

  //! Opérateur d'accès vectoriel sans indirection.
  SimdDirectSetter<DataType> operator[](SimdItemDirectIndexT<ItemType> simd_item) const requires(_HasSimd)
  {
    return SimdDirectSetter<DataType>(m_values + simd_item.baseLocalId());
  }

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE Accessor operator[](IndexerType item) const
  {
    ARCANE_CHECK_AT(item.asInt32(), m_size);
    return Accessor(this->m_values + item.asInt32());
  }

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE Accessor operator()(IndexerType item) const
  {
    ARCANE_CHECK_AT(item.asInt32(), m_size);
    return Accessor(this->m_values + item.asInt32());
  }

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE Accessor value(IndexerType item) const
  {
    ARCANE_CHECK_AT(item.asInt32(), m_size);
    return Accessor(this->m_values + item.asInt32());
  }

  //! Positionne la valeur pour l'entité \a item à \a v
  ARCCORE_HOST_DEVICE void setValue(IndexerType item, const DataType& v) const
  {
    ARCANE_CHECK_AT(item.asInt32(), m_size);
    this->m_values[item.asInt32()] = v;
  }

 private:

  DataType* m_values;
  Int32 m_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture sur une variable scalaire du maillage.
 */
template <typename _ItemType, typename _Accessor>
class ItemVariableScalarOutViewT
: public ItemVariableScalarOutViewBaseT<_ItemType, _Accessor, typename ItemTraitsT<_ItemType>::LocalIdType, true>
{
  using BaseClass = ItemVariableScalarOutViewBaseT<_ItemType, _Accessor, typename ItemTraitsT<_ItemType>::LocalIdType, true>;

 public:

  using DataType = typename _Accessor::ValueType;

 public:

  ItemVariableScalarOutViewT(const ViewBuildInfo& command, IVariable* var, SmallSpan<DataType> v)
  : BaseClass(command, var, v)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture sur une variable partielle scalaire du maillage.
 */
template <typename _ItemType, typename _Accessor>
class ItemPartialVariableScalarOutViewT
: public ItemVariableScalarOutViewBaseT<_ItemType, _Accessor, ItemEnumeratorIndexT<_ItemType>, false>
{
  using BaseClass = ItemVariableScalarOutViewBaseT<_ItemType, _Accessor, ItemEnumeratorIndexT<_ItemType>, false>;

 public:

  using DataType = typename _Accessor::ValueType;

 public:

  ItemPartialVariableScalarOutViewT(const ViewBuildInfo& command, IVariable* var, SmallSpan<DataType> v)
  : BaseClass(command, var, v)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture sur une variable scalaire du maillage.
 */
template <typename _ItemType, typename _DataType, typename _IndexerType, bool _HasSimd>
class ItemVariableScalarInViewBaseT
: public VariableViewBase
{
 public:

  using ItemType = _ItemType;
  using DataType = _DataType;
  using IndexerType = _IndexerType;

 public:

  ItemVariableScalarInViewBaseT(const ViewBuildInfo& command, IVariable* var, SmallSpan<const DataType> v)
  : VariableViewBase(command, var)
  , m_values(v)
  {}

  //! Opérateur d'accès vectoriel avec indirection.
  typename SimdTypeTraits<DataType>::SimdType
  operator[](SimdItemIndexT<ItemType> simd_item) const requires(_HasSimd)
  {
    typedef typename SimdTypeTraits<DataType>::SimdType SimdType;
    return SimdType(m_values.data(), simd_item.simdLocalIds());
  }

  //! Opérateur d'accès vectoriel avec indirection.
  typename SimdTypeTraits<DataType>::SimdType
  operator[](SimdItemDirectIndexT<ItemType> simd_item) const requires(_HasSimd)
  {
    typedef typename SimdTypeTraits<DataType>::SimdType SimdType;
    return SimdType(m_values.data() + simd_item.baseLocalId());
  }

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE const DataType& operator[](IndexerType item) const
  {
    return this->m_values[item.asInt32()];
  }

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE const DataType& operator()(IndexerType item) const
  {
    return this->m_values[item.asInt32()];
  }

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE const DataType& value(IndexerType item) const
  {
    return this->m_values[item.asInt32()];
  }

 private:

  SmallSpan<const DataType> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture sur une variable scalaire du maillage.
 */
template <typename _ItemType, typename _DataType>
class ItemVariableScalarInViewT
: public ItemVariableScalarInViewBaseT<_ItemType, _DataType, typename ItemTraitsT<_ItemType>::LocalIdType, true>
{
 private:

  using BaseClass = ItemVariableScalarInViewBaseT<_ItemType, _DataType, typename ItemTraitsT<_ItemType>::LocalIdType, true>;

 public:

  using ItemType = _ItemType;
  using DataType = _DataType;

 public:

  ItemVariableScalarInViewT(const ViewBuildInfo& command, IVariable* var, SmallSpan<const DataType> v)
  : BaseClass(command, var, v)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture sur une variable partielle scalaire du maillage.
 */
template <typename _ItemType, typename _DataType>
class ItemPartialVariableScalarInViewT
: public ItemVariableScalarInViewBaseT<_ItemType, _DataType, ItemEnumeratorIndexT<_ItemType>, false>
{
 private:

  using BaseClass = ItemVariableScalarInViewBaseT<_ItemType, _DataType, ItemEnumeratorIndexT<_ItemType>, false>;

 public:

  using ItemType = _ItemType;
  using DataType = _DataType;

 public:

  ItemPartialVariableScalarInViewT(const ViewBuildInfo& command, IVariable* var, SmallSpan<const DataType> v)
  : BaseClass(command, var, v)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture sur une variable tableau du maillage.
 */
template <typename ItemType, typename DataType, typename _Indexer>
class ItemVariableArrayInViewBaseT
: public VariableViewBase
{
 private:

  using IndexerType = _Indexer;

 public:

  ItemVariableArrayInViewBaseT(const ViewBuildInfo& command, IVariable* var, SmallSpan2<const DataType> v)
  : VariableViewBase(command, var)
  , m_values(v)
  {}

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE const SmallSpan<const DataType> operator[](IndexerType i) const
  {
    return this->m_values[i.asInt32()];
  }

  //! Opérateur d'accès pour la \a i-ème valeur de l'entité \a item
  ARCCORE_HOST_DEVICE const DataType& operator()(IndexerType item, Int32 i) const
  {
    return this->m_values[item.asInt32()][i];
  }

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE const SmallSpan<const DataType> value(IndexerType i) const
  {
    return this->m_values[i.asInt32()];
  }

 private:

  SmallSpan2<const DataType> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture sur une variable tableau du maillage.
 */
template <typename _ItemType, typename _DataType>
class ItemPartialVariableArrayInViewT
: public ItemVariableArrayInViewBaseT<_ItemType, _DataType, ItemEnumeratorIndexT<_ItemType>>
{
 private:

  using BaseClass = ItemVariableArrayInViewBaseT<_ItemType, _DataType, ItemEnumeratorIndexT<_ItemType>>;

 public:

  using DataType = _DataType;

 public:

  ItemPartialVariableArrayInViewT(const ViewBuildInfo& command, IVariable* var, SmallSpan2<const DataType> v)
  : BaseClass(command, var, v)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture sur une variable tableau du maillage.
 */
template <typename _ItemType, typename _DataType>
class ItemVariableArrayInViewT
: public ItemVariableArrayInViewBaseT<_ItemType, _DataType, typename ItemTraitsT<_ItemType>::LocalIdType>
{
 private:

  using BaseClass = ItemVariableArrayInViewBaseT<_ItemType, _DataType, typename ItemTraitsT<_ItemType>::LocalIdType>;

 public:

  using DataType = _DataType;

 public:

  ItemVariableArrayInViewT(const ViewBuildInfo& command, IVariable* var, SmallSpan2<const DataType> v)
  : BaseClass(command, var, v)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture sur une variable tableau du maillage.
 */
template <typename _ItemType, typename _Accessor, typename _Indexer>
class ItemVariableArrayOutViewBaseT
: public VariableViewBase
{
 private:

  using ItemType = _ItemType;
  using Accessor = _Accessor;
  using IndexerType = _Indexer;
  using DataType = typename Accessor::ValueType;
  using DataTypeReturnType = typename Accessor::DataTypeReturnReference;

 public:

  ItemVariableArrayOutViewBaseT(const ViewBuildInfo& command, IVariable* var, SmallSpan2<DataType> v)
  : VariableViewBase(command, var)
  , m_values(v)
  {}

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE DataTypeReturnType operator[](IndexerType item) const
  {
    return DataTypeReturnType(this->m_values[item.asInt32()]);
  }

  //! Opérateur d'accès pour la \a i-ème valeur de l'entité \a item
  ARCCORE_HOST_DEVICE DataType& operator()(IndexerType item, Int32 i) const
  {
    return this->m_values[item.asInt32()][i];
  }

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE SmallSpan<DataType> value(IndexerType item) const
  {
    return DataTypeReturnType(this->m_values[item.asInt32()]);
  }

 private:

  SmallSpan2<DataType> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture sur une variable tableau du maillage.
 */
template <typename _ItemType, typename _Accessor>
class ItemVariableArrayOutViewT
: public ItemVariableArrayOutViewBaseT<_ItemType, _Accessor, typename ItemTraitsT<_ItemType>::LocalIdType>
{
 private:

  using DataType = typename _Accessor::ValueType;
  using BaseClass = ItemVariableArrayOutViewBaseT<_ItemType, _Accessor, typename ItemTraitsT<_ItemType>::LocalIdType>;

 public:

  ItemVariableArrayOutViewT(const ViewBuildInfo& command, IVariable* var, SmallSpan2<DataType> v)
  : BaseClass(command, var, v)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture sur une variable partielle tableau du maillage.
 */
template <typename _ItemType, typename _Accessor>
class ItemPartialVariableArrayOutViewT
: public ItemVariableArrayOutViewBaseT<_ItemType, _Accessor, ItemEnumeratorIndexT<_ItemType>>
{
 private:

  using DataType = typename _Accessor::ValueType;
  using BaseClass = ItemVariableArrayOutViewBaseT<_ItemType, _Accessor, ItemEnumeratorIndexT<_ItemType>>;

 public:

  ItemPartialVariableArrayOutViewT(const ViewBuildInfo& command, IVariable* var, SmallSpan2<DataType> v)
  : BaseClass(command, var, v)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture sur une variable scalaire de type 'RealN' du maillage.
 *
 * Cette classe spécialise les vues modifiable pour les réels 'Real2', 'Real3',
 * 'Real2x2' et 'Real3x3'. La spécialisation s'assure qu'on ne puisse pas
 * modifier uniquement une composante de ces vecteurs de réels. Par exemple:
 *
 * \code
 * VariableNodeReal3View force_view = ...;
 * Node node = ...;
 *
 * // OK:
 * force_view[node] = Real3(x,y,z);
 *
 * // Interdit:
 * // force_view[node].x = ...;
 *
 * \endcode
 */
template <typename _ItemType, typename _Accessor, typename _IndexerType, bool _HasSimd>
class ItemVariableRealNScalarOutViewBaseT
: public VariableViewBase
{
 public:

  using ItemType = _ItemType;
  using Accessor = _Accessor;
  using IndexerType = _IndexerType;
  using DataType = typename _Accessor::ValueType;
  using DataTypeReturnReference = DataType&;

 public:

  //! Construit la vue
  ItemVariableRealNScalarOutViewBaseT(const ViewBuildInfo& command, IVariable* var, SmallSpan<DataType> v)
  : VariableViewBase(command, var)
  , m_values(v.data())
  , m_size(v.size())
  {}

  //! Opérateur d'accès vectoriel avec indirection.
  SimdSetter<DataType> operator[](SimdItemIndexT<ItemType> simd_item) const requires(_HasSimd)
  {
    return SimdSetter<DataType>(m_values, simd_item.simdLocalIds());
  }

  //! Opérateur d'accès vectoriel avec indirection.
  SimdSetter<DataType> operator()(SimdItemIndexT<ItemType> simd_item) const requires(_HasSimd)
  {
    return SimdSetter<DataType>(m_values, simd_item.simdLocalIds());
  }

  //! Opérateur d'accès vectoriel sans indirection.
  SimdDirectSetter<DataType> operator[](SimdItemDirectIndexT<ItemType> simd_item) const requires(_HasSimd)
  {
    return SimdDirectSetter<DataType>(m_values + simd_item.baseLocalId());
  }

  //! Opérateur d'accès vectoriel sans indirection.
  SimdDirectSetter<DataType> operator()(SimdItemDirectIndexT<ItemType> simd_item) const requires(_HasSimd)
  {
    return SimdDirectSetter<DataType>(m_values + simd_item.baseLocalId());
  }

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE Accessor operator[](IndexerType item) const
  {
    ARCANE_CHECK_AT(item.asInt32(), m_size);
    return Accessor(this->m_values + item.asInt32());
  }

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE Accessor operator()(IndexerType item) const
  {
    ARCANE_CHECK_AT(item.asInt32(), m_size);
    return Accessor(this->m_values + item.asInt32());
  }

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE Accessor value(IndexerType item) const
  {
    ARCANE_CHECK_AT(item.asInt32(), m_size);
    return Accessor(this->m_values + item.asInt32());
  }

  //! Positionne la valeur pour l'entité \a item à \a v
  ARCCORE_HOST_DEVICE void setValue(IndexerType item, const DataType& v) const
  {
    ARCANE_CHECK_AT(item.asInt32(), m_size);
    this->m_values[item.asInt32()] = v;
  }

 private:

  DataType* m_values;
  Int32 m_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename _ItemType, typename _Accessor>
class ItemVariableRealNScalarOutViewT
: public ItemVariableRealNScalarOutViewBaseT<_ItemType, _Accessor, typename ItemTraitsT<_ItemType>::LocalIdType, true>
{
 private:

  using BaseClass = ItemVariableRealNScalarOutViewBaseT<_ItemType, _Accessor, typename ItemTraitsT<_ItemType>::LocalIdType, true>;

 public:

  using DataType = typename _Accessor::ValueType;

 public:

  ItemVariableRealNScalarOutViewT(const ViewBuildInfo& command, IVariable* var, SmallSpan<DataType> v)
  : BaseClass(command, var, v)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename _ItemType, typename _Accessor>
class ItemPartialVariableRealNScalarOutViewT
: public ItemVariableRealNScalarOutViewBaseT<_ItemType, _Accessor, ItemEnumeratorIndexT<_ItemType>, false>
{
 private:

  using BaseClass = ItemVariableRealNScalarOutViewBaseT<_ItemType, _Accessor, ItemEnumeratorIndexT<_ItemType>, false>;

 public:

  using DataType = typename _Accessor::ValueType;

 public:

  ItemPartialVariableRealNScalarOutViewT(const ViewBuildInfo& command, IVariable* var, SmallSpan<DataType> v)
  : BaseClass(command, var, v)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture.
 */
template <typename ItemType, typename DataType> auto
viewOut(const ViewBuildInfo& command, MeshVariableScalarRefT<ItemType, DataType>& var)
{
  using Accessor = DataViewSetter<DataType>;
  return ItemVariableScalarOutViewT<ItemType, Accessor>(command, var.variable(), var.asArray());
}

/*!
 * \brief Vue en écriture.
 */
template <typename ItemType, typename DataType> auto
viewOut(const ViewBuildInfo& command, MeshPartialVariableScalarRefT<ItemType, DataType>& var)
{
  using Accessor = DataViewSetter<DataType>;
  return ItemPartialVariableScalarOutViewT<ItemType, Accessor>(command, var.variable(), var.asArray());
}

/*!
 * \brief Vue en écriture.
 */
template <typename ItemType> auto
viewOut(const ViewBuildInfo& command, MeshVariableScalarRefT<ItemType, Real3>& var)
{
  using Accessor = DataViewSetter<Real3>;
  return ItemVariableRealNScalarOutViewT<ItemType, Accessor>(command, var.variable(), var.asArray());
}

/*!
 * \brief Vue en écriture.
 */
template <typename ItemType> auto
viewOut(const ViewBuildInfo& command, MeshPartialVariableScalarRefT<ItemType, Real3>& var)
{
  using Accessor = DataViewSetter<Real3>;
  return ItemPartialVariableRealNScalarOutViewT<ItemType, Accessor>(command, var.variable(), var.asArray());
}

/*!
 * \brief Vue en écriture.
 */
template <typename ItemType> auto
viewOut(const ViewBuildInfo& command, MeshVariableScalarRefT<ItemType, Real2>& var)
{
  using Accessor = DataViewSetter<Real2>;
  return ItemVariableRealNScalarOutViewT<ItemType, Accessor>(command, var.variable(), var.asArray());
}

/*!
 * \brief Vue en écriture.
 */
template <typename ItemType> auto
viewOut(const ViewBuildInfo& command, MeshPartialVariableScalarRefT<ItemType, Real2>& var)
{
  using Accessor = DataViewSetter<Real2>;
  return ItemPartialVariableRealNScalarOutViewT<ItemType, Accessor>(command, var.variable(), var.asArray());
}

/*!
 * \brief Vue en écriture.
 */
template <typename ItemType, typename DataType> auto
viewOut(const ViewBuildInfo& command, MeshVariableArrayRefT<ItemType, DataType>& var)
{
  using Accessor = View1DSetter<DataType>;
  return ItemVariableArrayOutViewT<ItemType, Accessor>(command, var.variable(), var.asArray());
}

/*!
 * \brief Vue en écriture.
 */
template <typename ItemType, typename DataType> auto
viewOut(const ViewBuildInfo& command, MeshPartialVariableArrayRefT<ItemType, DataType>& var)
{
  using Accessor = View1DSetter<DataType>;
  return ItemPartialVariableArrayOutViewT<ItemType, Accessor>(command, var.variable(), var.asArray());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture/écriture.
 */
template <typename ItemType, typename DataType> auto
viewInOut(const ViewBuildInfo& command, MeshVariableScalarRefT<ItemType, DataType>& var)
{
  using Accessor = DataViewGetterSetter<DataType>;
  return ItemVariableScalarOutViewT<ItemType, Accessor>(command, var.variable(), var.asArray());
}

/*!
 * \brief Vue en lecture/écriture.
 */
template <typename ItemType, typename DataType> auto
viewInOut(const ViewBuildInfo& command, MeshPartialVariableScalarRefT<ItemType, DataType>& var)
{
  using Accessor = DataViewGetterSetter<DataType>;
  return ItemPartialVariableScalarOutViewT<ItemType, Accessor>(command, var.variable(), var.asArray());
}

/*!
 * \brief Vue en lecture/écriture.
 */
template <typename ItemType> auto
viewInOut(const ViewBuildInfo& command, MeshVariableScalarRefT<ItemType, Real3>& var)
{
  using Accessor = DataViewGetterSetter<Real3>;
  return ItemVariableRealNScalarOutViewT<ItemType, Accessor>(command, var.variable(), var.asArray());
}

/*!
 * \brief Vue en lecture/écriture.
 */
template <typename ItemType> auto
viewInOut(const ViewBuildInfo& command, MeshPartialVariableScalarRefT<ItemType, Real3>& var)
{
  using Accessor = DataViewGetterSetter<Real3>;
  return ItemPartialVariableRealNScalarOutViewT<ItemType, Accessor>(command, var.variable(), var.asArray());
}

/*!
 * \brief Vue en lecture/écriture.
 */
template <typename ItemType> auto
viewInOut(const ViewBuildInfo& command, MeshVariableScalarRefT<ItemType, Real2>& var)
{
  using Accessor = DataViewGetterSetter<Real2>;
  return ItemVariableRealNScalarOutViewT<ItemType, Accessor>(command, var.variable(), var.asArray());
}

/*!
 * \brief Vue en lecture/écriture.
 */
template <typename ItemType> auto
viewInOut(const ViewBuildInfo& command, MeshPartialVariableScalarRefT<ItemType, Real2>& var)
{
  using Accessor = DataViewGetterSetter<Real2>;
  return ItemPartialVariableRealNScalarOutViewT<ItemType, Accessor>(command, var.variable(), var.asArray());
}

/*!
 * \brief Vue en lecture/écriture.
 */
template <typename ItemType, typename DataType> auto
viewInOut(const ViewBuildInfo& command, MeshVariableArrayRefT<ItemType, DataType>& var)
{
  using Accessor = View1DGetterSetter<DataType>;
  return ItemVariableArrayOutViewT<ItemType, Accessor>(command, var.variable(), var.asArray());
}

/*!
 * \brief Vue en lecture/écriture.
 */
template <typename ItemType, typename DataType> auto
viewInOut(const ViewBuildInfo& command, MeshPartialVariableArrayRefT<ItemType, DataType>& var)
{
  using Accessor = View1DGetterSetter<DataType>;
  return ItemPartialVariableArrayOutViewT<ItemType, Accessor>(command, var.variable(), var.asArray());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture.
 */
template <typename ItemType, typename DataType> auto
viewIn(const ViewBuildInfo& command, const MeshPartialVariableScalarRefT<ItemType, DataType>& var)
{
  return ItemPartialVariableScalarInViewT<ItemType, DataType>(command, var.variable(), var.asArray());
}

/*!
 * \brief Vue en lecture.
 */
template <typename ItemType, typename DataType> auto
viewIn(const ViewBuildInfo& command, const MeshVariableScalarRefT<ItemType, DataType>& var)
{
  return ItemVariableScalarInViewT<ItemType, DataType>(command, var.variable(), var.asArray());
}

/*!
 * \brief Vue en lecture.
 */
template <typename ItemType, typename DataType> auto
viewIn(const ViewBuildInfo& command, const MeshPartialVariableArrayRefT<ItemType, DataType>& var)
{
  return ItemPartialVariableArrayInViewT<ItemType, DataType>(command, var.variable(), var.asArray());
}

/*!
 * \brief Vue en lecture.
 */
template <typename ItemType, typename DataType> auto
viewIn(const ViewBuildInfo& command, const MeshVariableArrayRefT<ItemType, DataType>& var)
{
  return ItemVariableArrayInViewT<ItemType, DataType>(command, var.variable(), var.asArray());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef ItemVariableScalarInViewT<Node, Byte> VariableNodeByteInView;
typedef ItemVariableScalarInViewT<Edge, Byte> VariableEdgeByteInView;
typedef ItemVariableScalarInViewT<Face, Byte> VariableFaceByteInView;
typedef ItemVariableScalarInViewT<Cell, Byte> VariableCellByteInView;
typedef ItemVariableScalarInViewT<Particle, Byte> VariableParticleByteInView;
typedef ItemVariableScalarInViewT<DoF, Byte> VariableDoFByteInView;

typedef ItemVariableScalarInViewT<Node, Int16> VariableNodeInt16InView;
typedef ItemVariableScalarInViewT<Edge, Int16> VariableEdgeInt16InView;
typedef ItemVariableScalarInViewT<Face, Int16> VariableFaceInt16InView;
typedef ItemVariableScalarInViewT<Cell, Int16> VariableCellInt16InView;
typedef ItemVariableScalarInViewT<Particle, Int16> VariableParticleInt16InView;
typedef ItemVariableScalarInViewT<DoF, Int16> VariableDoFInt16InView;

typedef ItemVariableScalarInViewT<Node, Int32> VariableNodeInt32InView;
typedef ItemVariableScalarInViewT<Edge, Int32> VariableEdgeInt32InView;
typedef ItemVariableScalarInViewT<Face, Int32> VariableFaceInt32InView;
typedef ItemVariableScalarInViewT<Cell, Int32> VariableCellInt32InView;
typedef ItemVariableScalarInViewT<Particle, Int32> VariableParticleInt32InView;
typedef ItemVariableScalarInViewT<DoF, Int32> VariableDoFInt32InView;

typedef ItemVariableScalarInViewT<Node, Int64> VariableNodeInt64InView;
typedef ItemVariableScalarInViewT<Edge, Int64> VariableEdgeInt64InView;
typedef ItemVariableScalarInViewT<Face, Int64> VariableFaceInt64InView;
typedef ItemVariableScalarInViewT<Cell, Int64> VariableCellInt64InView;
typedef ItemVariableScalarInViewT<Particle, Int64> VariableParticleInt64InView;
typedef ItemVariableScalarInViewT<DoF, Int64> VariableDoFInt64InView;

typedef ItemVariableScalarInViewT<Node, Real> VariableNodeRealInView;
typedef ItemVariableScalarInViewT<Edge, Real> VariableEdgeRealInView;
typedef ItemVariableScalarInViewT<Face, Real> VariableFaceRealInView;
typedef ItemVariableScalarInViewT<Cell, Real> VariableCellRealInView;
typedef ItemVariableScalarInViewT<Particle, Real> VariableParticleRealInView;
typedef ItemVariableScalarInViewT<DoF, Real> VariableDoFRealInView;

typedef ItemVariableScalarInViewT<Node, Real2> VariableNodeReal2InView;
typedef ItemVariableScalarInViewT<Edge, Real2> VariableEdgeReal2InView;
typedef ItemVariableScalarInViewT<Face, Real2> VariableFaceReal2InView;
typedef ItemVariableScalarInViewT<Cell, Real2> VariableCellReal2InView;
typedef ItemVariableScalarInViewT<Particle, Real2> VariableParticleReal2InView;
typedef ItemVariableScalarInViewT<DoF, Real2> VariableDoFReal2InView;

typedef ItemVariableScalarInViewT<Node, Real3> VariableNodeReal3InView;
typedef ItemVariableScalarInViewT<Edge, Real3> VariableEdgeReal3InView;
typedef ItemVariableScalarInViewT<Face, Real3> VariableFaceReal3InView;
typedef ItemVariableScalarInViewT<Cell, Real3> VariableCellReal3InView;
typedef ItemVariableScalarInViewT<Particle, Real3> VariableParticleReal3InView;
typedef ItemVariableScalarInViewT<DoF, Real3> VariableDoFReal3InView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef ItemVariableScalarOutViewT<Node, DataViewSetter<Byte>> VariableNodeByteOutView;
typedef ItemVariableScalarOutViewT<Edge, DataViewSetter<Byte>> VariableEdgeByteOutView;
typedef ItemVariableScalarOutViewT<Face, DataViewSetter<Byte>> VariableFaceByteOutView;
typedef ItemVariableScalarOutViewT<Cell, DataViewSetter<Byte>> VariableCellByteOutView;
typedef ItemVariableScalarOutViewT<Particle, DataViewSetter<Byte>> VariableParticleByteOutView;
typedef ItemVariableScalarOutViewT<DoF, DataViewSetter<Byte>> VariableDoFByteOutView;

typedef ItemVariableScalarOutViewT<Node, DataViewSetter<Int16>> VariableNodeInt16OutView;
typedef ItemVariableScalarOutViewT<Edge, DataViewSetter<Int16>> VariableEdgeInt16OutView;
typedef ItemVariableScalarOutViewT<Face, DataViewSetter<Int16>> VariableFaceInt16OutView;
typedef ItemVariableScalarOutViewT<Cell, DataViewSetter<Int16>> VariableCellInt16OutView;
typedef ItemVariableScalarOutViewT<Particle, DataViewSetter<Int16>> VariableParticleInt16OutView;
typedef ItemVariableScalarOutViewT<DoF, DataViewSetter<Int16>> VariableDoFInt16OutView;

typedef ItemVariableScalarOutViewT<Node, DataViewSetter<Int32>> VariableNodeInt32OutView;
typedef ItemVariableScalarOutViewT<Edge, DataViewSetter<Int32>> VariableEdgeInt32OutView;
typedef ItemVariableScalarOutViewT<Face, DataViewSetter<Int32>> VariableFaceInt32OutView;
typedef ItemVariableScalarOutViewT<Cell, DataViewSetter<Int32>> VariableCellInt32OutView;
typedef ItemVariableScalarOutViewT<Particle, DataViewSetter<Int32>> VariableParticleInt32OutView;
typedef ItemVariableScalarOutViewT<DoF, DataViewSetter<Int32>> VariableDoFInt32OutView;

typedef ItemVariableScalarOutViewT<Node, DataViewSetter<Int64>> VariableNodeInt64OutView;
typedef ItemVariableScalarOutViewT<Edge, DataViewSetter<Int64>> VariableEdgeInt64OutView;
typedef ItemVariableScalarOutViewT<Face, DataViewSetter<Int64>> VariableFaceInt64OutView;
typedef ItemVariableScalarOutViewT<Cell, DataViewSetter<Int64>> VariableCellInt64OutView;
typedef ItemVariableScalarOutViewT<Particle, DataViewSetter<Int64>> VariableParticleInt64OutView;
typedef ItemVariableScalarOutViewT<DoF, DataViewSetter<Int64>> VariableDoFInt64OutView;

typedef ItemVariableScalarOutViewT<Node, DataViewSetter<Real>> VariableNodeRealOutView;
typedef ItemVariableScalarOutViewT<Edge, DataViewSetter<Real>> VariableEdgeRealOutView;
typedef ItemVariableScalarOutViewT<Face, DataViewSetter<Real>> VariableFaceRealOutView;
typedef ItemVariableScalarOutViewT<Cell, DataViewSetter<Real>> VariableCellRealOutView;
typedef ItemVariableScalarOutViewT<Particle, DataViewSetter<Real>> VariableParticleRealOutView;
typedef ItemVariableScalarOutViewT<DoF, DataViewSetter<Real>> VariableDoFRealOutView;

typedef ItemVariableRealNScalarOutViewT<Node, DataViewSetter<Real2>> VariableNodeReal2OutView;
typedef ItemVariableRealNScalarOutViewT<Edge, DataViewSetter<Real2>> VariableEdgeReal2OutView;
typedef ItemVariableRealNScalarOutViewT<Face, DataViewSetter<Real2>> VariableFaceReal2OutView;
typedef ItemVariableRealNScalarOutViewT<Cell, DataViewSetter<Real2>> VariableCellReal2OutView;
typedef ItemVariableRealNScalarOutViewT<Particle, DataViewSetter<Real2>> VariableParticleReal2OutView;
typedef ItemVariableRealNScalarOutViewT<DoF, DataViewSetter<Real2>> VariableDoFReal2OutView;

typedef ItemVariableRealNScalarOutViewT<Node, DataViewSetter<Real3>> VariableNodeReal3OutView;
typedef ItemVariableRealNScalarOutViewT<Edge, DataViewSetter<Real3>> VariableEdgeReal3OutView;
typedef ItemVariableRealNScalarOutViewT<Face, DataViewSetter<Real3>> VariableFaceReal3OutView;
typedef ItemVariableRealNScalarOutViewT<Cell, DataViewSetter<Real3>> VariableCellReal3OutView;
typedef ItemVariableRealNScalarOutViewT<Particle, DataViewSetter<Real3>> VariableParticleReal3OutView;
typedef ItemVariableRealNScalarOutViewT<DoF, DataViewSetter<Real3>> VariableDoFReal3OutView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
