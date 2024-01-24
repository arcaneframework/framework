// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableView.h                                              (C) 2000-2023 */
/*                                                                           */
/* Classes gérant les vues sur les variables.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VARIABLEVIEW_H
#define ARCANE_VARIABLEVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arcane/core/SimdItem.h"
#include "arcane/core/DataView.h"
#include "arcane/core/ItemLocalId.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file VariableView.h
 *
 * Ce fichier contient les déclarations des types pour gérer
 * les vues sur les variables du maillage.
 *
 * Les types et méthodes de ce fichier sont obsolètes. La nouvelle version
 * des vues avec support des accélérateurs est dans le fichier
 * 'accelerator/Views.h'.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: Faire les vues en ReadWrite pour les accesseurs SIMD

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Pour compatibilité avec le code existant
template<typename DataType>
using ViewSetter ARCANE_DEPRECATED_REASON("Use 'DataViewSetter' type instead") = DataViewSetter<DataType>;
template<typename DataType>
using ViewGetterSetter ARCANE_DEPRECATED_REASON("Use 'DataViewGetterSetter' type instead") = DataViewGetterSetter<DataType>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des vues sur les variables.
 */
class VariableViewBase
{
 public:
  // Pour l'instant n'utilise pas encore \a var
  // mais il ne faut pas le supprimer
  VariableViewBase(IVariable*) {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour accéder à un tableau 1D d'une vue en lecture/écriture.
 */
template<typename DataType>
class View1DGetterSetter
{
 public:
  using ValueType = DataType;
  using DataTypeReturnReference = Span<DataType>;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour accéder à un tableau 1D d'une vue en lecture/écriture.
 */
template<typename DataType>
class View1DSetter
{
 public:
  using ValueType = DataType;
  using DataTypeReturnReference = View1DSetter<DataType>;
  View1DSetter(Span<DataType> data) : m_data(data){}
  DataViewSetter<DataType> operator[](Int64 index) const
  {
    return DataViewSetter<DataType>(m_data.ptrAt(index));
  }
 private:
  Span<DataType> m_data;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture sur une variable scalaire du maillage.
 */
template<typename ItemType, typename Accessor>
class ItemVariableScalarOutViewT
: public VariableViewBase
{
 public:

  using DataType = typename Accessor::ValueType;
  using DataTypeReturnReference = DataType&;
  using ItemIndexType = typename ItemTraitsT<ItemType>::LocalIdType;

 public:

  ItemVariableScalarOutViewT(IVariable* var,Span<DataType> v)
  : VariableViewBase(var), m_values(v.data()),
    m_size(v.size()){}

  //! Opérateur d'accès vectoriel avec indirection.
  SimdSetter<DataType> operator[](SimdItemIndexT<ItemType> simd_item) const
  {
    return SimdSetter<DataType>(m_values,simd_item.simdLocalIds());
  }

  //! Opérateur d'accès vectoriel sans indirection.
  SimdDirectSetter<DataType> operator[](SimdItemDirectIndexT<ItemType> simd_item) const
  {
    return SimdDirectSetter<DataType>(m_values+simd_item.baseLocalId());
  }

  //! Opérateur d'accès pour l'entité \a item
  Accessor operator[](ItemIndexType i) const
  {
    ARCANE_CHECK_AT(i.localId(),m_size);
    return Accessor(this->m_values + i.localId());
  }

  //! Opérateur d'accès pour l'entité \a item
  Accessor value(ItemIndexType i) const
  {
    ARCANE_CHECK_AT(i.localId(),m_size);
    return Accessor(this->m_values + i.localId());
  }

  //! Positionne la valeur pour l'entité \a item à \a v
  void setValue(ItemIndexType i,const DataType& v) const
  {
    ARCANE_CHECK_AT(i.localId(),m_size);
    this->m_values[i.localId()] = v;
  }

 private:
  DataType* m_values;
  Int64 m_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture sur une variable scalaire du maillage.
 */
template<typename ItemType,typename DataType>
class ItemVariableScalarInViewT
: public VariableViewBase
{
 private:

  typedef typename ItemTraitsT<ItemType>::LocalIdType ItemIndexType;

 public:

  ItemVariableScalarInViewT(IVariable* var,Span<const DataType> v)
  : VariableViewBase(var), m_values(v){}

  //! Opérateur d'accès vectoriel avec indirection.
  typename SimdTypeTraits<DataType>::SimdType
  operator[](SimdItemIndexT<ItemType> simd_item) const
  {
    typedef typename SimdTypeTraits<DataType>::SimdType SimdType;
    return SimdType(m_values.data(),simd_item.simdLocalIds());
  }

  //! Opérateur d'accès vectoriel avec indirection.
  typename SimdTypeTraits<DataType>::SimdType
  operator[](SimdItemDirectIndexT<ItemType> simd_item) const
  {
    typedef typename SimdTypeTraits<DataType>::SimdType SimdType;
    return SimdType(m_values.data()+simd_item.baseLocalId());
  }

  //! Opérateur d'accès pour l'entité \a item
  const DataType& operator[](ItemIndexType i) const
  {
    return this->m_values[i.localId()];
  }

  //! Opérateur d'accès pour l'entité \a item
  const DataType& value(ItemIndexType i) const
  {
    return this->m_values[i.localId()];
  }

 private:
  Span<const DataType> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture sur une variable tableau du maillage.
 */
template<typename ItemType,typename DataType>
class ItemVariableArrayInViewT
: public VariableViewBase
{
 private:

  typedef typename ItemTraitsT<ItemType>::LocalIdType ItemIndexType;

 public:

  ItemVariableArrayInViewT(IVariable* var,Span2<const DataType> v)
  : VariableViewBase(var), m_values(v){}

  //! Opérateur d'accès pour l'entité \a item
  Span<const DataType> operator[](ItemIndexType i) const
  {
    return this->m_values[i.localId()];
  }

  //! Opérateur d'accès pour l'entité \a item
  Span<const DataType> value(ItemIndexType i) const
  {
    return this->m_values[i.localId()];
  }

 private:
  Span2<const DataType> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture sur une variable tableau du maillage.
 */
template<typename ItemType,typename Accessor>
class ItemVariableArrayOutViewT
: public VariableViewBase
{
 private:

  using DataType = typename Accessor::ValueType;
  using DataTypeReturnType = typename Accessor::DataTypeReturnReference;
  using ItemIndexType = typename ItemTraitsT<ItemType>::LocalIdType;

 public:

  ItemVariableArrayOutViewT(IVariable* var,Span2<DataType> v)
  : VariableViewBase(var), m_values(v){}

  //! Opérateur d'accès pour l'entité \a item
  DataTypeReturnType operator[](ItemIndexType i) const
  {
    return DataTypeReturnType(this->m_values[i.localId()]);
  }

  //! Opérateur d'accès pour l'entité \a item
  Span<DataType> value(ItemIndexType i) const
  {
    return DataTypeReturnType(this->m_values[i.localId()]);
  }

 private:
  Span2<DataType> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture sur une variable scalaire de type 'RealN' du maillage.
 
 Cette classe spécialise les vues modifiable pour les réels 'Real2', 'Real3',
 'Real2x2' et 'Real3x3'. La spécialisation s'assure qu'on ne puisse pas
 modifier uniquement une composante de ces vecteurs de réels. Par exemple:

 \code
 VariableNodeReal3View force_view = ...;
 Node node = ...;

 // OK:
 force_view[node] = Real3(x,y,z);

 // Interdit:
 // force_view[node].x = ...;

 \endcode
*/
template<typename ItemType,typename Accessor>
class ItemVariableRealNScalarOutViewT
: public VariableViewBase
{
 public:

  using DataType = typename Accessor::ValueType;
  using DataTypeReturnReference = DataType&;
  using ItemIndexType = typename ItemTraitsT<ItemType>::LocalIdType;

 public:

  //! Construit la vue
  ItemVariableRealNScalarOutViewT(IVariable* var,Span<DataType> v)
  : VariableViewBase(var), m_values(v.data()), m_size(v.size()){}

  //! Opérateur d'accès vectoriel avec indirection.
  SimdSetter<DataType> operator[](SimdItemIndexT<ItemType> simd_item) const
  {
    return SimdSetter<DataType>(m_values,simd_item.simdLocalIds());
  }

  //! Opérateur d'accès vectoriel sans indirection.
  SimdDirectSetter<DataType> operator[](SimdItemDirectIndexT<ItemType> simd_item) const
  {
    return SimdDirectSetter<DataType>(m_values+simd_item.baseLocalId());
  }

  //! Opérateur d'accès pour l'entité \a item
  Accessor operator[](ItemIndexType item) const
  {
    ARCANE_CHECK_AT(item.localId(),m_size);
    return Accessor(this->m_values+item.localId());
  }

  //! Opérateur d'accès pour l'entité \a item
  Accessor value(ItemIndexType item) const
  {
    ARCANE_CHECK_AT(item.localId(),m_size);
    return Accessor(this->m_values+item.localId());
  }

  //! Positionne la valeur pour l'entité \a item à \a v
  void setValue(ItemIndexType item,const DataType& v) const
  {
    ARCANE_CHECK_AT(item.localId(),m_size);
    this->m_values[item.localId()] = v;
  }

 private:
  DataType* m_values;
  Int64 m_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture.
 */
template<typename ItemType,typename DataType> auto
viewOut(MeshVariableScalarRefT<ItemType,DataType>& var)
{
  using Accessor = DataViewSetter<DataType>;
  return ItemVariableScalarOutViewT<ItemType,Accessor>(var.variable(),var.asArray());
}

/*!
 * \brief Vue en écriture.
 */
template<typename ItemType> auto
viewOut(MeshVariableScalarRefT<ItemType,Real3>& var)
{
  using Accessor = DataViewSetter<Real3>;
  return ItemVariableRealNScalarOutViewT<ItemType,Accessor>(var.variable(),var.asArray());
}

/*!
 * \brief Vue en écriture.
 */
template<typename ItemType> auto
viewOut(MeshVariableScalarRefT<ItemType,Real2>& var)
{
  using Accessor = DataViewSetter<Real2>;
  return ItemVariableRealNScalarOutViewT<ItemType,Accessor>(var.variable(),var.asArray());
}

/*!
 * \brief Vue en écriture.
 */
template<typename ItemType,typename DataType> auto
viewOut(MeshVariableArrayRefT<ItemType,DataType>& var)
{
  using Accessor = View1DSetter<DataType>;
  return ItemVariableArrayOutViewT<ItemType,Accessor>(var.variable(),var.asArray());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture/écriture.
 */
template<typename ItemType,typename DataType> auto
viewInOut(MeshVariableScalarRefT<ItemType,DataType>& var)
{
  using Accessor = DataViewGetterSetter<DataType>;
  return ItemVariableScalarOutViewT<ItemType,Accessor>(var.variable(),var.asArray());
}

/*!
 * \brief Vue en lecture/écriture.
 */
template<typename ItemType> auto
viewInOut(MeshVariableScalarRefT<ItemType,Real3>& var)
{
  using Accessor = DataViewGetterSetter<Real3>;
  return ItemVariableRealNScalarOutViewT<ItemType,Accessor>(var.variable(),var.asArray());
}

/*!
 * \brief Vue en lecture/écriture.
 */
template<typename ItemType> auto 
viewInOut(MeshVariableScalarRefT<ItemType,Real2>& var)
{
  using Accessor = DataViewGetterSetter<Real2>;
  return ItemVariableRealNScalarOutViewT<ItemType,Accessor>(var.variable(),var.asArray());
}

/*!
 * \brief Vue en lecture/écriture.
 */
template<typename ItemType,typename DataType> auto
viewInOut(MeshVariableArrayRefT<ItemType,DataType>& var)
{
  using Accessor = View1DGetterSetter<DataType>;
  return ItemVariableArrayOutViewT<ItemType,Accessor>(var.variable(),var.asArray());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture.
 */
template<typename ItemType,typename DataType> auto
viewIn(const MeshVariableScalarRefT<ItemType,DataType>& var)
{
  return ItemVariableScalarInViewT<ItemType,DataType>(var.variable(),var.asArray());
}

/*!
 * \brief Vue en lecture.
 */
template<typename ItemType,typename DataType> auto
viewIn(const MeshVariableArrayRefT<ItemType,DataType>& var)
{
  return ItemVariableArrayInViewT<ItemType,DataType>(var.variable(),var.asArray());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef ItemVariableScalarInViewT<Node,Byte> VariableNodeByteInView;
typedef ItemVariableScalarInViewT<Edge,Byte> VariableEdgeByteInView;
typedef ItemVariableScalarInViewT<Face,Byte> VariableFaceByteInView;
typedef ItemVariableScalarInViewT<Cell,Byte> VariableCellByteInView;
typedef ItemVariableScalarInViewT<Particle,Byte> VariableParticleByteInView;

typedef ItemVariableScalarInViewT<Node,Int16> VariableNodeInt16InView;
typedef ItemVariableScalarInViewT<Edge,Int16> VariableEdgeInt16InView;
typedef ItemVariableScalarInViewT<Face,Int16> VariableFaceInt16InView;
typedef ItemVariableScalarInViewT<Cell,Int16> VariableCellInt16InView;
typedef ItemVariableScalarInViewT<Particle,Int16> VariableParticleInt16InView;

typedef ItemVariableScalarInViewT<Node,Int32> VariableNodeInt32InView;
typedef ItemVariableScalarInViewT<Edge,Int32> VariableEdgeInt32InView;
typedef ItemVariableScalarInViewT<Face,Int32> VariableFaceInt32InView;
typedef ItemVariableScalarInViewT<Cell,Int32> VariableCellInt32InView;
typedef ItemVariableScalarInViewT<Particle,Int32> VariableParticleInt32InView;

typedef ItemVariableScalarInViewT<Node,Int64> VariableNodeInt64InView;
typedef ItemVariableScalarInViewT<Edge,Int64> VariableEdgeInt64InView;
typedef ItemVariableScalarInViewT<Face,Int64> VariableFaceInt64InView;
typedef ItemVariableScalarInViewT<Cell,Int64> VariableCellInt64InView;
typedef ItemVariableScalarInViewT<Particle,Int64> VariableParticleInt64InView;

typedef ItemVariableScalarInViewT<Node,Real> VariableNodeRealInView;
typedef ItemVariableScalarInViewT<Edge,Real> VariableEdgeRealInView;
typedef ItemVariableScalarInViewT<Face,Real> VariableFaceRealInView;
typedef ItemVariableScalarInViewT<Cell,Real> VariableCellRealInView;
typedef ItemVariableScalarInViewT<Particle,Real> VariableParticleRealInView;

typedef ItemVariableScalarInViewT<Node,Real2> VariableNodeReal2InView;
typedef ItemVariableScalarInViewT<Edge,Real2> VariableEdgeReal2InView;
typedef ItemVariableScalarInViewT<Face,Real2> VariableFaceReal2InView;
typedef ItemVariableScalarInViewT<Cell,Real2> VariableCellReal2InView;
typedef ItemVariableScalarInViewT<Particle,Real2> VariableParticleReal2InView;

typedef ItemVariableScalarInViewT<Node,Real3> VariableNodeReal3InView;
typedef ItemVariableScalarInViewT<Edge,Real3> VariableEdgeReal3InView;
typedef ItemVariableScalarInViewT<Face,Real3> VariableFaceReal3InView;
typedef ItemVariableScalarInViewT<Cell,Real3> VariableCellReal3InView;
typedef ItemVariableScalarInViewT<Particle,Real3> VariableParticleReal3InView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef ItemVariableScalarOutViewT<Node,DataViewSetter<Byte>> VariableNodeByteOutView;
typedef ItemVariableScalarOutViewT<Edge,DataViewSetter<Byte>> VariableEdgeByteOutView;
typedef ItemVariableScalarOutViewT<Face,DataViewSetter<Byte>> VariableFaceByteOutView;
typedef ItemVariableScalarOutViewT<Cell,DataViewSetter<Byte>> VariableCellByteOutView;
typedef ItemVariableScalarOutViewT<Particle,DataViewSetter<Byte>> VariableParticleByteOutView;

typedef ItemVariableScalarOutViewT<Node,DataViewSetter<Int16>> VariableNodeInt16OutView;
typedef ItemVariableScalarOutViewT<Edge,DataViewSetter<Int16>> VariableEdgeInt16OutView;
typedef ItemVariableScalarOutViewT<Face,DataViewSetter<Int16>> VariableFaceInt16OutView;
typedef ItemVariableScalarOutViewT<Cell,DataViewSetter<Int16>> VariableCellInt16OutView;
typedef ItemVariableScalarOutViewT<Particle,DataViewSetter<Int16>> VariableParticleInt16OutView;

typedef ItemVariableScalarOutViewT<Node,DataViewSetter<Int32>> VariableNodeInt32OutView;
typedef ItemVariableScalarOutViewT<Edge,DataViewSetter<Int32>> VariableEdgeInt32OutView;
typedef ItemVariableScalarOutViewT<Face,DataViewSetter<Int32>> VariableFaceInt32OutView;
typedef ItemVariableScalarOutViewT<Cell,DataViewSetter<Int32>> VariableCellInt32OutView;
typedef ItemVariableScalarOutViewT<Particle,DataViewSetter<Int32>> VariableParticleInt32OutView;

typedef ItemVariableScalarOutViewT<Node,DataViewSetter<Int64>> VariableNodeInt64OutView;
typedef ItemVariableScalarOutViewT<Edge,DataViewSetter<Int64>> VariableEdgeInt64OutView;
typedef ItemVariableScalarOutViewT<Face,DataViewSetter<Int64>> VariableFaceInt64OutView;
typedef ItemVariableScalarOutViewT<Cell,DataViewSetter<Int64>> VariableCellInt64OutView;
typedef ItemVariableScalarOutViewT<Particle,DataViewSetter<Int64>> VariableParticleInt64OutView;

typedef ItemVariableScalarOutViewT<Node,DataViewSetter<Real>> VariableNodeRealOutView;
typedef ItemVariableScalarOutViewT<Edge,DataViewSetter<Real>> VariableEdgeRealOutView;
typedef ItemVariableScalarOutViewT<Face,DataViewSetter<Real>> VariableFaceRealOutView;
typedef ItemVariableScalarOutViewT<Cell,DataViewSetter<Real>> VariableCellRealOutView;
typedef ItemVariableScalarOutViewT<Particle,DataViewSetter<Real>> VariableParticleRealOutView;

typedef ItemVariableRealNScalarOutViewT<Node,DataViewSetter<Real2>> VariableNodeReal2OutView;
typedef ItemVariableRealNScalarOutViewT<Edge,DataViewSetter<Real2>> VariableEdgeReal2OutView;
typedef ItemVariableRealNScalarOutViewT<Face,DataViewSetter<Real2>> VariableFaceReal2OutView;
typedef ItemVariableRealNScalarOutViewT<Cell,DataViewSetter<Real2>> VariableCellReal2OutView;
typedef ItemVariableRealNScalarOutViewT<Particle,DataViewSetter<Real2>> VariableParticleReal2OutView;

typedef ItemVariableRealNScalarOutViewT<Node,DataViewSetter<Real3>> VariableNodeReal3OutView;
typedef ItemVariableRealNScalarOutViewT<Edge,DataViewSetter<Real3>> VariableEdgeReal3OutView;
typedef ItemVariableRealNScalarOutViewT<Face,DataViewSetter<Real3>> VariableFaceReal3OutView;
typedef ItemVariableRealNScalarOutViewT<Cell,DataViewSetter<Real3>> VariableCellReal3OutView;
typedef ItemVariableRealNScalarOutViewT<Particle,DataViewSetter<Real3>> VariableParticleReal3OutView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

