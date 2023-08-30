// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MaterialVariableViews.h                                     (C) 2000-2023 */
/*                                                                           */
/* Gestion des vues sur les variables matériaux pour les accélérateurs.      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_MATERIALVARIABLEVIEWS_H
#define ARCANE_ACCELERATOR_MATERIALVARIABLEVIEWS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/core/materials/IMeshMaterialVariable.h>
#include <arcane/core/materials/MatItem.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::Materials;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des vues sur les variables matériaux.
 */
class MatVariableViewBase
{
 public:
  // Pour l'instant n'utilise pas encore les paramètres
  MatVariableViewBase(RunCommand&, IMeshMaterialVariable*)
  {
  }
 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture sur une variable scalaire du maillage.
 * TODO: Faut-il rajouter la gestion des SIMD comme dans ItemVariableScalarInViewT ?
 */
template<typename ItemType,typename DataType>
class MatItemVariableScalarInViewT
: public MatVariableViewBase
{
 public:

  MatItemVariableScalarInViewT(RunCommand& cmd, IMeshMaterialVariable* var, ArrayView<DataType>* v)
  : MatVariableViewBase(cmd, var), m_value(v), m_value0(v[0].data()){}

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE const DataType& operator[](ComponentItemLocalId lid) const
  {
    return this->m_value[lid.localId().arrayIndex()][lid.localId().valueIndex()];
  }

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE const DataType& operator[](PureMatVarIndex pmvi) const
  {
    return this->m_value0[pmvi.valueIndex()];
  }

  //! Surcharge pour accéder à la valeure globale à partir du cell id
  ARCCORE_HOST_DEVICE const DataType& operator[](Integer cid) const
  {
    return this->m_value0[cid];
  }

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE const DataType& value(ComponentItemLocalId mvi) const
  {
    return this->m_value[mvi.localId().arrayIndex()][mvi.localId().valueIndex()];
  }

  ARCCORE_HOST_DEVICE const DataType& value0(PureMatVarIndex idx) const
  {
    return this->m_value0[idx.valueIndex()];
  }

 private:

  ArrayView<DataType>* m_value;
  DataType* m_value0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * TODO: A faire ?
 *
template<typename ItemType,typename DataType>
class MatItemVariableArrayInViewT
: public MatVariableViewBase
{
};
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture sur une variable scalaire du maillage.
 * TODO: Faut il rajouter la gestion des types SIMD comme dans ItemVariableScalarOutViewT ?
 */
template<typename ItemType,typename Accessor>
class MatItemVariableScalarOutViewT
: public MatVariableViewBase
{
 private:

  using DataType = typename Accessor::ValueType;
  using DataTypeReturnType = DataType&;

  // TODO: faut il rajouter des ARCANE_CHECK_AT(mvi.arrayIndex(), m_value.size()); ? il manquera tjrs le check sur l'autre dimension

 public:

  MatItemVariableScalarOutViewT(RunCommand& cmd,IMeshMaterialVariable* var,ArrayView<DataType>* v)
  : MatVariableViewBase(cmd, var), m_value(v), m_value0(v[0].data()){}

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE Accessor operator[](ComponentItemLocalId lid) const
  {
    return Accessor(this->m_value[lid.localId().arrayIndex()].data()+lid.localId().valueIndex());
  }

  ARCCORE_HOST_DEVICE Accessor operator[](PureMatVarIndex pmvi) const
  {
    return Accessor(this->m_value0[pmvi.valueIndex()]);
  }

  //! Surcharge pour accéder à la valeure globale à partir du cell id
  ARCCORE_HOST_DEVICE const DataType& operator[](Integer cid) const
  {
    return Accessor(this->m_value0[cid]);
  }

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE Accessor value(ComponentItemLocalId lid) const
  {
    return Accessor(this->m_value[lid.localId().arrayIndex()].data()+lid.localId().valueIndex());
  }

  //! Positionne la valeur pour l'entité \a item à \a v
  ARCCORE_HOST_DEVICE void setValue(ComponentItemLocalId lid,const DataType& v) const
  {
    this->m_value[lid.localId().arrayIndex()][lid.localId().valueIndex()] = v;
  }

  ARCCORE_HOST_DEVICE Accessor value0(PureMatVarIndex idx) const
  {
    return Accessor(this->m_value0[idx.valueIndex()]);
  }

 private:

  ArrayView<DataType>* m_value;
  DataType* m_value0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Vue en écriture pour les variables materiaux scalaire
 */
template<typename DataType> auto
viewOut(RunCommand& cmd, CellMaterialVariableScalarRef<DataType>& var)
{
  using Accessor = DataViewSetter<DataType>;
  return MatItemVariableScalarOutViewT<Cell,Accessor>(cmd, var.materialVariable(),var._internalValue());
}

/*!
 * \brief Vue en écriture pour les variables materiaux scalaire.
 * Spécialisation pour le Real2 pour éviter les mauvais usages
 *
 * TODO: A faire plus tard ?  => faire MatItemVariableRealNScalarOutViewT
 * 
template<> auto
viewOut(RunCommand& cmd, CellMaterialVariableScalarRef<Real2>& var)
{
  using Accessor = DataViewSetter<Real2>;
  return MatItemVariableRealNScalarOutViewT<Cell,Accessor>(cmd, var.materialVariable(),var._internalValue());

}
*/

/*!
 * \brief Vue en écriture pour les variables materiaux scalaire.
 * Spécialisation pour le Real3 pour éviter les mauvais usages
 *
 * TODO: A faire plus tard ? => faire MatItemVariableRealNScalarOutViewT
 * 
template<> auto
viewOut(RunCommand& cmd, CellMaterialVariableScalarRef<Real3>& var)
{
  using Accessor = DataViewSetter<Real3>;
  return MatItemVariableRealNScalarOutViewT<Cell,Accessor>(cmd, var.materialVariable(),var._internalValue());
}
 */

/*!
 * \brief Vue en écriture pour les variables materiaux tableau
 *
 * TODO: A faire plus tard ?
 *
template<typename DataType> auto
viewOut(RunCommand& cmd, CellMaterialVariableArrayRef<DataType>& var)
{
}
*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Vue en lecture/écriture pour les variables materiaux scalaire
 */
template<typename DataType> auto
viewInOut(RunCommand& cmd, CellMaterialVariableScalarRef<DataType>& var)
{
  using Accessor = DataViewGetterSetter<DataType>;
  return MatItemVariableScalarOutViewT<Cell,Accessor>(cmd, var.materialVariable(),var._internalValue());
}

/*!
 * \brief Vue en lecture/écriture pour les variables materiaux scalaire.
 * Spécialisation pour le Real2 pour éviter les mauvais usages
 *
 * TODO: A faire plus tard ? => faire MatItemVariableRealNScalarOutViewT
 * 
template<> auto
viewInOut(RunCommand& cmd, CellMaterialVariableScalarRef<Real2>& var)
{
}
*/

/*!
 * \brief Vue en lecture/écriture pour les variables materiaux scalaire.
 * Spécialisation pour le Real3 pour éviter les mauvais usages
 * 
 * TODO: A faire plus tard ? => faire MatItemVariableRealNScalarOutViewT
 *
template<> auto
viewInOut(RunCommand& cmd, CellMaterialVariableScalarRef<Real3>& var)
{
}
*/

/*!
 * \brief Vue en lecture/écriture pour les variables materiaux tableau
 *
 * TODO: A faire plus tard ?
 * 
template<typename DataType> auto
viewInOut(RunCommand& cmd, CellMaterialVariableArrayRef<DataType>& var)
{
}
*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Vue en lecture pour les variables materiaux scalaire
 */
template<typename DataType> auto
viewIn(RunCommand& cmd,const CellMaterialVariableScalarRef<DataType>& var)
{
  return MatItemVariableScalarInViewT<Cell,DataType>(cmd, var.materialVariable(),var._internalValue());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

#endif
