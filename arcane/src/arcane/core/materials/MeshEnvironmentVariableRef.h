// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshEnvironmentVariableRef.h                                (C) 2000-2024 */
/*                                                                           */
/* Référence à une variable sur un milieu du maillage.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHENVIRONMENTVARIABLEREF_H
#define ARCANE_MATERIALS_MESHENVIRONMENTVARIABLEREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file MeshEnvironmentVariableRef.h
 *
 * Ce fichier contient les différents types gérant les références
 * sur les variables milieux.
 */
#include "arcane/core/materials/MeshMaterialVariableRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Variable scalaire sur les mailles d'un milieu du maillage.
 *
 * Ce type de variable est identique à ceci près qu'elle n'a de valeurs
 * que sur les milieux et les mailles globales mais pas sur les matériaux.
 */
template <typename DataType_>
class CellEnvironmentVariableScalarRef
: public MeshMaterialVariableRef
{
 public:

  using DataType = DataType_;
  using PrivatePartType = IScalarMeshMaterialVariable<Cell, DataType>;
  using ThatClass = CellEnvironmentVariableScalarRef<DataType>;
  using ItemType = Cell;
  using GlobalVariableRefType = MeshVariableScalarRefT<ItemType, DataType>;

 public:

  explicit ARCANE_CORE_EXPORT CellEnvironmentVariableScalarRef(const VariableBuildInfo& vb);
  //! Construit une référence à la variable spécifiée dans \a vb
  explicit ARCANE_CORE_EXPORT CellEnvironmentVariableScalarRef(const MaterialVariableBuildInfo& vb);
  ARCANE_CORE_EXPORT CellEnvironmentVariableScalarRef(const ThatClass& rhs);

 public:

  //! Opérateur de recopie (interdit)
  ARCANE_CORE_EXPORT ThatClass& operator=(const ThatClass& rhs) = delete;
  //! Constructeur vide (interdit)
  CellEnvironmentVariableScalarRef() = delete;

 public:

  //! Positionne la référence de l'instance à la variable \a rhs.
  ARCANE_CORE_EXPORT virtual void refersTo(const ThatClass& rhs);

  /*!
   * \internal
   */
  ARCANE_CORE_EXPORT void updateFromInternal() override;

 protected:

  DataType operator[](MatVarIndex mvi) const
  {
    return m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }
  DataType& operator[](MatVarIndex mvi)
  {
    return m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }

 public:

  //! Valeur partielle de la variable pour la maille matériau \a mc
  DataType operator[](ComponentItemLocalId mc) const
  {
    return this->operator[](mc.localId());
  }

  //! Valeur partielle de la variable pour la maille matériau \a mc
  DataType& operator[](ComponentItemLocalId mc)
  {
    return this->operator[](mc.localId());
  }

  //! Valeur globale de la variable pour la maille \a c
  DataType operator[](CellLocalId c) const
  {
    return m_value[0][c.localId()];
  }

  //! Valeur globale de la variable pour la maille \a c
  DataType& operator[](CellLocalId c)
  {
    return m_value[0][c.localId()];
  }

  /*!
   * \brief Valeur de la variable pour le milieu d'index \a env_id de
   * la maille \a ou 0 si absent de la maille.
   */
  ARCANE_CORE_EXPORT DataType envValue(AllEnvCell c,Int32 env_id) const;

 public:
  
  ARCANE_CORE_EXPORT void fill(const DataType& value);
  ARCANE_CORE_EXPORT void fillPartialValues(const DataType& value);

 public:

  //! Variable globale associée à cette variable matériau
  ARCANE_CORE_EXPORT GlobalVariableRefType& globalVariable();
  //! Variable globale associée à cette variable matériau
  ARCANE_CORE_EXPORT const GlobalVariableRefType& globalVariable() const;

 private:

  PrivatePartType* m_private_part = nullptr;
  ArrayView<DataType>* m_value = nullptr;
  ArrayView<ArrayView<DataType>> m_container_value;

 public:

  // TODO: Temporaire. a supprimer.
  ArrayView<DataType>* _internalValue() const { return m_value; }

 private:

  void _init();
  void _setContainerView();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Variable tableau sur les mailles d'un matériau du maillage.
 * Pour l'instant, cette classe n'est instanciée que pour les mailles
 */
template<typename DataType_>
class CellEnvironmentVariableArrayRef
: public MeshMaterialVariableRef
{
 public:

  using DataType = DataType_;
  using PrivatePartType = IArrayMeshMaterialVariable<Cell,DataType>;
  using ItemType = Cell;
  using GlobalVariableRefType = MeshVariableArrayRefT<ItemType, DataType>;
  using ThatClass = CellEnvironmentVariableArrayRef<DataType>;

 public:

  explicit ARCANE_CORE_EXPORT CellEnvironmentVariableArrayRef(const VariableBuildInfo& vb);
  //! Construit une référence à la variable spécifiée dans \a vb
  explicit ARCANE_CORE_EXPORT CellEnvironmentVariableArrayRef(const MaterialVariableBuildInfo& vb);
  ARCANE_CORE_EXPORT CellEnvironmentVariableArrayRef(const ThatClass& rhs);

 public:

  //! Opérateur de recopie (interdit)
  ThatClass& operator=(const ThatClass& rhs) = delete;
  //! Constructeur vide (interdit)
  CellEnvironmentVariableArrayRef() = delete;

 public:

  //! Positionne la référence de l'instance à la variable \a rhs.
  ARCANE_CORE_EXPORT virtual void refersTo(const ThatClass& rhs);

  /*!
   * \internal
   */
  ARCANE_CORE_EXPORT void updateFromInternal() override;

 public:

  //! Variable globale associée à cette variable matériau
  ARCANE_CORE_EXPORT GlobalVariableRefType& globalVariable();
  //! Variable globale associée à cette variable matériau
  ARCANE_CORE_EXPORT const GlobalVariableRefType& globalVariable() const;

 public:

  /*!
   * \brief Redimensionne le nombre d'éléments du tableau.
   *
   * La première dimension reste toujours égale au nombre d'éléments du maillage.
   * Seule la deuxième composante est retaillée.
   */
  ARCANE_CORE_EXPORT void resize(Integer dim2_size);


 protected:

  ConstArrayView<DataType> operator[](MatVarIndex mvi) const
  {
    return m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }
  ArrayView<DataType> operator[](MatVarIndex mvi)
  {
    return m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }

 public:

  //! Valeur partielle de la variable pour la maille matériau \a mc
  ConstArrayView<DataType> operator[](ComponentItemLocalId mc) const
  {
    return this->operator[](mc.localId());
  }

  //! Valeur partielle de la variable pour la maille matériau \a mc
  ArrayView<DataType> operator[](ComponentItemLocalId mc)
  {
    return this->operator[](mc.localId());
  }

  //! Valeur globale de la variable pour la maille \a c
  ConstArrayView<DataType> operator[](CellLocalId c) const
  {
    return m_value[0][c.localId()];
  }

  //! Valeur globale de la variable pour la maille \a c
  ArrayView<DataType> operator[](CellLocalId c)
  {
    return m_value[0][c.localId()];
  }

 private:

  PrivatePartType* m_private_part = nullptr;
  Array2View<DataType>* m_value = nullptr;
  ArrayView<Array2View<DataType>> m_container_value;

 private:

  void _init();
  void _setContainerView();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! %Variable milieu de type \a #Byte
typedef CellEnvironmentVariableScalarRef<Byte> EnvironmentVariableCellByte;
//! %Variable milieu de type \a #Real
typedef CellEnvironmentVariableScalarRef<Real> EnvironmentVariableCellReal;
//! %Variable milieu de type \a #Int16
typedef CellEnvironmentVariableScalarRef<Int16> EnvironmentVariableCellInt16;
//! %Variable milieu de type \a #Int32
typedef CellEnvironmentVariableScalarRef<Int32> EnvironmentVariableCellInt32;
//! %Variable milieu de type \a #Int64
typedef CellEnvironmentVariableScalarRef<Int64> EnvironmentVariableCellInt64;
//! %Variable milieu de type \a Real2
typedef CellEnvironmentVariableScalarRef<Real2> EnvironmentVariableCellReal2;
//! %Variable milieu de type \a Real3
typedef CellEnvironmentVariableScalarRef<Real3> EnvironmentVariableCellReal3;
//! %Variable milieu de type \a Real2x2
typedef CellEnvironmentVariableScalarRef<Real2x2> EnvironmentVariableCellReal2x2;
//! %Variable milieu de type \a Real3x3
typedef CellEnvironmentVariableScalarRef<Real3x3> EnvironmentVariableCellReal3x3;

#ifdef ARCANE_64BIT
//! %Variable milieu de type \a #Integer
typedef EnvironmentVariableCellInt64 EnvironmentVariableCellInteger;
#else
//! %Variable milieu de type \a #Integer
typedef EnvironmentVariableCellInt32 EnvironmentVariableCellInteger;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! %Variable milieu de type tableau de \a #Byte
typedef CellEnvironmentVariableArrayRef<Byte> EnvironmentVariableCellArrayByte;
//! %Variable milieu de type tableau de \a #Real
typedef CellEnvironmentVariableArrayRef<Real> EnvironmentVariableCellArrayReal;
//! %Variable milieu de type tableau de \a #Int16
typedef CellEnvironmentVariableArrayRef<Int16> EnvironmentVariableCellArrayInt16;
//! %Variable milieu de type tableau de \a #Int32
typedef CellEnvironmentVariableArrayRef<Int32> EnvironmentVariableCellArrayInt32;
//! %Variable milieu de type tableau de \a #Int64
typedef CellEnvironmentVariableArrayRef<Int64> EnvironmentVariableCellArrayInt64;
//! %Variable milieu de type tableau de \a Real2
typedef CellEnvironmentVariableArrayRef<Real2> EnvironmentVariableCellArrayReal2;
//! %Variable milieu de type tableau de \a Real3
typedef CellEnvironmentVariableArrayRef<Real3> EnvironmentVariableCellArrayReal3;
//! %Variable milieu de type tableau de \a Real2x2
typedef CellEnvironmentVariableArrayRef<Real2x2> EnvironmentVariableCellArrayReal2x2;
//! %Variable milieu de type tableau de \a Real3x3
typedef CellEnvironmentVariableArrayRef<Real3x3> EnvironmentVariableCellArrayReal3x3;

#ifdef ARCANE_64BIT
//! %Variable milieu de type tableau de \a #Integer
typedef EnvironmentVariableCellInt64 EnvironmentVariableCellArrayInteger;
#else
//! %Variable milieu de type tableau de \a #Integer
typedef EnvironmentVariableCellInt32 EnvironmentVariableCellArrayInteger;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

