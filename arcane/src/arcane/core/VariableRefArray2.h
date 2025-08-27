// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableRefArray2.h                                         (C) 2000-2025 */
/*                                                                           */
/* Classe gérant une référence sur une variable tableau 2D.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLEREFARRAY2_H
#define ARCANE_CORE_VARIABLEREFARRAY2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array2View.h"
#include "arcane/core/VariableRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Variable tableau bi dimensionnel.
 *
 * Cette variable gère des tableaux 2D classiques.
 */
template<typename T>
class VariableRefArray2T
: public VariableRef
, public Array2View<T>
{
 public:
  
  typedef T DataType;
  typedef Array2<T> ValueType;
  typedef ConstArrayView<T> ConstReturnReferenceType;
  typedef ArrayView<T> ReturnReferenceType;
  
  //! Type des éléments de la variable
  typedef DataType ElementType;
  //! Type de la classe de base
  typedef VariableRef BaseClass;
  //! Type de la classe gérant la valeur de la variable
  typedef Array2<DataType> ContainerType;
  //! Type du tableau permettant d'accéder à la variable
  typedef Array2View<DataType> ArrayBase;

  typedef Array2VariableT<DataType> PrivatePartType;

  typedef VariableRefArray2T<DataType> ThatClass;

 public:

  //! Construit une référence à une variable tableau spécifiée dans \a vb
  ARCANE_CORE_EXPORT VariableRefArray2T(const VariableBuildInfo& vb);
  //! Construit une référence à partir de \a rhs
  ARCANE_CORE_EXPORT VariableRefArray2T(const VariableRefArray2T<DataType>& rhs);
  //! Construit une référence à partir de \a var
  explicit ARCANE_CORE_EXPORT VariableRefArray2T(IVariable* var);
  /*!
   * \brief Opérateur de recopie.
   * \deprecated Utiliser refersTo() à la place.
   */
  ARCCORE_DEPRECATED_2021("Use refersTo() instead.")
  ARCANE_CORE_EXPORT void operator=(const VariableRefArray2T<DataType>& rhs);
  virtual ARCANE_CORE_EXPORT ~VariableRefArray2T(); //!< Libère les ressources

 public:

  /*!
   * \brief Réalloue le nombre d'éléments de la première dimension du tableau.
   *
   * Le nombre d'éléments de la seconde dimension est mis à zéro.
   * \warning la réallocation ne conserve pas les valeurs précédentes.
   */
  virtual ARCANE_CORE_EXPORT void resize(Integer new_size);

  /*!
   * \brief Réalloue le nombre d'éléments du tableau.
   *
   * Réalloue le tableau avec \a dim1_size comme taille de la première
   * dimension et \a dim2_size comme taille de la deuxième.
   * \warning la réallocation ne conserve pas les valeurs précédentes.
   */
  ARCANE_CORE_EXPORT void resize(Integer dim1_size,Integer dim2_size);

  //! Remplit la variable avev la valeur \a value
  ARCANE_CORE_EXPORT void fill(const DataType& value);

  //! Positionne la référence de l'instance à la variable \a rhs.
  ARCANE_CORE_EXPORT void refersTo(const VariableRefArray2T<DataType>& rhs);

 public:

  virtual bool isArrayVariable() const { return true; }
  virtual Integer arraySize() const { return this->dim1Size(); }
  Integer size() const { return this->dim1Size(); }
  virtual ARCANE_CORE_EXPORT void updateFromInternal();

  /*!
    \brief Retourne le conteneur des valeurs de cette variable.
    *
    L'appel à cette méthode n'est possible que pour les variables
    privées (propriété PPrivate).
    */
  ARCCORE_DEPRECATED_2021("Use _internalTrueData() instead.")
  ARCANE_CORE_EXPORT ContainerType& internalContainer();

 public:

  //! \internal
  ARCANE_CORE_EXPORT IArray2DataInternalT<T>* _internalTrueData();

 public:

  static ARCANE_CORE_EXPORT VariableTypeInfo _internalVariableTypeInfo();
  static ARCANE_CORE_EXPORT VariableInfo _internalVariableInfo(const VariableBuildInfo& vbi);

 private:

  PrivatePartType* m_private_part;

 private:

  static VariableFactoryRegisterer m_auto_registerer;
  static VariableRef* _autoCreate(const VariableBuildInfo& vb);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

