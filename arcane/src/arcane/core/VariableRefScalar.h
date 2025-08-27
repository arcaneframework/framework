// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableRefScalar.h                                         (C) 2000-2020 */
/*                                                                           */
/* Classe gérant une référence sur une variable scalaire.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VARIABLEREFSCALAR_H
#define ARCANE_VARIABLEREFSCALAR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableRef.h"
#include "arcane/core/Parallel.h"
#include "arcane/core/MathUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class VariableFactoryRegisterer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Référence à une variable scalaire.
 *
 * L'opérateur operator()() permet d'accéder à la valeur de la variable en
 * lecture seulement. Pour modifier la valeur de la variable, il faut
 * utiliser la méthode assign() ou l'opérateur operator=(). A noter que
 * l'affectation provoque une mise à jour des références et peut s'avérer
 * coûteuse.
 */
template<typename DataType>
class VariableRefScalarT
: public VariableRef
{
 public:

  //! Type des éléments de la variable
  typedef DataType ElementType;
  //! Type de la classe de base
  typedef VariableRef BaseClass;

  typedef VariableScalarT<DataType> PrivatePartType;

  typedef VariableRefScalarT<DataType> ThatClass;

 public:

  //! Construit une référence à une variable scalaire spécifiée dans \a vb
  explicit ARCANE_CORE_EXPORT VariableRefScalarT(const VariableBuildInfo& b);
  //! Construit une référence à partir de \a rhs
  ARCANE_CORE_EXPORT VariableRefScalarT(const VariableRefScalarT<DataType>& rhs);
  //! Construit une référence à partir de \a var
  explicit ARCANE_CORE_EXPORT VariableRefScalarT(IVariable* var);
  //! Positionne la référence de l'instance à la variable \a rhs.
  ARCANE_CORE_EXPORT void refersTo(const VariableRefScalarT<DataType>& rhs);

#ifdef ARCANE_DOTNET
 public:
#else
 protected:
#endif

  //! Constructeur vide
  VariableRefScalarT() : m_private_part(nullptr) {}

 public:

  virtual bool isArrayVariable() const { return false; }
  virtual Integer arraySize() const { return 0; }
  virtual ARCANE_CORE_EXPORT void updateFromInternal();

 public:

  ArrayView<DataType> asArray() { return ArrayView<DataType>(1,&(m_private_part->value())); }
  ConstArrayView<DataType> asArray() const { return ConstArrayView<DataType>(1,&(m_private_part->value())); }

 public:
	
  void operator=(const DataType& v) { assign(v); }
  VariableRefScalarT<DataType>& operator=(const VariableRefScalarT<DataType>& v)
  {
    assign(v());
    return (*this);
  }

  //! Réinitialise la variable avec sa valeur par défaut
  void reset() { assign(DataType()); }

  //! Valeur du scalaire
  const DataType& operator()() const { return m_private_part->value(); }

  //! Valeur du scalaire
  const DataType& value() const { return m_private_part->value(); }

  /*!
   * \brief Compare la variable avec la valeur \a v.
   */
  bool isEqual(const DataType& v) const
    { return math::isEqual(m_private_part->value(),v); }

  /*!
   * \brief Compare la variable avec la valeur 0.
   * \sa isEqual().
   */
  bool isZero() const
    { return math::isZero(m_private_part->value()); }

  /*!
   * \brief Compare la variable avec la valeur \a v.
   *
   * Pour un type flottant, la comparaison se fait à un epsilon près,
   * défini dans float_info<T>::nearlyEpsilon().
   */
  bool isNearlyEqual(const DataType& v) const
    { return math::isNearlyEqual(m_private_part->value(),v); }
  /*!
   * \brief Compare la variable avec la valeur 0.
   * \sa isEqual().
   */
  bool isNearlyZero() const
    { return math::isNearlyZero(m_private_part->value()); }

  //! Affecte à la variable la valeur \a v
  ARCANE_CORE_EXPORT void assign(const DataType& v);

  //! Effectue une réduction de type \a type sur la variable
  ARCANE_CORE_EXPORT void reduce(Parallel::eReduceType type);

  ARCANE_CORE_EXPORT void swapValues(VariableRefScalarT<DataType>& rhs);

 protected:
  
 private:

  PrivatePartType* m_private_part;

 private:

  static VariableFactoryRegisterer m_auto_registerer;
  static VariableTypeInfo _buildVariableTypeInfo();
  static VariableInfo _buildVariableInfo(const VariableBuildInfo& vbi);
  static VariableRef* _autoCreate(const VariableBuildInfo& vb);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
