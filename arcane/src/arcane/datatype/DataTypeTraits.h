// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataTypeTraits.h                                            (C) 2000-2020 */
/*                                                                           */
/* Caractéristiques d'un type de donnée.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATATYPE_DATATYPETRAITS_H
#define ARCANE_DATATYPE_DATATYPETRAITS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/utils/BasicDataType.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DataTypeScalarReal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType>
class DataTypeTraitsT;
class Real2Proxy;
class Real3Proxy;
class Real3x3Proxy;
class Real2x2Proxy;
template<typename Type>
class BuiltInProxy;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de DataTypeTraitsT pour le type \c bool.
 */
template<>
class DataTypeTraitsT<bool>
{
 public:

  //! Type de donnée
  typedef bool Type;

  //! Type de donnée de base de ce type de donnée
  typedef bool BasicType;

  //! Nombre d'éléments du type de base
  static constexpr int nbBasicType() { return 1; }

  //! Nom du type de donnée
  static constexpr const char* name() { return "Bool"; }

  /*! Type de donnée
   * \todo: creer type DT_Bool a la place.
   */
  static constexpr eDataType type() { return DT_Byte; }

  //! Type du proxy associé
  typedef BuiltInProxy<bool> ProxyType;

  //! Elément initialisé à NAN
  static Type nanValue() { return 0; }

  //! Remplit les éléments de \a values avec des Nan.
  static void fillNan(ArrayView<Type> values);

  //! Valeur par défaut.
  static constexpr Type defaultValue() { return false; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de DataTypeTraitsT pour le type \c Byte.
 */
template<>
class DataTypeTraitsT<Byte>
{
 public:

  //! Type de donnée
  typedef Byte Type;

  //! Type de donnée de base de ce type de donnée
  typedef Byte BasicType;

  //! Nombre d'éléments du type de base
  static constexpr int nbBasicType() { return 1; }

  //! Nom du type de donnée
  static constexpr const char* name() { return "Byte"; }

  //! Type de donnée
  static constexpr eDataType type() { return DT_Byte; }

  //! Type de donnée de base.
  static constexpr eBasicDataType basicDataType() { return eBasicDataType::Byte; }

  //! Type du proxy associé
  typedef BuiltInProxy<Byte> ProxyType;

  //! Elément initialisé à NAN
  static constexpr Type nanValue() { return 0; }

  //! Remplit les éléments de \a values avec des Nan.
  static void fillNan(ArrayView<Type> values);

  //! Valeur par défaut.
  static constexpr Type defaultValue() { return 0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de DataTypeTraitsT pour le type \c Real.
 */
template<>
class DataTypeTraitsT<Real>
{
 public:

  //! Type de donnée
  typedef Real Type;

  //! Type de donnée de base de ce type de donnée
  typedef Real BasicType;

  //! Nombre d'éléments du type de base
  static constexpr int nbBasicType() { return 1; }

  //! Nom du type de donnée
  static constexpr const char* name() { return "Real"; }

  //! Type de donnée
  static constexpr eDataType type() { return DT_Real; }

  //! Type de donnée de base.
  // TODO: calculer automatiquement la taille
  static constexpr eBasicDataType basicDataType() { return eBasicDataType::Float64; }

  //! Type du proxy associé
  typedef BuiltInProxy<Real> ProxyType;

  //! Remplit les éléments de \a values avec des Nan.
  static void fillNan(ArrayView<Type> values);

  //! Valeur par défaut.
  static constexpr Type defaultValue() { return 0.0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de DataTypeTraitsT pour le type <tt>Integer</tt>.
 */
template<>
class DataTypeTraitsT<Int16>
{
 public:

  //! Type de donnée
  typedef Int16 Type;

  //! Type de donnée de base de ce type de donnée
  typedef Int16 BasicType;

  //! Nombre d'éléments du type de base
  static constexpr int nbBasicType() { return 1; }

  //! Nom du type de donnée
  static constexpr const char* name() { return "Int16"; }

  //! Type de donnée
  static constexpr eDataType type() { return DT_Int16; }

  //! Type de donnée de base.
  static constexpr eBasicDataType basicDataType() { return eBasicDataType::Int16; }

  //! Type du proxy associé
  typedef BuiltInProxy<Int32> ProxyType;

  //! Remplit les éléments de \a values avec des Nan.
  static void fillNan(ArrayView<Type> values);

  //! Valeur par défaut.
  static constexpr Type defaultValue() { return 0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de DataTypeTraitsT pour le type <tt>Int32</tt>.
 */
template<>
class DataTypeTraitsT<Int32>
{
 public:

  //! Type de donnée
  typedef Int32 Type;

  //! Type de donnée de base de ce type de donnée
  typedef Int32 BasicType;

  //! Nombre d'éléments du type de base
  static constexpr int nbBasicType() { return 1; }

  //! Nom du type de donnée
  static constexpr const char* name() { return "Int32"; }

  //! Type de donnée
  static constexpr eDataType type() { return DT_Int32; }

  //! Type de donnée de base.
  static constexpr eBasicDataType basicDataType() { return eBasicDataType::Int32; }

  //! Type du proxy associé
  typedef BuiltInProxy<Int32> ProxyType;

  //! Remplit les éléments de \a values avec des Nan.
  static void fillNan(ArrayView<Type> values);

  //! Valeur par défaut.
  static constexpr Type defaultValue() { return 0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de DataTypeTraitsT pour le type <tt>Int64</tt>.
 */
template<>
class DataTypeTraitsT<Int64>
{
 public:

  //! Type de donnée
  typedef Int64 Type;

  //! Type de donnée de base de ce type de donnée
  typedef Int64 BasicType;

  //! Nombre d'éléments du type de base
  static constexpr int nbBasicType() { return 1; }

  //! Nom du type de donnée
  static constexpr const char* name() { return "Int64"; }

  //! Type de donnée
  static constexpr eDataType type() { return DT_Int64; }

  //! Type de donnée de base.
  static constexpr eBasicDataType basicDataType() { return eBasicDataType::Int64; }

  //! Type du proxy associé
  typedef BuiltInProxy<Int64> ProxyType;

  //! Remplit les éléments de \a values avec des Nan.
  static void fillNan(ArrayView<Type> values);

  //! Valeur par défaut.
  static constexpr Type defaultValue() { return 0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de DataTypeTraitsT pour le type <tt>String</tt>.
 */
template<>
class DataTypeTraitsT<String>
{
 public:

  //! Type de donnée
  typedef String Type;

  //! Type de donnée de base de ce type de donnée
  typedef String BasicType;

  //! Nombre d'éléments du type de base
  static constexpr int nbBasicType() { return 1; }

  //! Nom du type de donnée
  static constexpr const char* name() { return "String"; }

  //! Type de donnée
  static constexpr eDataType type() { return DT_String; }

  //! Type du proxy associé
  typedef String ProxyType;

  //! Remplit les éléments de \a values avec des Nan.
  static void fillNan(ArrayView<Type> values);

  //! Valeur par défaut.
  static ARCANE_CORE_EXPORT Type defaultValue();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de DataTypeTraitsT pour le type <tt>Real2</tt>.
 */
template<>
class DataTypeTraitsT<Real2>
{
 public:

  //! Type de donnée
  typedef Real2 Type;

  //! Type de donnée de base de ce type de donnée
  typedef Real BasicType;

  //! Type de retour de operator[] pour ce type
  using SubscriptType = Real;

  //! Nombre d'éléments du type de base
  static constexpr int nbBasicType() { return 2; }

  //! Nom du type de donnée
  static constexpr const char* name() { return "Real2"; }

  //! Type de donnée
  static constexpr eDataType type() { return DT_Real2; }

  //! Type de donnée de base.
  static constexpr eBasicDataType basicDataType() { return DataTypeTraitsT<Real>::basicDataType(); }

  //! Type du proxy associé
  typedef Real2Proxy ProxyType;

  //! Remplit les éléments de \a values avec des Nan.
  static void fillNan(ArrayView<Type> values);

  //! Valeur par défaut.
  static ARCANE_CORE_EXPORT Type defaultValue();

 public:

  static constexpr bool HasSubscriptOperator() { return true; }
  static constexpr bool HasComponentX() { return true; }
  static constexpr bool HasComponentY() { return true; }

  using ComponentType = Real;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de DataTypeTraitsT pour le type <tt>Real3</tt>.
 */
template<>
class DataTypeTraitsT<Real3>
{
 public:

  //! Type de donnée
  typedef Real3 Type;

  //! Type de donnée de base de ce type de donnée
  typedef Real BasicType;

  //! Type de retour de operator[] pour ce type
  using SubscriptType = Real;

  //! Nombre d'éléments du type de base
  static constexpr int nbBasicType() { return 3; }

  //! Nom du type de donnée
  static constexpr const char* name() { return "Real3"; }

  //! Type de donnée
  static constexpr eDataType type() { return DT_Real3; }

  //! Type de donnée de base.
  static constexpr eBasicDataType basicDataType() { return DataTypeTraitsT<Real>::basicDataType(); }

  //! Type du proxy associé
  typedef Real3Proxy ProxyType;

  //! Remplit les éléments de \a values avec des Nan.
  static void fillNan(ArrayView<Type> values);

  //! Valeur par défaut.
  static ARCANE_CORE_EXPORT Type defaultValue();

  static constexpr bool HasSubscriptOperator() { return true; }
  static constexpr bool HasComponentX() { return true; }
  static constexpr bool HasComponentY() { return true; }
  static constexpr bool HasComponentZ() { return true; }

  using ComponentType = Real;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de DataTypeTraitsT pour le type <tt>Real2x2</tt>.
 */
template<>
class DataTypeTraitsT<Real2x2>
{
 public:

  //! Type de donnée
  typedef Real2x2 Type;

  //! Type de donnée de base de ce type de donnée
  typedef Real BasicType;

  //! Type de retour de operator[] pour ce type
  using SubscriptType = Real2;

  //! Nombre d'éléments du type de base
  static constexpr int nbBasicType() { return 4; }

  //! Nom du type de donnée
  static constexpr const char* name() { return "Real2x2"; }

  //! Type de donnée
  static constexpr eDataType type() { return DT_Real2x2; }

  //! Type de donnée de base.
  static constexpr eBasicDataType basicDataType() { return DataTypeTraitsT<Real>::basicDataType(); }

  //! Type du proxy associé
  typedef Real2x2Proxy ProxyType;

  //! Remplit les éléments de \a values avec des Nan.
  static void fillNan(ArrayView<Type> values);

  //! Valeur par défaut.
  static ARCANE_CORE_EXPORT Type defaultValue();

  static constexpr bool HasSubscriptOperator() { return true; }

  static constexpr bool HasComponentXX() { return true; }
  static constexpr bool HasComponentYX() { return true; }
  static constexpr bool HasComponentXY() { return true; }
  static constexpr bool HasComponentYY() { return true; }

  static constexpr bool HasComponentX() { return true; }
  static constexpr bool HasComponentY() { return true; }

  using ComponentType = Real2;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de DataTypeTraitsT pour le type <tt>Real3x3</tt>.
 */
template<>
class DataTypeTraitsT<Real3x3>
{
 public:

  //! Type de donnée
  typedef Real3x3 Type;

  //! Type de donnée de base de ce type de donnée
  typedef Real BasicType;

  //! Type de retour de operator[] pour ce type
  using SubscriptType = Real3;

  //! Nombre d'éléments du type de base
  static constexpr int nbBasicType() { return 9; }

  //! Nom du type de donnée
  static constexpr const char* name() { return "Real3x3"; }

  //! Type de donnée
  static constexpr eDataType type() { return DT_Real3x3; }

  //! Type de donnée de base.
  static constexpr eBasicDataType basicDataType() { return DataTypeTraitsT<Real>::basicDataType(); }

  //! Type du proxy associé
  typedef Real3x3Proxy ProxyType;

  //! Remplit les éléments de \a values avec des Nan.
  static void fillNan(ArrayView<Type> values);

  //! Valeur par défaut.
  static ARCANE_CORE_EXPORT Type defaultValue();

  static constexpr bool HasSubscriptOperator() { return true; }

  static constexpr bool HasComponentXX() { return true; }
  static constexpr bool HasComponentYX() { return true; }
  static constexpr bool HasComponentZX() { return true; }
  static constexpr bool HasComponentXY() { return true; }
  static constexpr bool HasComponentYY() { return true; }
  static constexpr bool HasComponentZY() { return true; }
  static constexpr bool HasComponentXZ() { return true; }
  static constexpr bool HasComponentYZ() { return true; }
  static constexpr bool HasComponentZZ() { return true; }

  static constexpr bool HasComponentX() { return true; }
  static constexpr bool HasComponentY() { return true; }
  static constexpr bool HasComponentZ() { return true; }

  using ComponentType = Real3;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

