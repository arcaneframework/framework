﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableDataTypeTraits.h                                    (C) 2000-2025 */
/*                                                                           */
/* Classes spécialisées pour caractériser les types de données.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLEDATATYPETRAITS_H
#define ARCANE_CORE_VARIABLEDATATYPETRAITS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/BFloat16.h"
#include "arcane/utils/Float16.h"
#include "arcane/utils/Float128.h"
#include "arcane/utils/Int128.h"

#include "arcane/core/datatype/DataTypes.h"

#include <cmath>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecriture dans la chaine \a s d'un type basique de valeur \a v.
 */
template <typename DataType> inline void
builtInDumpValue(String& s, const DataType& v)
{
  std::ostringstream sbuf;
  sbuf << v << '\0';
  s = sbuf.str();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe template d'informations sur un type d'une variable.
 *
 * Cette classe doit être spécialisée pour chaque type.
 */
template <typename DataType>
class VariableDataTypeTraitsT
{
 public:

  static eDataType type() { return DT_Unknown; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de VariableDataTypeTraitsT pour le type <tt>byte</tt>.
 */
template <>
class ARCANE_CORE_EXPORT VariableDataTypeTraitsT<Byte>
{
 public:

  //! Type du paramètre template
  typedef Byte Type;

  //! Indique si le type peut être sauvé et relu
  typedef TrueType HasDump;

  //! Indique si le type peut être subir une réduction
  typedef TrueType HasReduce;

  //! Indique si le type peut être subir une réduction Min/Max
  typedef FalseType HasReduceMinMax;

  //! Indique si le type est numérique
  typedef TrueType IsNumeric;

  typedef Byte BasicType;

  static constexpr Integer nbBasicType() { return 1; }

  typedef Byte NormType;

 public:

  //! Retourne le nom du type de la variable
  static constexpr const char* typeName() { return "Byte"; }
  //! Retourne le type de la variable
  static constexpr eDataType type() { return DT_Byte; }
  //! Ecrit dans la chaîne \a s la valeur de \a v
  static void dumpValue(String& s, const Type& v) { builtInDumpValue(s, v); }
  /*!
   * \brief Stocke la conversion de la chaîne \a s en le type #Type dans \a v
   * \retval true en cas d'échec,
   * \retval false si la conversion est un succès
   */
  static bool getValue(Type& v, const String& s) { return builtInGetValue(v, s); }

  static bool verifDifferent(Byte v1, Byte v2, Byte& diff,
                             [[maybe_unused]] bool is_nan_equal = false)
  {
    return _verifDifferent(v1, v2, diff, v1);
  }
  static bool verifDifferentNorm(Type v1, Type v2, Type& diff, Type norm,
                                 [[maybe_unused]] bool is_nan_equal = false)
  {
    return _verifDifferent(v1, v2, diff, norm);
  }

  static NormType normeMax(Byte v)
  {
    return static_cast<Byte>(math::abs(v));
  }

 private:

  static bool _verifDifferent(Type v1, Type v2, Type& diff, Type divider)
  {
    if (v1 != v2) {
      auto abs_diff = v1 - v2;
      if (!math::isZero(divider))
        abs_diff = abs_diff / divider;
      diff = static_cast<Byte>(diff);
      return true;
    }
    return false;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de VariableDataTypeTraitsT pour le type \c Real.
 */
template <>
class ARCANE_CORE_EXPORT VariableDataTypeTraitsT<Real>
{
 public:

  //! Type du paramètre template
  typedef Real Type;

  //! Indique si le type peut être sauvé et relu
  typedef TrueType HasDump;

  //! Indique si le type peut être subir une réduction
  typedef TrueType HasReduce;

  //! Indique si le type peut être subir une réduction Min/Max
  typedef TrueType HasReduceMinMax;

  //! Indique si le type est numérique
  typedef TrueType IsNumeric;

  typedef Real BasicType;

  static constexpr Integer nbBasicType() { return 1; }

  typedef Real NormType;

 public:

  //! Retourne le nom du type de la variable
  static constexpr const char* typeName() { return "Real"; }
  //! Retourne le type de la variable
  static constexpr eDataType type() { return DT_Real; }
  //! Ecrit dans la chaîne \a s la valeur de \a v
  static void dumpValue(String& s, const Type& v) { builtInDumpValue(s, v); }
  /*!
   * \brief Stocke la conversion de la chaîne \a s en le type #Type dans \a v
   * \retval true en cas d'échec,
   * \retval false si la conversion est un succès
   */
  static bool getValue(Type& v, const String& s) { return builtInGetValue(v, s); }

  static bool verifDifferentNorm(Type v1, Type v2, Type& diff, Type norm, bool is_nan_equal = false)
  {
    return _verifDifferent(v1, v2, diff, norm, is_nan_equal);
  }

  static bool verifDifferent(Type v1, Type v2, Type& diff, bool is_nan_equal = false)
  {
    return _verifDifferent(v1, v2, diff, math::abs(v1), is_nan_equal);
  }

  static NormType normeMax(Real v)
  {
    return math::abs(v);
  }

 private:

  static bool _verifDifferent(Type v1, Type v2, Type& diff, Type divider, bool is_nan_equal)
  {
    if (is_nan_equal) {
      if (std::isnan(v1) && std::isnan(v2))
        return false;
    }
    // Vérifie avant de les comparer que les deux nombres sont valides
    // pour éviter une exception flottante sur certaines plates-formes
    if (platform::isDenormalized(v1) || platform::isDenormalized(v2)) {
      diff = 1.0;
      return true;
    }
    if (v1 != v2) {
      if (divider < 1.e-100) // TH: plantait pour v1 tres petit(math::isZero(v1))
        diff = v1 - v2;
      else
        diff = (v1 - v2) / divider;
      return true;
    }
    return false;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de VariableDataTypeTraitsT pour le type \c Real.
 */
template <>
class ARCANE_CORE_EXPORT VariableDataTypeTraitsT<Float128>
{
 public:

  //! Type du paramètre template
  typedef Float128 Type;

  //! Indique si le type peut être sauvé et relu
  typedef TrueType HasDump;

  //! Indique si le type peut être subir une réduction
  typedef FalseType HasReduce;

  //! Indique si le type peut être subir une réduction Min/Max
  typedef FalseType HasReduceMinMax;

  //! Indique si le type est numérique
  typedef TrueType IsNumeric;

  typedef Float128 BasicType;

  static constexpr Int32 nbBasicType() { return 1; }

  typedef Float128 NormType;

 public:

  //! Retourne le nom du type de la variable
  static constexpr const char* typeName() { return "Float128"; }
  //! Retourne le type de la variable
  static constexpr eDataType type() { return DT_Float128; }
  //! Ecrit dans la chaîne \a s la valeur de \a v
  static void dumpValue(String& s, const Type& v) { builtInDumpValue(s, v); }
  /*!
   * \brief Stocke la conversion de la chaîne \a s en le type #Type dans \a v
   * \retval true en cas d'échec,
   * \retval false si la conversion est un succès
   */
  static bool getValue(Type& v, const String& s) { return builtInGetValue(v, s); }

  static bool verifDifferent(Type v1, Type v2, Type& diff, [[maybe_unused]] bool is_nan_equal = false)
  {
    return _verifDifferent(v1, v2, diff, math::abs(v1));
  }

  static bool verifDifferentNorm(Type v1, Type v2, Type& diff, NormType norm, [[maybe_unused]] bool is_nan_equal)
  {
    return _verifDifferent(v1, v2, diff, norm);
  }

  static NormType normeMax(Float128 v)
  {
    return math::abs(v);
  }

 private:

  static bool _verifDifferent(Type v1, Type v2, Type& diff, Type divider)
  {
    if (v1 != v2) {
      if (divider < 1.e-100) // TH: plantait pour v1 tres petit(math::isZero(v1))
        diff = v1 - v2;
      else
        diff = (v1 - v2) / divider;
      return true;
    }
    return false;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de VariableDataTypeTraitsT pour le type <tt>Int8</tt>.
 */
template <>
class ARCANE_CORE_EXPORT VariableDataTypeTraitsT<Int8>
{
 public:

  //! Type du paramètre template
  typedef Int8 Type;

  //! Indique si le type peut être sauvé et relu
  typedef TrueType HasDump;

  //! Indique si le type peut être subir une réduction
  typedef TrueType HasReduce;

  //! Indique si le type peut être subir une réduction Min/Max
  typedef FalseType HasReduceMinMax;

  //! Indique si le type est numérique
  typedef TrueType IsNumeric;

  typedef Int8 BasicType;

  static constexpr Int32 nbBasicType() { return 1; }

  typedef Int8 NormType;

 public:

  //! Retourne le nom du type de la variable
  static constexpr const char* typeName() { return "Int8"; }
  //! Retourne le type de la variable
  static constexpr eDataType type() { return DT_Int8; }
  //! Ecrit dans la chaîne \a s la valeur de \a v
  static void dumpValue(String& s, const Type& v) { builtInDumpValue(s, v); }
  /*!
   * \brief Stocke la conversion de la chaîne \a s en le type #Type dans \a v
   * \retval true en cas d'échec,
   * \retval false si la conversion est un succès
   */
  static bool getValue(Type& v, const String& s) { return builtInGetValue(v, s); }
  static bool verifDifferent(Int8 v1, Int8 v2, Int8& diff,
                             [[maybe_unused]] bool is_nan_equal = false)
  {
    if (v1 != v2) {
      if (math::isZero(v1))
        diff = (Int8)(v1 - v2);
      else
        diff = (Int8)((v1 - v2) / v1);
      return true;
    }
    return false;
  }
  static bool verifDifferentNorm(Type v1, Type v2, Type& diff,
                                 [[maybe_unused]] NormType norm,
                                 [[maybe_unused]] bool is_nan_equal = false)
  {
    return verifDifferent(v1, v2, diff, is_nan_equal);
  }
  static NormType normeMax(Int8 v)
  {
    return static_cast<Int8>(math::abs(v));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de VariableDataTypeTraitsT pour le type <tt>Int16</tt>.
 */
template <>
class ARCANE_CORE_EXPORT VariableDataTypeTraitsT<Int16>
{
 public:

  //! Type du paramètre template
  typedef Int16 Type;

  //! Indique si le type peut être sauvé et relu
  typedef TrueType HasDump;

  //! Indique si le type peut être subir une réduction
  typedef TrueType HasReduce;

  //! Indique si le type peut être subir une réduction Min/Max
  typedef TrueType HasReduceMinMax;

  //! Indique si le type est numérique
  typedef TrueType IsNumeric;

  typedef Int16 BasicType;

  static constexpr Integer nbBasicType() { return 1; }

  typedef Int16 NormType;

 public:

  //! Retourne le nom du type de la variable
  static constexpr const char* typeName() { return "Int16"; }
  //! Retourne le type de la variable
  static constexpr eDataType type() { return DT_Int16; }
  //! Ecrit dans la chaîne \a s la valeur de \a v
  static void dumpValue(String& s, const Type& v) { builtInDumpValue(s, v); }
  /*!
   * \brief Stocke la conversion de la chaîne \a s en le type #Type dans \a v
   * \retval true en cas d'échec,
   * \retval false si la conversion est un succès
   */
  static bool getValue(Type& v, const String& s) { return builtInGetValue(v, s); }
  static bool verifDifferent(Int16 v1, Int16 v2, Int16& diff,
                             [[maybe_unused]] bool is_nan_equal = false)
  {
    if (v1 != v2) {
      if (math::isZero(v1))
        diff = (Int16)(v1 - v2);
      else
        diff = (Int16)((v1 - v2) / v1);
      return true;
    }
    return false;
  }
  static bool verifDifferentNorm(Type v1, Type v2, Type& diff,
                                 [[maybe_unused]] NormType norm,
                                 [[maybe_unused]] bool is_nan_equal = false)
  {
    return verifDifferent(v1, v2, diff, is_nan_equal);
  }
  static NormType normeMax(Int16 v)
  {
    return math::abs(v);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de VariableDataTypeTraitsT pour le type <tt>Int32</tt>.
 */
template <>
class ARCANE_CORE_EXPORT VariableDataTypeTraitsT<Int32>
{
 public:

  //! Type du paramètre template
  typedef Int32 Type;

  //! Indique si le type peut être sauvé et relu
  typedef TrueType HasDump;

  //! Indique si le type peut être subir une réduction
  typedef TrueType HasReduce;

  //! Indique si le type peut être subir une réduction Min/Max
  typedef TrueType HasReduceMinMax;

  //! Indique si le type est numérique
  typedef TrueType IsNumeric;

  typedef Int32 BasicType;

  static constexpr Integer nbBasicType() { return 1; }

  typedef Int32 NormType;

 public:

  //! Retourne le nom du type de la variable
  static constexpr const char* typeName() { return "Int32"; }
  //! Retourne le type de la variable
  static constexpr eDataType type() { return DT_Int32; }
  //! Ecrit dans la chaîne \a s la valeur de \a v
  static void dumpValue(String& s, const Type& v) { builtInDumpValue(s, v); }
  /*!
   * \brief Stocke la conversion de la chaîne \a s en le type #Type dans \a v
   * \retval true en cas d'échec,
   * \retval false si la conversion est un succès
   */
  static bool getValue(Type& v, const String& s) { return builtInGetValue(v, s); }
  static bool verifDifferent(Int32 v1, Int32 v2, Int32& diff,
                             [[maybe_unused]] bool is_nan_equal = false)
  {
    if (v1 != v2) {
      if (math::isZero(v1))
        diff = v1 - v2;
      else
        diff = (v1 - v2) / v1;
      return true;
    }
    return false;
  }
  static bool verifDifferentNorm(Type v1, Type v2, Type& diff,
                                 [[maybe_unused]] NormType norm,
                                 [[maybe_unused]] bool is_nan_equal = false)
  {
    return verifDifferent(v1, v2, diff, is_nan_equal);
  }

  static NormType normeMax(Int32 v)
  {
    return math::abs(v);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de VariableDataTypeTraitsT pour le type <tt>Int64</tt>.
 */
template <>
class ARCANE_CORE_EXPORT VariableDataTypeTraitsT<Int64>
{
 public:

  //! Type du paramètre template
  typedef Int64 Type;

  //! Indique si le type peut être sauvé et relu
  typedef TrueType HasDump;

  //! Indique si le type peut être subir une réduction
  typedef TrueType HasReduce;

  //! Indique si le type peut être subir une réduction Min/Max
  typedef TrueType HasReduceMinMax;

  //! Indique si le type est numérique
  typedef TrueType IsNumeric;

  typedef Int64 BasicType;

  static constexpr Integer nbBasicType() { return 1; }

  typedef Int64 NormType;

 public:

  //! Retourne le nom du type de la variable
  static constexpr const char* typeName() { return "Int64"; }
  //! Retourne le type de la variable
  static constexpr eDataType type() { return DT_Int64; }
  //! Ecrit dans la chaîne \a s la valeur de \a v
  static void dumpValue(String& s, const Type& v) { builtInDumpValue(s, v); }
  /*!
   * \brief Stocke la conversion de la chaîne \a s en le type #Type dans \a v
   * \retval true en cas d'échec,
   * \retval false si la conversion est un succès
   */
  static bool getValue(Type& v, const String& s) { return builtInGetValue(v, s); }

  static bool verifDifferent(Type v1, Type v2, Type& diff,
                             [[maybe_unused]] bool is_nan_equal = false)
  {
    if (v1 != v2) {
      if (math::isZero(v1))
        diff = v1 - v2;
      else
        diff = (v1 - v2) / v1;
      return true;
    }
    return false;
  }

  static bool verifDifferentNorm(Type v1, Type v2, Type& diff,
                                 [[maybe_unused]] NormType norm,
                                 [[maybe_unused]] bool is_nan_equal = false)
  {
    return verifDifferent(v1, v2, diff, is_nan_equal);
  }

  static NormType normeMax(Int64 v)
  {
    return v;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de VariableDataTypeTraitsT pour le type <tt>Int128</tt>.
 */
template <>
class ARCANE_CORE_EXPORT VariableDataTypeTraitsT<Int128>
{
 public:

  //! Type du paramètre template
  typedef Int128 Type;

  //! Indique si le type peut être sauvé et relu
  typedef TrueType HasDump;

  //! Indique si le type peut être subir une réduction
  typedef FalseType HasReduce;

  //! Indique si le type peut être subir une réduction Min/Max
  typedef FalseType HasReduceMinMax;

  //! Indique si le type est numérique
  typedef TrueType IsNumeric;

  typedef Int128 BasicType;

  static constexpr Int32 nbBasicType() { return 1; }

  typedef Int128 NormType;

 public:

  //! Retourne le nom du type de la variable
  static constexpr const char* typeName() { return "Int128"; }
  //! Retourne le type de la variable
  static constexpr eDataType type() { return DT_Int128; }
  //! Ecrit dans la chaîne \a s la valeur de \a v
  static void dumpValue(String& s, const Type& v) { builtInDumpValue(s, v); }
  /*!
   * \brief Stocke la conversion de la chaîne \a s en le type #Type dans \a v
   * \retval true en cas d'échec,
   * \retval false si la conversion est un succès
   */
  static bool getValue(Type& v, const String& s) { return builtInGetValue(v, s); }

  static bool verifDifferent(Type v1, Type v2, Type& diff,
                             [[maybe_unused]] bool is_nan_equal = false)
  {
    if (v1 != v2) {
      if (math::isZero(v1))
        diff = v1 - v2;
      else
        diff = (v1 - v2) / v1;
      return true;
    }
    return false;
  }

  static bool verifDifferentNorm(Type v1, Type v2, Type& diff,
                                 [[maybe_unused]] NormType norm,
                                 [[maybe_unused]] bool is_nan_equal = false)
  {
    return verifDifferent(v1, v2, diff, is_nan_equal);
  }

  static NormType normeMax(Int16 v)
  {
    return static_cast<Int16>(math::abs(v));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de VariableDataTypeTraitsT pour le type <tt>String</tt>.
 */
template <>
class ARCANE_CORE_EXPORT VariableDataTypeTraitsT<String>
{
 public:

  //! Type du paramètre template
  typedef String Type;

  //! Indique si le type peut être sauvé et relu
  typedef TrueType HasDump;

  //! Indique si le type peut être subir une réduction
  typedef FalseType HasReduce;

  //! Indique si le type peut être subir une réduction Min/Max
  typedef FalseType HasReduceMinMax;

  //! Indique si le type est numérique
  typedef FalseType IsNumeric;

  typedef String BasicType;

  // Uniquement utilisé pour compiler les routines de comparaison de valeurs.
  using NormType = String;

  static constexpr Integer nbBasicType() { return 1; }

 public:

  //! Retourne le nom du type de la variable
  static constexpr const char* typeName() { return "String"; }
  //! Retourne le type de la variable
  static constexpr eDataType type() { return DT_String; }
  //! Ecrit dans la chaîne \a s la valeur de \a v
  static void dumpValue(String& s, const Type& v) { s = v; }
  /*!
   * \brief Stocke la conversion de la chaîne \a s en le type #Type dans \a v
   * \retval true en cas d'échec,
   * \retval false si la conversion est un succès
   */
  static bool getValue(Type& v, const String& s) { return builtInGetValue(v, s); }

  static bool verifDifferent(const Type v1, const Type& v2, Type& diff,
                             [[maybe_unused]] bool is_nan_equal = false)
  {
    diff = v1;
    return (v1 != v2);
  }
  static bool verifDifferentNorm(const Type v1, const Type& v2, Type& diff,
                                 [[maybe_unused]] const String& divider,
                                 [[maybe_unused]] bool is_nan_equal = false)
  {
    return verifDifferent(v1, v2, diff, is_nan_equal);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de VariableDataTypeTraitsT pour le type \c BFloat16.
 */
template <>
class ARCANE_CORE_EXPORT VariableDataTypeTraitsT<BFloat16>
{
 public:

  //! Type du paramètre template
  typedef BFloat16 Type;

  //! Indique si le type peut être sauvé et relu
  typedef FalseType HasDump;

  //! Indique si le type peut être subir une réduction
  typedef FalseType HasReduce;

  //! Indique si le type peut être subir une réduction Min/Max
  typedef FalseType HasReduceMinMax;

  //! Indique si le type est numérique
  typedef TrueType IsNumeric;

  typedef BFloat16 BasicType;

  static constexpr Integer nbBasicType() { return 1; }

  typedef BFloat16 NormType;

 public:

  //! Retourne le nom du type de la variable
  static constexpr const char* typeName() { return "BFloat16"; }
  //! Retourne le type de la variable
  static constexpr eDataType type() { return DT_BFloat16; }
  //! Ecrit dans la chaîne \a s la valeur de \a v
  static void dumpValue(String& s, const Type& v) { builtInDumpValue(s, v); }
  /*!
   * \brief Stocke la conversion de la chaîne \a s en le type #Type dans \a v
   * \retval true en cas d'échec,
   * \retval false si la conversion est un succès
   */
  static bool getValue(Type& v, const String& s) { return builtInGetValue(v, s); }

  static bool verifDifferent(float v1, float v2, BFloat16& diff,
                             [[maybe_unused]] bool is_nan_equal = false)
  {
    return _verifDifferent(v1, v2, diff, math::abs(v1));
  }

  static bool verifDifferentNorm(float v1, float v2, BFloat16& diff, NormType norm,
                                 [[maybe_unused]] bool is_nan_equal = false)
  {
    return _verifDifferent(v1, v2, diff, norm);
  }

  static BFloat16 normeMax(float v)
  {
    return static_cast<Type>(math::abs(v));
  }

 private:

  static bool _verifDifferent(float v1, float v2, BFloat16& diff, float divider)
  {
    if (v1 != v2) {
      float fdiff = 0.0;
      if (divider != 0.0)
        fdiff = v1 - v2;
      else
        fdiff = (v1 - v2) / divider;
      diff = static_cast<Type>(fdiff);
      return true;
    }
    return false;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de VariableDataTypeTraitsT pour le type \c Float16.
 */
template <>
class ARCANE_CORE_EXPORT VariableDataTypeTraitsT<Float16>
{
 public:

  //! Type du paramètre template
  typedef Float16 Type;

  //! Indique si le type peut être sauvé et relu
  typedef FalseType HasDump;

  //! Indique si le type peut être subir une réduction
  typedef FalseType HasReduce;

  //! Indique si le type peut être subir une réduction Min/Max
  typedef FalseType HasReduceMinMax;

  //! Indique si le type est numérique
  typedef TrueType IsNumeric;

  typedef Float16 BasicType;

  static constexpr Integer nbBasicType() { return 1; }

  typedef Float16 NormType;

 public:

  //! Retourne le nom du type de la variable
  static constexpr const char* typeName() { return "Float16"; }
  //! Retourne le type de la variable
  static constexpr eDataType type() { return DT_Float16; }
  //! Ecrit dans la chaîne \a s la valeur de \a v
  static void dumpValue(String& s, const Type& v) { builtInDumpValue(s, v); }
  /*!
   * \brief Stocke la conversion de la chaîne \a s en le type #Type dans \a v
   * \retval true en cas d'échec,
   * \retval false si la conversion est un succès
   */
  static bool getValue(Type& v, const String& s) { return builtInGetValue(v, s); }

  static bool verifDifferent(float v1, float v2, Float16& diff,
                             [[maybe_unused]] bool is_nan_equal = false)
  {
    return _verifDifferent(v1, v2, diff, math::abs(v1));
  }

  static bool verifDifferentNorm(float v1, float v2, Float16& diff, NormType norm,
                                 [[maybe_unused]] bool is_nan_equal = false)
  {
    return _verifDifferent(v1, v2, diff, norm);
  }

  static Float16 normeMax(Float16 v)
  {
    return static_cast<Float16>(math::abs(v));
  }

 private:

  static bool _verifDifferent(float v1, float v2, Float16& diff, float divider)
  {
    if (v1 != v2) {
      float fdiff = 0.0;
      if (divider != 0.0)
        fdiff = v1 - v2;
      else
        fdiff = (v1 - v2) / divider;
      diff = static_cast<Type>(fdiff);
      return true;
    }
    return false;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de VariableDataTypeTraitsT pour le type \c Float32.
 */
template <>
class ARCANE_CORE_EXPORT VariableDataTypeTraitsT<Float32>
{
 public:

  //! Type du paramètre template
  typedef Float32 Type;

  //! Indique si le type peut être sauvé et relu
  typedef FalseType HasDump;

  //! Indique si le type peut être subir une réduction
  typedef TrueType HasReduce;

  //! Indique si le type peut être subir une réduction Min/Max
  typedef TrueType HasReduceMinMax;

  //! Indique si le type est numérique
  typedef TrueType IsNumeric;

  typedef Real BasicType;

  static constexpr Integer nbBasicType() { return 1; }

  typedef Float32 NormType;

 public:

  //! Retourne le nom du type de la variable
  static constexpr const char* typeName() { return "Float32"; }
  //! Retourne le type de la variable
  static constexpr eDataType type() { return DT_Float32; }
  //! Ecrit dans la chaîne \a s la valeur de \a v
  static void dumpValue(String& s, const Type& v) { builtInDumpValue(s, v); }
  /*!
   * \brief Stocke la conversion de la chaîne \a s en le type #Type dans \a v
   * \retval true en cas d'échec,
   * \retval false si la conversion est un succès
   */
  static bool getValue(Type& v, const String& s) { return builtInGetValue(v, s); }

  static bool verifDifferentNorm(Type v1, Type v2, Type& diff, NormType norm, bool is_nan_equal = false)
  {
    return _verifDifferent(v1, v2, diff, norm, is_nan_equal);
  }

  static bool verifDifferent(Type v1, Type v2, Type& diff, bool is_nan_equal = false)
  {
    return _verifDifferent(v1, v2, diff, math::abs(v1), is_nan_equal);
  }

  static Float32 normeMax(Float32 v)
  {
    return math::abs(v);
  }

 private:

  static bool _verifDifferent(Type v1, Type v2, Type& diff, Type divider, bool is_nan_equal)
  {
    if (is_nan_equal) {
      if (std::isnan(v1) && std::isnan(v2))
        return false;
    }
    // Vérifie avant de les comparer que les deux nombres sont valides
    // pour éviter une exception flottante sur certaines plates-formes
    if (platform::isDenormalized(v1) || platform::isDenormalized(v2)) {
      diff = 1.0;
      return true;
    }
    if (v1 != v2) {
      if (divider < 1.e-40)
        diff = v1 - v2;
      else
        diff = (v1 - v2) / divider;
      return true;
    }
    return false;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de VariableDataTypeTraitsT pour le type <tt>Real2</tt>.
 */
template <>
class ARCANE_CORE_EXPORT VariableDataTypeTraitsT<Real2>
{
 public:

  //! Type du paramètre template
  typedef Real2 Type;

  //! Indique si le type peut être sauvé et relu
  typedef TrueType HasDump;

  //! Indique si le type peut être subir une réduction
  typedef TrueType HasReduce;

  //! Indique si le type peut être subir une réduction Min/Max
  typedef TrueType HasReduceMinMax;

  //! Indique si le type est numérique
  typedef TrueType IsNumeric;

  typedef Real BasicType;

  static constexpr Integer nbBasicType() { return 2; }

  typedef Real NormType;

 private:

  using SubTraits = VariableDataTypeTraitsT<Real>;

 public:

  //! Retourne le nom du type de la variable
  static constexpr const char* typeName() { return "Real2"; }
  //! Retourne le type de la variable
  static constexpr eDataType type() { return DT_Real2; }
  //! Ecrit dans la chaîne \a s la valeur de \a v
  static void dumpValue(String& s, const Type&) { s = "N/A"; }
  /*!
   * \brief Stocke la conversion de la chaîne \a s en le type #Type dans \a v
   * \retval true en cas d'échec,
   * \retval false si la conversion est un succès
   */
  static bool getValue(Type& v, const String& s) { return builtInGetValue(v, s); }

  static bool verifDifferent(Real2 v1, Real2 v2, Real2& diff, bool is_nan_equal = false)
  {
    bool is_different = false;
    is_different |= SubTraits::verifDifferent(v1.x, v2.x, diff.x, is_nan_equal);
    is_different |= SubTraits::verifDifferent(v1.y, v2.y, diff.y, is_nan_equal);
    return is_different;
  }

  static bool verifDifferentNorm(const Real2& v1, const Real2& v2, Real2& diff,
                                 NormType norm, [[maybe_unused]] bool is_nan_equal)
  {
    if (norm < 1.e-100) {
      diff.x = math::abs(v2.x);
      diff.y = math::abs(v2.y);
    }
    else {
      diff.x = math::abs(v2.x - v1.x) / norm;
      diff.y = math::abs(v2.y - v1.y) / norm;
    }
    bool is_different = (normeMax(diff) != 0.);
    return is_different;
  }

  static NormType normeMax(const Real2& v)
  {
    Real vx = SubTraits::normeMax(v.x);
    Real vy = SubTraits::normeMax(v.y);
    return math::max(vx, vy);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de VariableDataTypeTraitsT pour le type <tt>Real3</tt>.
 */
template <>
class ARCANE_CORE_EXPORT VariableDataTypeTraitsT<Real3>
{
 public:

  //! Type du paramètre template
  typedef Real3 Type;

  //! Indique si le type peut être sauvé et relu
  typedef TrueType HasDump;

  //! Indique si le type peut être subir une réduction
  typedef TrueType HasReduce;

  //! Indique si le type peut être subir une réduction Min/Max
  typedef TrueType HasReduceMinMax;

  //! Indique si le type est numérique
  typedef TrueType IsNumeric;

  typedef Real BasicType;

  static constexpr Integer nbBasicType() { return 3; }

  typedef Real NormType;

 private:

  using SubTraits = VariableDataTypeTraitsT<Real>;

 public:

  //! Retourne le nom du type de la variable
  static constexpr const char* typeName() { return "Real3"; }
  //! Retourne le type de la variable
  static constexpr eDataType type() { return DT_Real3; }
  //! Ecrit dans la chaîne \a s la valeur de \a v
  static void dumpValue(String& s, const Type&) { s = "N/A"; }
  /*!
   * \brief Stocke la conversion de la chaîne \a s en le type #Type dans \a v
   * \retval true en cas d'échec,
   * \retval false si la conversion est un succès
   */
  static bool getValue(Type& v, const String& s) { return builtInGetValue(v, s); }

  static bool verifDifferent(Real3 v1, Real3 v2, Real3& diff, bool is_nan_equal = false)
  {
    bool is_different = false;
    is_different |= SubTraits::verifDifferent(v1.x, v2.x, diff.x, is_nan_equal);
    is_different |= SubTraits::verifDifferent(v1.y, v2.y, diff.y, is_nan_equal);
    is_different |= SubTraits::verifDifferent(v1.z, v2.z, diff.z, is_nan_equal);
    return is_different;
  }

  static bool verifDifferentNorm(const Real3& v1, const Real3& v2, Real3& diff,
                                 NormType norm, [[maybe_unused]] bool is_nan_equal)
  {
    if (norm < 1.e-100) {
      diff.x = math::abs(v2.x);
      diff.y = math::abs(v2.y);
      diff.z = math::abs(v2.z);
    }
    else {
      diff.x = math::abs(v2.x - v1.x) / norm;
      diff.y = math::abs(v2.y - v1.y) / norm;
      diff.z = math::abs(v2.z - v1.z) / norm;
    }
    bool is_different = (normeMax(diff) != 0.);
    return is_different;
  }

  static NormType normeMax(const Real3& v)
  {
    Real vx = SubTraits::normeMax(v.x);
    Real vy = SubTraits::normeMax(v.y);
    Real vz = SubTraits::normeMax(v.z);
    return math::max(vx, math::max(vy, vz));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de VariableDataTypeTraitsT pour le type <tt>Real3x3</tt>.
 */
template <>
class ARCANE_CORE_EXPORT VariableDataTypeTraitsT<Real2x2>
{
 public:

  //! Type du paramètre template
  typedef Real2x2 Type;

  //! Indique si le type peut être sauvé et relu
  typedef TrueType HasDump;

  //! Indique si le type peut être subir une réduction
  typedef TrueType HasReduce;

  //! Indique si le type peut être subir une réduction Min/Max
  typedef TrueType HasReduceMinMax;

  //! Indique si le type est numérique
  typedef TrueType IsNumeric;

  typedef Real BasicType;

  static constexpr Integer nbBasicType() { return 4; }

  typedef Real NormType;

 private:

  using SubTraits = VariableDataTypeTraitsT<Real2>;

 public:

  //! Retourne le nom du type de la variable
  static constexpr const char* typeName() { return "Real2x2"; }
  //! Retourne le type de la variable
  static constexpr eDataType type() { return DT_Real2x2; }
  //! Ecrit dans la chaîne \a s la valeur de \a v
  static void dumpValue(String& s, const Type&) { s = "N/A"; }
  /*!
   * \brief Stocke la conversion de la chaîne \a s en le type #Type dans \a v
   * \retval true en cas d'échec,
   * \retval false si la conversion est un succès
   */
  static bool getValue(Type&, const String&) { return true; }

  static bool verifDifferent(Real2x2 v1, Real2x2 v2, Real2x2& diff, bool is_nan_equal = false)
  {
    bool is_different = false;
    is_different |= SubTraits::verifDifferent(v1.x, v2.x, diff.x, is_nan_equal);
    is_different |= SubTraits::verifDifferent(v1.y, v2.y, diff.y, is_nan_equal);
    return is_different;
  }

  static bool verifDifferentNorm(const Real2x2& v1, const Real2x2& v2, Real2x2& diff,
                                 NormType norm, [[maybe_unused]] bool is_nan_equal)
  {
    if (norm < 1.e-100) {
      diff.x.x = math::abs(v2.x.x);
      diff.x.y = math::abs(v2.x.y);

      diff.y.x = math::abs(v2.y.x);
      diff.y.y = math::abs(v2.y.y);
    }
    else {
      diff.x.x = math::abs(v2.x.x - v1.x.x) / norm;
      diff.x.y = math::abs(v2.x.y - v1.x.y) / norm;

      diff.y.x = math::abs(v2.y.x - v1.y.x) / norm;
      diff.y.y = math::abs(v2.y.y - v1.y.y) / norm;
    }
    bool is_different = (normeMax(diff) != 0.);
    return is_different;
  }

  static NormType normeMax(const Real2x2& v)
  {
    Real vx = SubTraits::normeMax(v.x);
    Real vy = SubTraits::normeMax(v.y);
    return SubTraits::normeMax(Real2(vx, vy));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Spécialisation de VariableDataTypeTraitsT pour le type <tt>Real3x3</tt>.
 */
template <>
class ARCANE_CORE_EXPORT VariableDataTypeTraitsT<Real3x3>
{
 public:

  //! Type du paramètre template
  typedef Real3x3 Type;

  //! Indique si le type peut être sauvé et relu
  typedef TrueType HasDump;

  //! Indique si le type peut être subir une réduction
  typedef TrueType HasReduce;

  //! Indique si le type peut être subir une réduction Min/Max
  typedef TrueType HasReduceMinMax;

  //! Indique si le type est numérique
  typedef TrueType IsNumeric;

  typedef Real BasicType;

  static constexpr Integer nbBasicType() { return 9; }

  typedef Real NormType;

 private:

  using SubTraits = VariableDataTypeTraitsT<Real3>;

 public:

  //! Retourne le nom du type de la variable
  static constexpr const char* typeName() { return "Real3x3"; }
  //! Retourne le type de la variable
  static constexpr eDataType type() { return DT_Real3x3; }
  //! Ecrit dans la chaîne \a s la valeur de \a v
  static void dumpValue(String& s, const Type&) { s = "N/A"; }
  /*!
   * \brief Stocke la conversion de la chaîne \a s en le type #Type dans \a v
   * \retval true en cas d'échec,
   * \retval false si la conversion est un succès
   */
  static bool getValue(Type&, const String&) { return true; }

  static bool verifDifferent(Real3x3 v1, Real3x3 v2, Real3x3& diff, bool is_nan_equal = false)
  {
    bool is_different = false;
    is_different |= SubTraits::verifDifferent(v1.x, v2.x, diff.x, is_nan_equal);
    is_different |= SubTraits::verifDifferent(v1.y, v2.y, diff.y, is_nan_equal);
    is_different |= SubTraits::verifDifferent(v1.z, v2.z, diff.z, is_nan_equal);
    return is_different;
  }

  static bool verifDifferentNorm(const Real3x3& v1, const Real3x3& v2, Real3x3& diff,
                                 NormType norm, [[maybe_unused]] bool is_nan_equal)
  {
    if (norm < 1.e-100) {
      diff.x.x = math::abs(v2.x.x);
      diff.x.y = math::abs(v2.x.y);
      diff.x.z = math::abs(v2.x.z);

      diff.y.x = math::abs(v2.y.x);
      diff.y.y = math::abs(v2.y.y);
      diff.y.z = math::abs(v2.y.z);

      diff.z.x = math::abs(v2.z.x);
      diff.z.y = math::abs(v2.z.y);
      diff.z.z = math::abs(v2.z.z);
    }
    else {
      diff.x.x = math::abs(v2.x.x - v1.x.x) / norm;
      diff.x.y = math::abs(v2.x.y - v1.x.y) / norm;
      diff.x.z = math::abs(v2.x.z - v1.x.z) / norm;

      diff.y.x = math::abs(v2.y.x - v1.y.x) / norm;
      diff.y.y = math::abs(v2.y.y - v1.y.y) / norm;
      diff.y.z = math::abs(v2.y.z - v1.y.z) / norm;

      diff.z.x = math::abs(v2.z.x - v1.z.x) / norm;
      diff.z.y = math::abs(v2.z.y - v1.z.y) / norm;
      diff.z.z = math::abs(v2.z.z - v1.z.z) / norm;
    }
    bool is_different = (normeMax(diff) != 0.);
    return is_different;
  }

  static NormType normeMax(const Real3x3& v)
  {
    Real vx = SubTraits::normeMax(v.x);
    Real vy = SubTraits::normeMax(v.y);
    Real vz = SubTraits::normeMax(v.z);
    return VariableDataTypeTraitsT<Real3>::normeMax(Real3(vx, vy, vz));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
