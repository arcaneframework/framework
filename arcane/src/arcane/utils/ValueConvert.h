// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ValueConvert.h                                              (C) 2000-2024 */
/*                                                                           */
/* Fonctions pour convertir une chaîne de caractère en un type donné.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_VALUECONVERT_H
#define ARCANE_UTILS_VALUECONVERT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iostream.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/BFloat16.h"
#include "arcane/utils/Float16.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti la valeur de la chaîne \a s dans le type basique \a T
 * et stocke la valeur dans \a v.
 *
 * \retval true en cas d'échec.
 * \retval false en cas de succès
 */
template<class T> inline bool
builtInGetValue(T& v,const String& s)
{
  T read_val = T();
  std::istringstream sbuf(s.localstr());
  sbuf >> read_val;
  if (sbuf.fail() || sbuf.bad())
    return true;
  if (!sbuf.eof())
    return true;
  v = read_val;
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(String& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(double& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(float& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(int& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(unsigned int& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(long& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(long long& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(short& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(unsigned short& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(unsigned long& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(unsigned long long& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Float16& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(BFloat16& v,const String& s);
#ifdef ARCANE_REAL_NOT_BUILTIN
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real& v,const String& s);
#endif
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(RealArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2Array& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3Array& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2x2Array& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3x3Array& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int8Array& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int16Array& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int32Array& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int64Array& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(BFloat16Array& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Float16Array& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Float32Array& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(BoolArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(StringArray& v,const String& s);

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(RealSharedArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2SharedArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3SharedArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2x2SharedArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3x3SharedArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int8SharedArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int16SharedArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int32SharedArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int64SharedArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(BFloat16SharedArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Float16SharedArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Float32SharedArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(BoolSharedArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(StringSharedArray& v,const String& s);

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(RealUniqueArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2UniqueArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3UniqueArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2x2UniqueArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3x3UniqueArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int8UniqueArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int16UniqueArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int32UniqueArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int64UniqueArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(BFloat16UniqueArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Float16UniqueArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Float32UniqueArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(BoolUniqueArray& v,const String& s);
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(StringUniqueArray& v,const String& s);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti la valeur de la chaîne \a s en un booléen
 *
 * Stocke la valeur convertie dans \a v.
 *
 * \retval true en cas d'échec et la valeur de \a v reste inchangée
 * \retval false en cas de succès
 */
inline bool
builtInGetValue(bool& v,const String& s)
{
  if (s.null())
    return true;
  if (s=="false" || s=="faux" || s=="0"){
    v = false;
    return false;
  }

  if (s=="true" || s=="vrai" || s=="1"){
    v = true;
    return false;
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//@{
/*!
 * \brief Converti la valeur \a v dans la chaîne \a s.
 *
 * \retval true en cas d'échec.
 * \retval false en cas de succès
 */
ARCANE_UTILS_EXPORT bool builtInPutValue(const String& v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(double v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(float v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(int v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(unsigned int v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(long v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(long long v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(short v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(unsigned short v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(unsigned long v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(unsigned long long v,String& s);
#ifdef ARCANE_REAL_NOT_BUILTIN
ARCANE_UTILS_EXPORT bool builtInPutValue(Real v,String& s);
#endif
ARCANE_UTILS_EXPORT bool builtInPutValue(Real2 v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Real3 v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(const Real2x2& v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(const Real3x3& v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const Real> v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const Real2> v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const Real3> v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const Real2x2> v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const Real3x3> v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const Int16> v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const Int32> v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const Int64> v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const bool> v,String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const String> v,String& s);
//@}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline const char* typeToName(bool){ return "boolean"; }
inline const char* typeToName(Real){ return "real"; }
inline const char* typeToName(Real2){ return "real2"; }
inline const char* typeToName(Real3){ return "real3"; }
inline const char* typeToName(Real2x2){ return "real2x2"; }
inline const char* typeToName(Real3x3){ return "real3x3"; }
inline const char* typeToName(short) { return "short"; }
inline const char* typeToName(unsigned short) { return "unsigned short"; }
inline const char* typeToName(int) { return "integer"; }
inline const char* typeToName(long) { return "long"; }
inline const char* typeToName(unsigned long) { return "unsigned long"; }
inline const char* typeToName(unsigned long long) { return "unsigned long long"; }
inline const char* typeToName(const String&) { return "string"; }
inline const char* typeToName(long long) { return "long long"; }
inline const char* typeToName(unsigned int) { return "unsigned integer"; }
inline const char* typeToName(BFloat16){ return "bfloat16"; }
inline const char* typeToName(Float16){ return "float16"; }
inline const char* typeToName(Float32){ return "float32"; }
inline const char* typeToName(const StringArray&){ return "string[]"; }
inline const char* typeToName(const BoolArray&){ return "boolean[]"; }
inline const char* typeToName(const RealArray&){ return "real[]"; }
inline const char* typeToName(const Real2Array&){ return "real2[]"; }
inline const char* typeToName(const Real3Array&){ return "real3[]"; }
inline const char* typeToName(const Real2x2Array&){ return "real2x2[]"; }
inline const char* typeToName(const Real3x3Array&){ return "real3x3[]"; }
inline const char* typeToName(const Int16Array&) { return "Int16[]"; }
inline const char* typeToName(const Int32Array&) { return "Int32[]"; }
inline const char* typeToName(const Int64Array&) { return "Int64[]"; }


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Convert
{

template<typename T>
class Type;

template<typename T>
class ScalarType
{
 public:

  //! Convertit \a s en le type \a T
  ARCANE_UTILS_EXPORT static std::optional<T> tryParse(StringView s);

  /*!
   * \brief Convertit \a s en le type \a T.
   *
   * Si \a s.empty() est vrai, alors retourne \a default_value.
   */
  static std::optional<T>
  tryParseIfNotEmpty(StringView s,const T& default_value)
  {
    return (s.empty()) ? default_value : tryParse(s);
  }

  /*!
   * \brief Convertit la valeur de la variable d'environnement \a s en le type \a T.
   *
   * Si platform::getEnvironmentVariable(s) est nul, return std::nullopt.
   * Sinon, retourne cette valeur convertie en le type \a T. Si la conversion
   * n'est pas possible, retourne std::nullopt si \a throw_if_invalid vaut \a false ou
   * lève une exception s'il vaut \a true.
   */
  ARCANE_UTILS_EXPORT static std::optional<T>
  tryParseFromEnvironment(StringView s,bool throw_if_invalid);
};

//! Spécialisation pour les types scalaires
template<> class Type<Int64> : public ScalarType<Int64>{};
template<> class Type<Int32> : public ScalarType<Int32>{};
template<> class Type<Real> : public ScalarType<Real>{};

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

