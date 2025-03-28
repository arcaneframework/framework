// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ValueConvert.h                                              (C) 2000-2025 */
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
#include "arcane/utils/NumericTypes.h"
#include "arcane/utils/Convert.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace impl
{
/*!
 * \brief Encapsule un std::istream pour un StringView.
 *
 * Actuellement (C++20) std::istringstream utilise en
 * entrée un std::string ce qui nécessite une instance de ce type
 * et donc une allocation potentielle. Cette classe sert à éviter
 * cela en utilisant directement la mémoire pointée par l'instance
 * de StringView passé dans le constructeur. Cette dernière doit
 * rester valide durant toute l'ulisation de cette classe.
 */
class ARCANE_UTILS_EXPORT StringViewInputStream
: private std::streambuf
{
 public:

  StringViewInputStream(StringView v);

 public:

  std::istream& stream() { return m_stream; }

 private:

  StringView m_view;
  std::istream m_stream;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti la valeur de la chaîne \a s dans le type basique \a T
 * et stocke la valeur dans \a v.
 *
 * \retval true en cas d'échec.
 * \retval false en cas de succès
 */
template <class T> inline bool
builtInGetValueGeneric(T& v, StringView s)
{
  T read_val = T();
  impl::StringViewInputStream svis(s);
  std::istream& sbuf = svis.stream();
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

} // namespace impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti la valeur de la chaîne \a s dans le type basique \a T
 * et stocke la valeur dans \a v.
 *
 * \retval true en cas d'échec.
 * \retval false en cas de succès
 */
template <class T> inline bool
builtInGetValue(T& v, StringView s)
{
  return impl::builtInGetValueGeneric(v, s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <> ARCANE_UTILS_EXPORT bool builtInGetValue(String& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(double& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(float& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(int& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(unsigned int& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(long& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(long long& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(short& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(unsigned short& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(unsigned long& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(unsigned long long& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Float16& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(BFloat16& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Float128& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Int128& v, StringView s);
#ifdef ARCANE_REAL_NOT_BUILTIN
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Real& v, StringView s);
#endif
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2x2& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3x3& v, StringView s);

template <> ARCANE_UTILS_EXPORT bool builtInGetValue(RealArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2Array& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3Array& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2x2Array& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3x3Array& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Int8Array& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Int16Array& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Int32Array& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Int64Array& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(BFloat16Array& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Float16Array& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Float32Array& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Array<Float128>& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Array<Int128>& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(BoolArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(StringArray& v, StringView s);

template <> ARCANE_UTILS_EXPORT bool builtInGetValue(RealSharedArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2SharedArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3SharedArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2x2SharedArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3x3SharedArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Int8SharedArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Int16SharedArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Int32SharedArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Int64SharedArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(BFloat16SharedArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Float16SharedArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Float32SharedArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(SharedArray<Float128>& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(SharedArray<Int128>& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(BoolSharedArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(StringSharedArray& v, StringView s);

template <> ARCANE_UTILS_EXPORT bool builtInGetValue(RealUniqueArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2UniqueArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3UniqueArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2x2UniqueArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3x3UniqueArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Int8UniqueArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Int16UniqueArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Int32UniqueArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Int64UniqueArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(BFloat16UniqueArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Float16UniqueArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(Float32UniqueArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(UniqueArray<Float128>& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(UniqueArray<Int128>& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(BoolUniqueArray& v, StringView s);
template <> ARCANE_UTILS_EXPORT bool builtInGetValue(StringUniqueArray& v, StringView s);

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
builtInGetValue(bool& v, StringView s)
{
  if (s.empty())
    return true;
  if (s == "false" || s == "faux" || s == "0") {
    v = false;
    return false;
  }

  if (s == "true" || s == "vrai" || s == "1") {
    v = true;
    return false;
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti la valeur de la chaîne \a s dans le type basique \a T
 * et stocke la valeur dans \a v.
 *
 * \retval true en cas d'échec.
 * \retval false en cas de succès
 */
template <class T> inline bool
builtInGetValue(T& v, const String& s)
{
  return builtInGetValue(v, s.view());
}

//! Spécialisation pour 'String'
template <> inline bool
builtInGetValue(String& v, const String& s)
{
  v = s;
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti la valeur de la chaîne \a s dans le type basique \a T
 * et stocke la valeur dans \a v.
 *
 * \retval true en cas d'échec.
 * \retval false en cas de succès
 */
template <class T> inline bool
builtInGetValue(T& v, const char* s)
{
  return builtInGetValue(v, StringView(s));
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
ARCANE_UTILS_EXPORT bool builtInPutValue(const String& v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(double v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(float v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(int v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(unsigned int v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(long v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(long long v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(short v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(unsigned short v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(unsigned long v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(unsigned long long v, String& s);
#ifdef ARCANE_REAL_NOT_BUILTIN
ARCANE_UTILS_EXPORT bool builtInPutValue(Real v, String& s);
#endif
ARCANE_UTILS_EXPORT bool builtInPutValue(Real2 v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Real3 v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(const Real2x2& v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(const Real3x3& v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const Real> v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const Real2> v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const Real3> v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const Real2x2> v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const Real3x3> v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const Int16> v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const Int32> v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const Int64> v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const bool> v, String& s);
ARCANE_UTILS_EXPORT bool builtInPutValue(Span<const String> v, String& s);
//@}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline const char* typeToName(bool)
{
  return "boolean";
}
inline const char* typeToName(Real)
{
  return "real";
}
inline const char* typeToName(Real2)
{
  return "real2";
}
inline const char* typeToName(Real3)
{
  return "real3";
}
inline const char* typeToName(Real2x2)
{
  return "real2x2";
}
inline const char* typeToName(Real3x3)
{
  return "real3x3";
}
inline const char* typeToName(short)
{
  return "short";
}
inline const char* typeToName(unsigned short)
{
  return "unsigned short";
}
inline const char* typeToName(int)
{
  return "integer";
}
inline const char* typeToName(long)
{
  return "long";
}
inline const char* typeToName(unsigned long)
{
  return "unsigned long";
}
inline const char* typeToName(unsigned long long)
{
  return "unsigned long long";
}
inline const char* typeToName(const String&)
{
  return "string";
}
inline const char* typeToName(long long)
{
  return "long long";
}
inline const char* typeToName(unsigned int)
{
  return "unsigned integer";
}
inline const char* typeToName(BFloat16)
{
  return "bfloat16";
}
inline const char* typeToName(Float16)
{
  return "float16";
}
inline const char* typeToName(Float32)
{
  return "float32";
}
inline const char* typeToName(Float128)
{
  return "float128";
}
inline const char* typeToName(Int128)
{
  return "int128";
}
inline const char* typeToName(const StringArray&)
{
  return "string[]";
}
inline const char* typeToName(const BoolArray&)
{
  return "boolean[]";
}
inline const char* typeToName(const RealArray&)
{
  return "real[]";
}
inline const char* typeToName(const Real2Array&)
{
  return "real2[]";
}
inline const char* typeToName(const Real3Array&)
{
  return "real3[]";
}
inline const char* typeToName(const Real2x2Array&)
{
  return "real2x2[]";
}
inline const char* typeToName(const Real3x3Array&)
{
  return "real3x3[]";
}
inline const char* typeToName(const Int16Array&)
{
  return "Int16[]";
}
inline const char* typeToName(const Int32Array&)
{
  return "Int32[]";
}
inline const char* typeToName(const Int64Array&)
{
  return "Int64[]";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

