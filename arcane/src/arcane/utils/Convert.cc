// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Convert.cc                                                  (C) 2000-2020 */
/*                                                                           */
/* Fonctions pour convertir un type en un autre.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/Convert.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/OStringStream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#ifdef ARCANE_REAL_NOT_BUILTIN
template<> ARCANE_UTILS_EXPORT bool
builtInGetValue(Real& v,const String& s)
{
#if 0
  double vz = 0.0;
  if (builtInGetValue(vz,s))
    return true;
  v = vz;
  cout << "** CONVERT DOUBLE TO REAL s=" << s << " vz=" << vz << " v=" << v << '\n';
  return false;
#endif
  double vz = 0.0;
  if (builtInGetValue(vz,s))
    return true;
  v = Real((char*)s.localstr(),1000);
  cout << "** CONVERT DOUBLE TO REAL s=" << s << '\n';
  return false;
}
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<> bool
builtInGetValue(String& v,const String& s)
{
  v = s;
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> bool
builtInGetArrayValue(Array<T>& v,const String& s)
{
  T read_val = T();
  std::istringstream sbuf(s.localstr());
  if (!sbuf.eof())
    sbuf >> ws;
  while(!sbuf.eof()){
    sbuf >> read_val;
    if (sbuf.fail() || sbuf.bad())
      return true;
    v.add(read_val);
    sbuf >> ws;
  }
  return false;
}

namespace{
bool _builtInGetBoolArrayValue(BoolArray& v,const String& s)
{
  // Le type 'bool' est un peu spécial car il doit pouvoir lire les
  // valeurs comme 'true' ou 'false'.
  // On le lit donc comme un 'StringUniqueArray', puis on converti en bool
  //cout << "** GET BOOL ARRAY V=" << s << '\n';
  //return builtInGetArrayValue(v,s);

  StringUniqueArray sa;
  if (builtInGetValue(sa,s))
    return true;
  for( Integer i=0, is=sa.size(); i<is; ++i ){
    bool read_val = false;
    if (builtInGetValue(read_val,sa[i]))
      return true;
    v.add(read_val);
  }
  return false;
}

bool
_builtInGetStringArrayValue(StringArray& v,const String& s)
{
  std::string s2;
  String read_val = String();
  std::istringstream sbuf(s.localstr());
  while(!sbuf.eof()) {
    sbuf >> s2;
    //cout << " ** CHECK READ v='" << s2 << "' '" << sv << "'\n";
    if (sbuf.bad()) // non-recoverable error
      return true;
    if (sbuf.fail()) // recoverable error : this means good conversion
      return false; 
    read_val = StringView(s2.c_str());
    v.add(read_val);
  }
  return false;
}
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(RealArray& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2Array& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3Array& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2x2Array& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3x3Array& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int32Array& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int64Array& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(BoolArray& v,const String& s)
{
  return _builtInGetBoolArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(StringArray& v,const String& s)
{
  return _builtInGetStringArrayValue(v,s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(RealUniqueArray& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2UniqueArray& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3UniqueArray& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2x2UniqueArray& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3x3UniqueArray& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int16UniqueArray& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int32UniqueArray& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int64UniqueArray& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(BoolUniqueArray& v,const String& s)
{
  return _builtInGetBoolArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(StringUniqueArray& v,const String& s)
{
  return _builtInGetStringArrayValue(v,s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(RealSharedArray& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2SharedArray& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3SharedArray& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real2x2SharedArray& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Real3x3SharedArray& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int16SharedArray& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int32SharedArray& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(Int64SharedArray& v,const String& s)
{
  return builtInGetArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(BoolSharedArray& v,const String& s)
{
  return _builtInGetBoolArrayValue(v,s);
}

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(StringSharedArray& v,const String& s)
{
  return _builtInGetStringArrayValue(v,s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
template<class T> inline bool
_builtInPutValue(const T& v,String& s)
{
  OStringStream ostr;
  ostr() << v;
  if (ostr().fail() || ostr().bad())
    return true;
  s = ostr.str();
  return false;
}
template<class T> inline bool
_builtInPutArrayValue(ConstArrayView<T> v,String& s)
{
  OStringStream ostr;
  for( Integer i=0, n=v.size(); i<n; ++i ){
    if (i!=0)
      ostr() << ' ';
    ostr() << v[i];
  }
  if (ostr().fail() || ostr().bad())
    return true;
  s = ostr.str();
  return false;
}
}

bool builtInPutValue(const String& v,String& s)
{ return _builtInPutValue(v,s); }
bool builtInPutValue(double v,String& s)
{ return _builtInPutValue(v,s); }
bool builtInPutValue(float v,String& s)
{ return _builtInPutValue(v,s); }
bool builtInPutValue(int v,String& s)
{ return _builtInPutValue(v,s); }
bool builtInPutValue(unsigned int v,String& s)
{ return _builtInPutValue(v,s); }
bool builtInPutValue(long v,String& s)
{ return _builtInPutValue(v,s); }
bool builtInPutValue(long long v,String& s)
{ return _builtInPutValue(v,s); }
bool builtInPutValue(short v,String& s)
{ return _builtInPutValue(v,s); }
bool builtInPutValue(unsigned short v,String& s)
{ return _builtInPutValue(v,s); }
bool builtInPutValue(unsigned long v,String& s)
{ return _builtInPutValue(v,s); }
bool builtInPutValue(unsigned long long v,String& s)
{ return _builtInPutValue(v,s); }
#ifdef ARCANE_REAL_NOT_BUILTIN
bool builtInPutValue(Real v,String& s)
{ return _builtInPutValue(v,s); }
#endif
bool builtInPutValue(Real2 v,String& s)
{ return _builtInPutValue(v,s); }
bool builtInPutValue(Real3 v,String& s)
{ return _builtInPutValue(v,s); }
bool builtInPutValue(const Real2x2& v,String& s)
{ return _builtInPutValue(v,s); }
bool builtInPutValue(const Real3x3& v,String& s)
{ return _builtInPutValue(v,s); }

bool builtInPutValue(RealConstArrayView v,String& s)
{ return _builtInPutArrayValue(v,s); }
bool builtInPutValue(Real2ConstArrayView v,String& s)
{ return _builtInPutArrayValue(v,s); }
bool builtInPutValue(Real3ConstArrayView v,String& s)
{ return _builtInPutArrayValue(v,s); }
bool builtInPutValue(Real2x2ConstArrayView v,String& s)
{ return _builtInPutArrayValue(v,s); }
bool builtInPutValue(Real3x3ConstArrayView v,String& s)
{ return _builtInPutArrayValue(v,s); }
bool builtInPutValue(Int16ConstArrayView v,String& s)
{ return _builtInPutArrayValue(v,s); }
bool builtInPutValue(Int32ConstArrayView v,String& s)
{ return _builtInPutArrayValue(v,s); }
bool builtInPutValue(Int64ConstArrayView v,String& s)
{ return _builtInPutArrayValue(v,s); }
bool builtInPutValue(BoolConstArrayView v,String& s)
{ return _builtInPutArrayValue(v,s); }
bool builtInPutValue(StringConstArrayView v,String& s)
{ return _builtInPutArrayValue(v,s); }
//@}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static char global_hexa[16] = {'0','1', '2', '3', '4', '5', '6', '7', '8', '9',
                               'a', 'b', 'c', 'd', 'e', 'f' };

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String Convert::
toHexaString(ByteConstArrayView input)
{
  UniqueArray<char> out_buf;
  Integer len = input.size();
  out_buf.resize((len*2)+1);
  for( Integer i=0; i<len; ++i ){
    int v = (int)input[i];
    out_buf[(i*2)] = global_hexa[v/16];
    out_buf[(i*2)+1] = global_hexa[v%16];
  }
  out_buf[len*2] = '\0';
  return StringView(out_buf.data());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Convert::
toHexaString(Int64 input,Span<Byte> output)
{
  for (Integer i=0; i<8; ++i ){
    Byte v = (Byte)(input % 256);
    output[(i*2)] = global_hexa[v/16];
    output[(i*2)+1] = global_hexa[v%16];
    input = input / 256;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String Convert::
toHexaString(Real input)
{
  return toHexaString(ByteConstArrayView(sizeof(Real),(Byte*)&input));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

