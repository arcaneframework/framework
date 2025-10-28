// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Convert.cc                                                  (C) 2000-2025 */
/*                                                                           */
/* Fonctions pour convertir un type en un autre.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Convert.h"

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctions pour convertir un type en un autre.
 * \namespace Arcane::Convert
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static char global_hexa[16] = {'0','1', '2', '3', '4', '5', '6', '7', '8', '9',
                               'a', 'b', 'c', 'd', 'e', 'f' };

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
String
_toHexaString(Span<const std::byte> input)
{
  UniqueArray<Byte> out_buf;
  Int64 len = input.size();
  out_buf.resize((len*2)+1);
  for( Int64 i=0; i<len; ++i ){
    int v = std::to_integer<int>(input[i]);
    out_buf[(i*2)] = global_hexa[v/16];
    out_buf[(i*2)+1] = global_hexa[v%16];
  }
  out_buf[len*2] = '\0';
  return String(out_buf);
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String Convert::
toHexaString(Span<const std::byte> input)
{
  return _toHexaString(input);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String Convert::
toHexaString(ByteConstArrayView input)
{
  return _toHexaString(asBytes(Span<const Byte>(input)));
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

namespace Convert
{

template<typename T> std::optional<T>
ScalarType<T>::tryParse(StringView s)
{
  T v;
  if (s.empty())
    return std::nullopt;
  bool is_bad = builtInGetValue(v,s);
  if (is_bad)
    return std::nullopt;
  return v;
}

template<typename T> std::optional<T>
ScalarType<T>::tryParseFromEnvironment(StringView s,bool throw_if_invalid)
{
  String env_value = platform::getEnvironmentVariable(s);
  if (env_value.null())
    return std::nullopt;
  auto v = tryParse(env_value);
  if (!v && throw_if_invalid)
    ARCANE_FATAL("Invalid value '{0}' for environment variable {1}. Can not convert to type '{2}'",
                 env_value,s,typeToName(T{}));
  return v;
}

template class ScalarType<Int32>;
template class ScalarType<Int64>;
template class ScalarType<Real>;

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

