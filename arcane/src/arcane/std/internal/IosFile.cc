// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IosFile.cc                                                  (C) 2000-2024 */
/*                                                                           */
/* Routines des Lecture/Ecriture d'un fichier.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/IosFile.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/IOException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Real3.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool IosFile::
isEnd()
{
  (*m_stream) >> ws;
  return m_stream->eof();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const char* IosFile::
getNextLine(const char* comment_char)
{
  while (m_stream->good()) {
    m_stream->getline(m_buf, sizeof(m_buf) - 1);
    if (m_stream->eof())
      break;
    if (m_buf[0] == '\n' || m_buf[0] == '\r')
      continue;
    bool is_comment = true; // Comments are searched for by default
    if (!comment_char)
      is_comment = false; // If none has been set, just skip their track of it
    // Regarde si un caractère de commentaire est présent
    for (int i = 0; is_comment && i < IOS_BFR_SZE && m_buf[i] != '\0'; ++i) {
      if (!isspace(m_buf[i])) {
        is_comment = (m_buf[i] == *comment_char);
        break;
      }
    }

    if (!is_comment) {
      // Supprime le '\n' ou '\r' final
      for (int i = 0; i < IOS_BFR_SZE && m_buf[i] != '\0'; ++i) {
        //cout << " V=" << m_buf[i] << " I=" << (int)m_buf[i] << "\n";
        if (m_buf[i] == '\n' || m_buf[i] == '\r') {
          m_buf[i] = '\0';
          break;
        }
      }
      return m_buf;
    }
  }
  throw IOException("IosFile::getNexLine()", "Unexpected EndOfFile");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const char* IosFile::
getNextLine()
{
  return getNextLine(nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real IosFile::
getReal()
{
  Real v = 0.;
  (*m_stream) >> ws >> v;
  if (m_stream->good())
    return v;
  throw IOException("IosFile::getReal()", "Bad Real");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer IosFile::
getInteger()
{
  Integer v = 0;
  (*m_stream) >> ws >> v;
  if (m_stream->good())
    return v;
  throw IOException("IosFile::getInteger()", "Bad Integer");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 IosFile::
getInt64()
{
  Int64 v = 0;
  (*m_stream) >> ws >> v;
  if (m_stream->good())
    return v;
  throw IOException("IosFile::getInteger()", "Bad Int64");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool IosFile::
lookForString(const String& str)
{
  const char* bfr = getNextLine();
  std::cout << "[IosFile::getString] Looking for '" << str << "' len=" << str.length() << "\n";
  std::istringstream iline(bfr);
  std::string got;
  iline >> got;
  std::cout << "[IosFile::getString] got='" << got << "' len=" << got.length() << "\n";
  return isEqualString(got, str);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IosFile::
checkString(const String& current_value, const String& expected_value)
{
  String current_value_low = current_value.lower();
  String expected_value_low = expected_value.lower();

  if (current_value_low != expected_value_low) {
    String s = "Expecting chain '" + expected_value + "', found '" + current_value + "'";
    throw IOException("IosFile::checkString()", s);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IosFile::
checkString(const String& current_value, const String& expected_value1, const String& expected_value2)
{
  String current_value_low = current_value.lower();
  String expected_value1_low = expected_value1.lower();
  String expected_value2_low = expected_value2.lower();

  if (current_value_low != expected_value1_low && current_value_low != expected_value2_low) {
    String s = "Expecting chain '" + expected_value1 + "' or '" + expected_value2 + "', found '" + current_value + "'";
    throw IOException("IosFile::checkString()", s);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool IosFile::
isEqualString(const String& current_value, const String& expected_value)
{
  String current_value_low = current_value.lower();
  String expected_value_low = expected_value.lower();
  return (current_value_low == expected_value_low);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IosFile::
readBytes(SmallSpan<std::byte> bytes)
{
  m_stream->read(reinterpret_cast<char*>(bytes.data()),bytes.size());
  if (!m_stream->good())
    throw IOException("IosFile::readBytes()",
                      String::format("Can not read '{0}' bytes",bytes.size()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IosFile::
binaryRead(SmallSpan<Int32> values)
{
  readBytes(asWritableBytes(values));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IosFile::
binaryRead(SmallSpan<Int64> values)
{
  readBytes(asWritableBytes(values));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IosFile::
binaryRead(SmallSpan<double> values)
{
  readBytes(asWritableBytes(values));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IosFile::
binaryRead(SmallSpan<Real3> values)
{
  readBytes(asWritableBytes(values));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IosFile::
binaryRead(SmallSpan<Byte> values)
{
  readBytes(asWritableBytes(values));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
