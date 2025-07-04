// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IosFile.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Routines des Lecture/Ecriture d'un fichier.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_INTERNAL_SERVICES_H
#define ARCANE_STD_INTERNAL_SERVICES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iostream.h"
#include "arcane/utils/ArrayView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Routines des Lecture/Ecriture d'un fichier.
 */
class IosFile
{
 public:

  static const int IOS_BFR_SZE = 8192;

 public:

  IosFile(std::istream* stream)
  : m_stream(stream)
  {}
  const char* getNextLine(const char*);
  const char* getNextLine(void);
  void goToEndOfLine(void);
  Real getReal(void);
  Integer getInteger(void);
  Int64 getInt64(void);
  bool lookForString(const String& str);
  void checkString(const String& current_value, const String& expected_value);
  void checkString(const String& current_value, const String& expected_value1, const String& expected_value2);
  static bool isEqualString(const String& current_value, const String& expected_value);
  bool isEnd(void);
  void readBytes(SmallSpan<std::byte> bytes);
  void binaryRead(SmallSpan<Int32> values);
  void binaryRead(SmallSpan<Int64> values);
  void binaryRead(SmallSpan<Real3> values);
  void binaryRead(SmallSpan<double> values);
  void binaryRead(SmallSpan<Byte> values);

 private:

  std::istream* m_stream = nullptr;
  char m_buf[IOS_BFR_SZE];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
