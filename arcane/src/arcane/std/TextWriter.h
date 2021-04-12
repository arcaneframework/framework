// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TextWriter.h                                                (C) 2000-2021 */
/*                                                                           */
/* Ecrivain de données.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_TEXTWRITER_H
#define ARCANE_STD_TEXTWRITER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/String.h"

#include <fstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IDeflateService;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe d'écriture d'un fichier texte pour les protections/reprises
 */
class TextWriter
{
  class Impl;
 public:

  TextWriter(const String& filename,bool is_binary);
  TextWriter(const TextWriter& rhs) = delete;
  TextWriter();
  ~TextWriter();
  TextWriter& operator=(const TextWriter& rhs) = delete;

 public:

  void open(const String& filename,bool is_binary);
  void write(const String& comment,Span<const Real> values);
  void write(const String& comment,Span<const Int16> values);
  void write(const String& comment,Span<const Int32> values);
  void write(const String& comment,Span<const Int64> values);
  void write(const String& comment,Span<const Byte> values);
 public:
  const String& fileName() const;
  bool isBinary() const;
  void setDeflater(IDeflateService* ds);
  Int64 fileOffset();
 private:
  Impl* m_p;
 private:
  void _writeComments(const String& comment);
  void _binaryWrite(const void* bytes,Int64 len);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
