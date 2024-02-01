// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TextWriter2.h                                               (C) 2000-2024 */
/*                                                                           */
/* Ecrivain de données.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_INTERNAL_TEXTWRITER2_H
#define ARCANE_STD_INTERNAL_TEXTWRITER2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IDataCompressor;
}

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe d'écriture d'un fichier texte pour les protections/reprises
 */
class TextWriter2
{
  class Impl;

 public:

  explicit TextWriter2(const String& filename);
  TextWriter2(const TextWriter2& rhs) = delete;
  TextWriter2();
  ~TextWriter2();
  TextWriter2& operator=(const TextWriter2& rhs) = delete;

 public:

  void open(const String& filename);
  void write(Span<const std::byte> values);

 public:

  String fileName() const;
  void setDataCompressor(Ref<IDataCompressor> ds);
  Ref<IDataCompressor> dataCompressor() const;
  Int64 fileOffset();
  std::ostream& stream();

 private:

  Impl* m_p;

 private:

  void _binaryWrite(const void* bytes, Int64 len);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
