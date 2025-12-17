// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayDebugInfo.h                                            (C) 2000-2025 */
/*                                                                           */
/* Informations de debug pour les classes tableaux.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ARRAYDEBUGINFO_H
#define ARCCORE_COMMON_ARRAYDEBUGINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

#include "arccore/base/String.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations de debug pour les classes tableaux.
 *
 * Cette classe utilise un compteur de référence. Toutes les instances
 * doivent donc être allouée dynamiquement.
 */
class ARCCORE_COMMON_EXPORT ArrayDebugInfo
{
 public:

  ArrayDebugInfo() = default;

 private:

  ~ArrayDebugInfo() = default;

 public:

  void addReference() { ++m_nb_ref; }
  void removeReference()
  {
    Int32 n = --m_nb_ref;
    if (n == 0)
      delete this;
  }
  void setName(const String& name) { m_name = name; }
  const String& name() const { return m_name; }

 private:

  std::atomic<Int32> m_nb_ref = 0;
  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
