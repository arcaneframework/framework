// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionException.h                                       (C) 2000-2025 */
/*                                                                           */
/* Exception en rapport avec le jeu de données.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CASEOPTIONEXCEPTION_H
#define ARCANE_CORE_CASEOPTIONEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"
#include "arcane/utils/String.h"

#include "arcane/core/XmlNode.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Exception en rapport avec le jeu de données.
 * \ingroup CaseOption
 */
class ARCANE_CORE_EXPORT CaseOptionException
: public Exception
{
 public:

  /*!
   * \brief Exception lorsqu'une option d'une jeu de données n'est pas valide.
   *
   * Cette exception est collective.
   */
  CaseOptionException(const String& where, const String& node_name,
                      const XmlNode& parent, const String& value,
                      const String& type);
  /*!
   * \brief Exception lorsqu'une option d'une jeu de données n'est pas trouvé.
   *
   * Cette exception est collective.
   */
  CaseOptionException(const String& where, const String& node_name,
                      const XmlNode& parent);
  //! Exception générique
  CaseOptionException(const String& where, const String& message, bool is_collective = false);
  //! Exception générique
  CaseOptionException(const TraceInfo& where, const String& message, bool is_collective = false);
  // Constructeur de recopie
  CaseOptionException(const CaseOptionException& rhs) ARCANE_NOEXCEPT;
  ~CaseOptionException() ARCANE_NOEXCEPT override;

 public:

  void explain(std::ostream& m) const override;

 private:

  String m_node_name;
  XmlNode m_parent;
  String m_value;
  String m_type;
  String m_message;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

