// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionError.h                                           (C) 2000-2025 */
/*                                                                           */
/* Erreur dans le jeu de données.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CASEOPTIONERROR_H
#define ARCANE_CORE_CASEOPTIONERROR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/core/XmlNode.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ICaseDocument;
class ICaseDocumentFragment;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup CaseOption
 * \brief Erreur dans le jeu de données.
 */
class ARCANE_CORE_EXPORT CaseOptionError
{
 public:
	
  //! Erreur générique
  CaseOptionError(const TraceInfo& where,const String& node_name,
                  const String& message,bool is_collective=false);
  
 public:
  
  /*!
   * \brief Erreur lorsqu'une valeur d'une jeu de données n'est pas du bon type.
   * Cette erreur est collective.
   */
  static void addInvalidTypeError(ICaseDocumentFragment* document,
                                  const TraceInfo& where,
                                  const String& node_name,
                                  const XmlNode& parent,
                                  const String& value,
                                  const String& expected_type);

  /*!
   * \brief Erreur lorsqu'une valeur d'une jeu de données n'est pas du bon type.
   * Cette erreur est collective.
   */
  static void addInvalidTypeError(ICaseDocumentFragment* document,
                                  const TraceInfo& where,
                                  const String& node_name,
                                  const XmlNode& parent,
                                  const String& value,
                                  const String& expected_type,
                                  StringConstArrayView valid_values);

  /*!
   * \brief Erreur lorsqu'une option du jeu de données n'est pas trouvée.
   * Cette erreur est collective.
   */
  static void addOptionNotFoundError(ICaseDocumentFragment* document,
                                const TraceInfo& where,
                                const String& node_name,
                                const XmlNode& parent);

  //! Erreur générique
  static void addError(ICaseDocumentFragment* document,
                       const TraceInfo& where,const String& node_name,
                       const String& message,bool is_collective=false);

  //! Erreur générique
  static void addWarning(ICaseDocumentFragment* document,
                         const TraceInfo& where,const String& node_name,
                         const String& message,bool is_collective=false);

 public:
	
  const String& nodeName() const { return m_node_name; }

  const String& message() const { return m_message; }

  bool isCollective() const { return m_is_collective; }

  const TraceInfo& trace() const { return m_func_info; }

 private:

  TraceInfo m_func_info;
  String m_node_name;
  String m_message;
  bool m_is_collective;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

