// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionError.h                                           (C) 2000-2025 */
/*                                                                           */
/* Dataset error.                                                            */
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
 * \brief Error in the dataset.
 */
class ARCANE_CORE_EXPORT CaseOptionError
{
 public:
	
  //! Generic error
  CaseOptionError(const TraceInfo& where,const String& node_name,
                  const String& message,bool is_collective=false);
  
 public:
  
  /*!
   * \brief Error when a dataset value is not of the correct type.
   * This error is collective.
   */
  static void addInvalidTypeError(ICaseDocumentFragment* document,
                                  const TraceInfo& where,
                                  const String& node_name,
                                  const XmlNode& parent,
                                  const String& value,
                                  const String& expected_type);

  /*!
   * \brief Error when a dataset value is not of the correct type.
   * This error is collective.
   */
  static void addInvalidTypeError(ICaseDocumentFragment* document,
                                  const TraceInfo& where,
                                  const String& node_name,
                                  const XmlNode& parent,
                                  const String& value,
                                  const String& expected_type,
                                  StringConstArrayView valid_values);

  /*!
   * \brief Error when a dataset option is not found.
   * This error is collective.
   */
  static void addOptionNotFoundError(ICaseDocumentFragment* document,
                                const TraceInfo& where,
                                const String& node_name,
                                const XmlNode& parent);

  //! Generic error
  static void addError(ICaseDocumentFragment* document,
                       const TraceInfo& where,const String& node_name,
                       const String& message,bool is_collective=false);

  //! Generic error
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
