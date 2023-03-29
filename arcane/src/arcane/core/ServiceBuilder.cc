// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceBuilder.cc                                           (C) 2000-2023 */
/*                                                                           */
/* Classe utilitaire pour instantier un service.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ServiceBuilder.h"

#include "arcane/core/ICaseMng.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/CaseOptions.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ReferenceCounter<ICaseOptions> ServiceBuilderWithOptionsBase::
_buildCaseOptions(const String& xml_content) const
{
  String prolog = "<?xml version=\"1.0\"?>\n<root>\n<test1>\n";
  String epilog = "</test1>\n</root>\n";
  String service_xml_value = prolog + xml_content + epilog;

  // TODO: à détruire
  ITraceMng* tm = m_case_mng->traceMng();
  IXmlDocumentHolder* xml_doc = IXmlDocumentHolder::loadFromBuffer(service_xml_value.bytes(), String(), tm);
  XmlNode xml_root_node = xml_doc->documentNode().documentElement();

  ICaseOptions* co = new CaseOptions(m_case_mng, "test1", xml_root_node);
  return ReferenceCounter<ICaseOptions>(co);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IApplication* ServiceBuilderWithOptionsBase::
_application() const
{
  return m_case_mng->application();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceBuilderWithOptionsBase::
_readOptions(ICaseOptions* opt) const
{
  m_case_mng->_internalReadOneOption(opt, true);
  m_case_mng->_internalReadOneOption(opt, false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

