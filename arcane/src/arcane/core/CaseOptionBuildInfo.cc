// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionBuildInfo.cc                                      (C) 2000-2018 */
/*                                                                           */
/* Informations pour construire une option du jeu de données                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/CaseOptionBuildInfo.h"
#include "arcane/ICaseMng.h"
#include "arcane/CaseOptions.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionBuildInfo::
CaseOptionBuildInfo(ICaseOptionList* icl,const String& s,
                    const XmlNode& element,const String& def_val,
                    Integer min_occurs,Integer max_occurs)
: m_case_mng(icl->caseMng())
, m_case_option_list(icl)
, m_name(s)
, m_default_value(def_val)
, m_element(element)
, m_min_occurs(min_occurs)
, m_max_occurs(max_occurs)
, m_is_optional(false)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionBuildInfo::
CaseOptionBuildInfo(ICaseOptionList* icl,const String& s,
                    const XmlNode& element,const String& def_val,
                    Integer min_occurs,Integer max_occurs,bool is_optional)
: m_case_mng(icl->caseMng())
, m_case_option_list(icl)
, m_name(s)
, m_default_value(def_val)
, m_element(element)
, m_min_occurs(min_occurs)
, m_max_occurs(max_occurs)
, m_is_optional(is_optional)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

