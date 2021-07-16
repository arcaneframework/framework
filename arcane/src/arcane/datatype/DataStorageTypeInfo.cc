// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataStorageTypeInfo.cc                                      (C) 2000-2020 */
/*                                                                           */
/* Informations sur le type du conteneur d'une donnée.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/datatype/DataStorageTypeInfo.h"

#include "arcane/utils/StringBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataStorageTypeInfo::
DataStorageTypeInfo(eBasicDataType basic_data_type,Integer nb_basic_element,
                    Integer dimension,Integer multi_tag,const String& impl_name)
: m_basic_data_type(basic_data_type)
, m_nb_basic_element(nb_basic_element)
, m_dimension(dimension)
, m_multi_tag(multi_tag)
, m_impl_name(impl_name)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String DataStorageTypeInfo::
_buildFullName() const
{
  StringBuilder full_name_b;
  full_name_b += Arccore::basicDataTypeName(m_basic_data_type);
  full_name_b += ".";
  full_name_b += m_nb_basic_element;
  full_name_b += ".";
  full_name_b += m_dimension;
  full_name_b += ".";
  full_name_b += m_multi_tag;
  if (!m_impl_name.empty()){
    full_name_b += ".";
    full_name_b += m_impl_name;
  }
  return full_name_b.toString();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
