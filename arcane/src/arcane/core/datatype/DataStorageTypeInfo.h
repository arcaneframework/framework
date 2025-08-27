// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataStorageTypeInfo.h                                       (C) 2000-2020 */
/*                                                                           */
/* Informations sur le type du conteneur d'une donnée.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATATYPES_DATASTORAGETYPEINFO_H
#define ARCANE_DATATYPES_DATASTORAGETYPEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/BasicDataType.h"
#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations de type pour un conteneur de données.
 */
class ARCANE_DATATYPE_EXPORT DataStorageTypeInfo
{
 public:
  //! Constructeur.
  DataStorageTypeInfo(eBasicDataType basic_data_type, Int32 nb_basic_element,
                      Int32 dimension, Int32 multi_tag,const String& impl_name = String());
  DataStorageTypeInfo() = default;

 public:

  eBasicDataType basicDataType() const { return m_basic_data_type; }
  Int32 nbBasicElement() const { return m_nb_basic_element; }
  Int32 dimension() const { return m_dimension; }
  Int32 multiTag() const { return m_multi_tag; }
  String fullName() const { return _buildFullName(); }

 private:

  eBasicDataType m_basic_data_type = eBasicDataType::Unknown;
  Int32 m_nb_basic_element = 0;
  Int32 m_dimension = 0;
  Int32 m_multi_tag = 0;
  String m_impl_name;

 private:

  String _buildFullName() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
