// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableMetaData.h                                           C) 2000-2018 */
/*                                                                           */
/* Meta-données sur une variable.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VARIABLEMETADATA_H
#define ARCANE_VARIABLEMETADATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Meta-données sur une variable.
 *
 * Les informations de cette classe permettent de reconstruire une
 * variable.
 */
class ARCANE_CORE_EXPORT VariableMetaData
{
 public:
  /*!
   * \brief Constructeur.
   *
   * Contruit l'instance pour une variable de nom \a base_name,
   * de la famille \a item_family_name et de groupe \a item_group_name.
   * Si la variable n'est pas sur un maillage, alors \a mesh_name,
   * \a item_family_name et \a item_group_name sont nuls.
   */
  VariableMetaData(const String& base_name,const String& mesh_name,
                   const String& item_family_name,const String& item_group_name,
                   bool is_partial);

 public:

  //! Nom complet de la variable
  String fullName() const { return m_full_name; }
  //! Nom de base de la variable
  String baseName() const { return m_base_name; }
  String meshName() const { return m_mesh_name; }
  String itemFamilyName() const { return m_item_family_name; }
  String itemGroupName() const { return m_item_group_name; }
  bool isPartial() const { return m_is_partial; }

  String fullType() const { return m_full_type; }
  void setFullType(const String& v) { m_full_type = v; }

  String hash() const { return m_hash_str; }
  void setHash(const String& v) { m_hash_str = v; }

  Integer property() const { return m_property; }
  void setProperty(Integer v) { m_property = v; }

  String multiTag() const { return m_multi_tag; }
  void setMultiTag(const String& v) { m_multi_tag = v; }


 private:

  String m_base_name;
  String m_mesh_name;
  String m_item_family_name;
  String m_item_group_name;
  String m_full_type;
  String m_hash_str;
  Integer m_property;
  String m_multi_tag;
  String m_full_name;
  bool m_is_partial;

 private:

  void _buildFullName();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

