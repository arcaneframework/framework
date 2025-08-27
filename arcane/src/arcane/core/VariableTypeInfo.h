// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableTypeInfo.h                                          (C) 2000-2025 */
/*                                                                           */
/* Informations caractérisants le type d'une variable.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLETYPEINFO_H
#define ARCANE_CORE_VARIABLETYPEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations caractérisant le type d'une variable.
 *
 * Les instances de cette classes peuvent être utilisées dans les
 * constructeurs statiques. Pour éviter tout problème cette classe ne doit
 * pas utiliser d'allocation dynamique.
 */
class ARCANE_CORE_EXPORT VariableTypeInfo
{
 public:

  constexpr VariableTypeInfo(eItemKind item_kind,eDataType data_type,Integer dimension,
                             Integer multi_tag,bool is_partial)
 : m_item_kind(item_kind), m_data_type(data_type), m_dimension(dimension),
   m_multi_tag(multi_tag), m_is_partial(is_partial){}

 public:

  //! Type d'entité de maillage
  constexpr eItemKind itemKind() const { return m_item_kind; }
  //! Dimension
  constexpr Integer dimension() const { return m_dimension; }
  //! Multi-tag
  constexpr Integer multiTag() const { return m_multi_tag; }
  //! Type des données de la variable
  constexpr eDataType dataType() const { return m_data_type; }
  //! Indique si la variable est partielle
  constexpr bool isPartial() const { return m_is_partial; }

  //! Nom complet du type de la variable
  String fullName() const;

 public:

  //! Conteneur de donnée par défaut associé à ce type de variable
  DataStorageTypeInfo _internalDefaultDataStorage() const;

 private:

  //! Genre des entités de maillage (peut être nul)
  eItemKind m_item_kind;

  //! Type des données de la variable
  eDataType m_data_type;

  //! Dimension de la variable
  Integer m_dimension;

  //! Tag indiquant si on utilise des tableaux de taille variables.
  Integer m_multi_tag;

  //! Indique si la variable est partielle.
  bool m_is_partial;

 private:

  String _buildFullTypeName() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

