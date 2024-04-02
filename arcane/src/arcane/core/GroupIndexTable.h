// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GroupIndexTable.h                                           (C) 2000-2024 */
/*                                                                           */
/* Table de hachage entre un item et sa position dans la table.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_GROUPINDEXTABLE_H
#define ARCANE_CORE_GROUPINDEXTABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/HashTable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class ItemGroupImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base d'une table de hachage entre les items d'un groupe 
 * et leurs positions dans la table
 *
 * Cette table est utilisée pour les variables partielles : la position des
 * données d'une entité n'est pas son localId()  mais sa position dans
 * l'énumerateur du groupe (i.e: dans la table).         
 */
class ARCANE_CORE_EXPORT GroupIndexTable
: public HashTableBase
{
 public:

  typedef Integer KeyTypeValue;
  typedef Integer ValueType;
  typedef HashTraitsT<KeyTypeValue> KeyTraitsType;
  typedef KeyTraitsType::KeyTypeConstRef KeyTypeConstRef;

 public:

  GroupIndexTable(ItemGroupImpl* group_impl);

  ~GroupIndexTable() {}

  void update();

  void clear();

  void compact(const Int32ConstArrayView* info);

  ValueType operator[](Integer i) const { return _lookup(i); }

  KeyTypeValue keyLocalId(Integer i) const { return m_key_buffer[i]; }

  Integer size() const { return m_key_buffer.size(); }

 private:

  /*!
   * \brief Fonction de hachage.
   *
   * Utilise la fonction de hachage de Arcane même si quelques
   * collisions sont constatées avec les petites valeurs
   */
  Integer _hash(KeyTypeConstRef id) const;

  //! \a true si une valeur avec la clé \a id est présente
  bool _hasKey(KeyTypeConstRef id) const;

  //! Recherche d'une clef dans un bucket
  Integer _lookupBucket(Integer bucket, KeyTypeConstRef id) const;

  //! Recherche d'une clef dans toute la table
  Integer _lookup(KeyTypeConstRef id) const;

  //! Teste l'initialisation de l'objet
  bool _initialized() const;

  //! Test l'intégrité de la table relativement à son groupe
  bool _checkIntegrity(bool full = true) const;

 private:

  ItemGroupImpl* m_group_impl = nullptr;
  UniqueArray<KeyTypeValue> m_key_buffer; //! Table des clés associées
  UniqueArray<Integer> m_next_buffer; //! Table des index suivant associés
  UniqueArray<Integer> m_buckets; //! Tableau des buckets
  bool m_disable_check_integrity = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
