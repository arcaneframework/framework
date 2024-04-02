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

  typedef Int32 KeyTypeValue;
  typedef Int32 ValueType;
  typedef HashTraitsT<KeyTypeValue> KeyTraitsType;
  typedef KeyTraitsType::KeyTypeConstRef KeyTypeConstRef;

 public:

  explicit GroupIndexTable(ItemGroupImpl* group_impl);

 public:

  void update();

  void clear();

  void compact(const Int32ConstArrayView* info);

  ARCCORE_HOST_DEVICE ValueType operator[](Int32 i) const { return _lookup(i); }

  KeyTypeValue keyLocalId(Int32 i) const { return m_key_buffer[i]; }

  ARCCORE_HOST_DEVICE Int32 size() const { return m_key_buffer.size(); }

 private:

  /*!
   * \brief Fonction de hachage.
   *
   * Utilise la fonction de hachage de Arcane même si quelques
   * collisions sont constatées avec les petites valeurs
   */
  ARCCORE_HOST_DEVICE Int32 _hash(KeyTypeConstRef id) const
  {
#ifndef ARCCORE_DEVICE_CODE
    ARCANE_ASSERT((_initialized()), ("GroupIndexTable not initialized"));
#endif
    return static_cast<Int32>(KeyTraitsType::hashFunction(id) % m_nb_bucket);
  }
  //! \a true si une valeur avec la clé \a id est présente
  bool _hasKey(KeyTypeConstRef id) const;

  //! Recherche d'une clef dans un bucket
  ARCCORE_HOST_DEVICE Integer _lookupBucket(Int32 bucket, KeyTypeConstRef id) const
  {
#ifndef ARCCORE_DEVICE_CODE
    ARCANE_ASSERT((_initialized()), ("GroupIndexTable not initialized"));
#endif
    for (Integer i = m_buckets_span[bucket]; i >= 0; i = m_next_buffer_span[i]) {
      if (m_key_buffer_span[i] == id)
        return i;
    }
    return -1;
  }

  //! Recherche d'une clef dans toute la table
  ARCCORE_HOST_DEVICE Int32 _lookup(KeyTypeConstRef id) const
  {
#ifndef ARCCORE_DEVICE_CODE
    ARCANE_ASSERT((_checkIntegrity(false)), ("GroupIndexTable integrity failed"));
#endif
    return _lookupBucket(_hash(id), id);
  }

  //! Teste l'initialisation de l'objet
  bool _initialized() const;

  //! Test l'intégrité de la table relativement à son groupe
  bool _checkIntegrity(bool full = true) const;

 private:

  ItemGroupImpl* m_group_impl = nullptr;
  UniqueArray<KeyTypeValue> m_key_buffer; //! Table des clés associées
  UniqueArray<Int32> m_next_buffer; //! Table des index suivant associés
  UniqueArray<Int32> m_buckets; //! Tableau des buckets
  bool m_disable_check_integrity = false;
  SmallSpan<const KeyTypeValue> m_key_buffer_span;
  SmallSpan<const Int32> m_next_buffer_span;
  SmallSpan<const Int32> m_buckets_span;

 private:

  void _updateSpan();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
