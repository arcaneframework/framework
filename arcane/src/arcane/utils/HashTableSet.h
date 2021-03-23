// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HashTableSet.h                                              (C) 2000-2005 */
/*                                                                           */
/* Ensemble utilisant une table de hachage.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_HASHTABLESET_H
#define ARCANE_UTILS_HASHTABLESET_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/HashTableMap.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//TODO: à intégrer avec HashTableMap
#if 0
/*!
 * \brief Ensemble d'éléments gérés par une table de hachage.
 
 Cette classe permet de gérer un ensemble quelconque d'éléments. Contrairement
 à la class HashTableMapT, elle n'associe aucune valeur à l'élément. Il est
 uniquement possible d'ajouter des éléments et de regarder s'ils sont
 présents.
 
*/
template<typename KeyType,typename KeyTraitsType = HashTraitsT<KeyType> >
class HashTableSetT
: public HashTableBaseT<KeyType,KeyTraitsType>
{
 public:

  typedef typename KeyTraitsType::KeyTypeConstRef KeyTypeConstRef;
  typedef typename KeyTraitsType::KeyTypeRef KeyTypeRef;
  typedef typename KeyTraitsType::KeyTypeValue KeyTypeValue;
  typedef HashTableBaseT<KeyType,KeyTraitsType> BaseClass;
  typedef typename BaseClass::HashData HashData;

 public:

  /*! \brief Crée une table de taille \a table_size
   *
   Si \a use_prime est vrai, utilise la fonction nearestPrimeNumber()
   pour avoir une taille de taille qui est un nombre premier.
  */  
  HashTableSetT(Integer table_size,bool use_prime)
  : BaseClass(table_size,use_prime), m_buffer(BaseClass::m_nb_bucket)
    {
    }

  ~HashTableSetT() {}


  /*! \brief Ajoute la valeur \a value correspondant à la clé \a id
    
    Si une valeur correspondant à \a id existe déjà, elle est remplacée
  */
  void add(KeyTypeConstRef id)
    {
      Integer hf = _hash(id);
      HashData* ht = _lookupBucket(hf,id);
      if (!ht)
        _add(hf,id);
    }

  /*! \brief Ajoute la valeur \a value correspondant à la clé \a id
    
    Si une valeur correspondant à \a id existe déjà, le résultat est
    indéfini.
  */
  void nocheckAdd(KeyTypeConstRef id)
    {
      Integer hf = _hash(id);
      _add(hf,id);
    }

 private:

  MultiBufferT<HashData> m_buffer; //!< Tampon d'allocation des valeurs
  KeyType m_null_value; //! Valeur nulle (non utilisée)

  KeyTypeRef _add(Integer hf,KeyTypeConstRef id)
    {
      HashData* hd = m_buffer.allocOne();
      BaseClass::_baseAdd(hf,id,hd);
      return hd->m_key;
    }
  inline HashData* _lookup(KeyTypeConstRef id)
    { return BaseClass::_baseLookup(id); }
  inline HashData* _lookupBucket(Integer bucket,KeyTypeConstRef id)
    { return BaseClass::_baseLookupBucket(bucket,id); }
  inline Integer _hash(KeyTypeConstRef id)
    { return BaseClass::_hash(id); }
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
