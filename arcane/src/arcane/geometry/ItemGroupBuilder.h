// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ITEMGROUPBUILDER_H
#define ITEMGROUPBUILDER_H

#include <set>
#include <cstring>
#include <cctype>

#include <arcane/ArcaneVersion.h>
#include <arcane/ItemGroup.h>
#include <arcane/IMesh.h>
#include <arcane/utils/String.h>
#include <arcane/utils/StringBuilder.h>
#include <arcane/IItemFamily.h>
#include <arcane/ItemGroupRangeIterator.h>

using namespace Arcane;

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

//! Macro de construction d'un nom d'objet
/*! Sert généralement à nommer des groupes pour ItemGroupBuilder */
#define IMPLICIT_NAME ItemGroupBuilder_cleanString(__FILE__ "__" TOSTRING(__LINE__),false)
#define IMPLICIT_UNIQ_NAME ItemGroupBuilder_cleanString(__FILE__ "__" TOSTRING(__LINE__),true)

/*
 * \internal
 * \brief Outil de construction assisté de groupe
 *
 * L'unicité des éléments du groupe est garantie par construction. Il
 * est possible d'utiliser la macro IMPLICIT_NAME pour nommer nom de
 * groupe.
 */
template<typename T>
class ItemGroupBuilder
{
 private:
  IMesh* m_mesh;
  std::set<Integer> m_ids;
  String m_group_name;
  
 public:
  //! Constructeur
  ItemGroupBuilder(IMesh* mesh,const String& groupName)
  : m_mesh(mesh), m_group_name(groupName) {}

  //! Destructeur
  virtual ~ItemGroupBuilder() {}

 public:
  //! Ajout d'un ensemble d'item fourni par un énumérateur
  void add(ItemEnumeratorT<T> enumerator) 
    { 
      while(enumerator.hasNext()) 
        {
          m_ids.insert(enumerator.localId());
          ++enumerator;
        }
    }

  //! Ajout d'un ensemble d'item fourni par un énumérateur
  void add(ItemGroupRangeIteratorT<T> enumerator) 
    { 
      while(enumerator.hasNext())
        {
          m_ids.insert(enumerator.itemLocalId());
          ++enumerator;
        }
    }

  //! Ajout d'un item unique
  void add(const T & item) 
    { 
      m_ids.insert(item.localId());
    }

  //! Constructeur du nouveau group
  ItemGroupT<T> buildGroup() 
    {
      Int32UniqueArray localIds(m_ids.size());

      std::set<Integer>::const_iterator is = m_ids.begin();
      Integer i = 0;

      while(is != m_ids.end())
        {
          localIds[i] = *is;
          ++is; ++i;
        }
    
//       ItemGroup newGroup(new ItemGroupImpl(m_mesh->itemFamily(ItemTraitsT<T>::kind()),
//                                            m_group_name));
      ItemGroup newGroup = m_mesh->itemFamily(ItemTraitsT<T>::kind())->findGroup(m_group_name,true);
// m_item_family->createGroup(own_name,ItemGroup(this));

      newGroup.clear();
      newGroup.setItems(localIds);
      // newGroup.setLocalToSubDomain(true); // Force le nouveau a être local : non transférer en cas de rééquilibrage

      return newGroup;
    }

  //! Nom du groupe
  String getName() const 
    { 
      return m_group_name; 
    }
};

#endif /* ITEMGROUPBUILDER_H */
