// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef ITEMGROUPBUILDER_H
#define ITEMGROUPBUILDER_H

#include <set>
#include <cstring>
#include <cctype>

#include <arcane/ArcaneVersion.h>
#include <arcane/ItemGroup.h>
#include <arcane/IMesh.h>
#include <arcane/utils/String.h>
#if (ARCANE_VERSION>=10600)
#include <arcane/utils/StringBuilder.h>
#endif
#include <arcane/utils/Buffer.h>
#include <arcane/IItemFamily.h>
#include <arcane/ItemGroupRangeIterator.h>

using namespace Arcane;

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

//! Macro de construction d'un nom d'objet
/*! Sert généralement à nommer des groupes pour ItemGroupBuilder */
#define IMPLICIT_NAME ItemGroupBuilder_cleanString(__FILE__ "__" TOSTRING(__LINE__),false)
#define IMPLICIT_UNIQ_NAME ItemGroupBuilder_cleanString(__FILE__ "__" TOSTRING(__LINE__),true)


inline String ItemGroupBuilder_cleanString(const char * origStr, const bool isUniq) {
  static int cmpt = 0;
  const int len = std::strlen(origStr);
  char * str = new char[len+1];
  for(int i=0;i<len;++i)
    {
      if (std::isalnum(origStr[i]))
        str[i] = origStr[i];
      else
        str[i] = '_';
    }
#if (ARCANE_VERSION<10600)
  String newString(str,len);
#else
  StringBuilder newString(str,len) ;
#endif
  newString += "_";
  if (isUniq) newString += cmpt++;
  delete[] str;
#if (ARCANE_VERSION<10600)
  return newString;
#else
  return newString.toString() ;
#endif
}


/* \brief Outil de construction assisté de groupe
 *
 * L'unicité des éléments du groupe est garantie par construction. Il
 * est possible d'utiliser la macro IMPLICIT_NAME pour nommer nom de
 * groupe.
 */
template<typename T>
class ItemGroupBuilder {
 private:
  IMesh * m_mesh;
  std::set<Integer> m_ids;
  String m_group_name;
  
 public:
  //! Constructeur
  ItemGroupBuilder(IMesh * mesh, const String groupName)
    : m_mesh(mesh),
    m_group_name(groupName)
    {
      ;
    }

  //! Destructeur
  virtual ~ItemGroupBuilder()
    {
      ;
    }

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
      BufferT<Int32> localIds(m_ids.size());

      std::set<Integer>::const_iterator is = m_ids.begin();
      Integer i = 0;

      while(is != m_ids.end())
        {
          localIds[i] = *is;
          ++is; ++i;
        }
    
      ItemGroup newGroup = m_mesh->itemFamily(ItemTraitsT<T>::kind())->findGroup(m_group_name,true);

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
