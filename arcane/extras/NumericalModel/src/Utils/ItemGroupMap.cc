// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//#include "arcane/geometry/ItemGroupMap.h"
#include "Utils/ItemGroupMap.h"

#include "arcane/IMesh.h"
#include "arcane/utils/ItemGroupObserver.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/ItemPrinter.h"
#include "arcane/ArcaneException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupMapAbstract::
ItemGroupMapAbstract()
  : HashTableBase(0,false)
  , m_group(ItemGroupImpl::checkSharedNull())
  , m_properties(0)
{
  ;
}

/*---------------------------------------------------------------------------*/

ItemGroupMapAbstract::
~ItemGroupMapAbstract()
{
  if (not m_group->null()) 
    {
      m_group->detachObserver(this);
    }
}

/*---------------------------------------------------------------------------*/

void
ItemGroupMapAbstract::
_init(const ItemGroup & group) 
{
  if (m_group != group.internal()) {
    _updateObservers(group.internal());
  }

  const Integer group_size = group.size();
  m_nb_bucket = this->nearestPrimeNumber(2*group_size);
  m_buckets.resize(m_nb_bucket);
  m_buckets.fill(-1);
  m_key_buffer.resize(group_size);
  m_next_buffer.resize(group_size);

  for(ItemEnumerator i(group.enumerator()) ; i.hasNext(); ++i)
    {
      const KeyTypeConstRef key = i.itemLocalId();
      const Integer bucket = _hash(key);
      const Integer lookup = _lookupBucket(bucket,key);
      ARCANE_ASSERT( (lookup < 0), ("Already assigned key"));
      m_key_buffer[i.index()] = key;
      m_next_buffer[i.index()] = m_buckets[bucket];
      m_buckets[bucket] = i.index();
    }
}

/*---------------------------------------------------------------------------*/

void
ItemGroupMapAbstract::
_updateObservers(ItemGroupImpl * group)
{
  if (not m_group->null())
    {
      m_group->detachObserver(this);
    }
  
  if (group)
    m_group = group;

  if (m_properties & eResetOnResize) 
    {
      m_group->attachObserver(this,newItemGroupObserverT(this,
                                                         &ItemGroupMapAbstract::_executeInvalidate));
    }
  else 
    {
      m_group->attachObserver(this,newItemGroupObserverT(this,
                                                         &ItemGroupMapAbstract::_executeExtend,
                                                         &ItemGroupMapAbstract::_executeReduce,
                                                         &ItemGroupMapAbstract::_executeCompact,
                                                         &ItemGroupMapAbstract::_executeInvalidate));
    }
}

/*---------------------------------------------------------------------------*/

void
ItemGroupMapAbstract::
setProperties(const Integer property) 
{
  m_properties |= property;
  _updateObservers();
}

/*---------------------------------------------------------------------------*/

void
ItemGroupMapAbstract::
unsetProperties(const Integer property) 
{
  m_properties &= ~property;
  _updateObservers();
}

/*---------------------------------------------------------------------------*/

ITraceMng *
ItemGroupMapAbstract::
traceMng() const 
{
  if (!m_group->null())
    return m_group->mesh()->traceMng();
  return NULL;
}

/*---------------------------------------------------------------------------*/

bool
ItemGroupMapAbstract::
checkSameGroup(const ItemGroup & group) const
{
  _checkGroupIntegrity();
  if (group.internal() == m_group)
    return true;
  if (group.size()       != m_group->size()     ||
      group.itemKind()   != m_group->itemKind() ||
      group.itemFamily() != m_group->itemFamily())
    return false;
  // Ici test sur les items mais meme taille, meme type, meme famille
  // On se fie aux ItemInternal's (meme les localIds peuvent suffir)
  const ItemInternalArrayView orgItems = m_group->itemsInternal();
  const ItemInternalArrayView testItems = group.internal()->itemsInternal();
  const Integer size = orgItems.size(); 
  for(Integer i=0;i<size;++i)
    if (orgItems[i] != testItems[i])
      return false;
  return true;
}

/*---------------------------------------------------------------------------*/

bool
ItemGroupMapAbstract::
checkSameGroup(const ItemVectorView & group) const
{
  _checkGroupIntegrity();
  if (group.size() != m_group->size()) {
    traceMng()->fatal() << "Bad sizes";
    return false;
  }
  // Ici test sur les items mais meme taille
  // On se fie donc aux ItemInternal's
  const ItemInternalArrayView orgItems = m_group->itemsInternal();
  const ItemInternalArrayView testItems = group.items();
  const Integer size = orgItems.size(); 
  for(Integer i=0;i<size;++i)
    if (orgItems[i] != testItems[i]) {
      traceMng()->fatal() << "Item " << i << " don't match : " << orgItems[i] << " vs " << testItems[i];
      return false;
    }
  return true;
}

/*---------------------------------------------------------------------------*/

String
ItemGroupMapAbstract::
name() const
{
  OStringStream oss;
  oss() << "ItemGroupMap(" << this << ") on " << group().name();
  return oss.str();
}

/*---------------------------------------------------------------------------*/

bool 
ItemGroupMapAbstract::
_checkGroupIntegrity() const {
  bool integrity = true;
  if (group().size() != m_key_buffer.size()) {
    traceMng()->fatal() << "ItemGroupMap BAD DATA size : group size=" << group().size() << " vs data size=" << m_key_buffer.size();
    integrity = false;
  }
  
  ENUMERATE_ITEM(iitem,group()) {
    const Integer hd = iitem.index();
    if (m_key_buffer[hd] != iitem.localId() || !_hasKey(iitem.localId())) {
      traceMng()->fatal() << "ItemGroupMap BAD DATA at " << iitem.index() << " :  lid=" << iitem.localId() << " vs key " << m_key_buffer[hd] << " chk:" << _hasKey(iitem.localId());
      integrity = false;
    }
  }
  return integrity;
}

/*---------------------------------------------------------------------------*/

Integer
ItemGroupMapAbstract::
_hash(KeyTypeConstRef id) const
{
  ARCANE_ASSERT((_initialized()),("ItemGroupMap not initialized"));
  return (Integer)(KeyTraitsType::hashFunction(id) % m_nb_bucket);
}

/*---------------------------------------------------------------------------*/

bool 
ItemGroupMapAbstract::
_hasKey(KeyTypeConstRef id) const
{
  const Integer hf = _hash(id);
  for( Integer i = m_buckets[hf]; i>=0 ; i=m_next_buffer[i] ) 
    {
      if (m_key_buffer[i]==id)
        return true;
    }
  return false;
}

/*---------------------------------------------------------------------------*/

Integer 
ItemGroupMapAbstract::
_lookupBucket(Integer bucket, KeyTypeConstRef id) const
{ 
  ARCANE_ASSERT((_initialized()),("ItemGroupMap not initialized"));
  for( Integer i=m_buckets[bucket]; i>=0 ; i=m_next_buffer[i] )
    {
      if (m_key_buffer[i]==id)
        return i;
    }
  return -1;
}

/*---------------------------------------------------------------------------*/

Integer 
ItemGroupMapAbstract::
_lookup(KeyTypeConstRef id) const
{ 
  return _lookupBucket(_hash(id),id);
}

/*---------------------------------------------------------------------------*/

bool 
ItemGroupMapAbstract::
_initialized() const
{
  return not m_buckets.empty();
}

/*---------------------------------------------------------------------------*/

void 
ItemGroupMapAbstract::
_throwItemNotFound(const TraceInfo & info, const Item & item) const throw()
{
  OStringStream oss;
  oss() << "Cannot find ItemGroupMap key " << ItemPrinter(item) << " for group " << group().name();
  throw InternalErrorException(info,oss.str());
}

/*---------------------------------------------------------------------------*/

void 
ItemGroupMapAbstract::
stats(std::ostream & o)
{
  Integer collision_count = 0;
  Array<bool> key_done(m_key_buffer.size());
  key_done.fill(false);
  for(Integer i=0; i<m_key_buffer.size() && !key_done[i] ; ++i)
    {
      for(Integer hd = i; m_next_buffer[hd]>=0 ; hd = m_next_buffer[hd])
        {
          key_done[hd] = true;
          o << "Collision in bucket " << i << " / " << m_nb_bucket << " between : " 
            << m_key_buffer[hd] << "(" << KeyTraitsType::hashFunction(m_key_buffer[hd]) << ") and "
            << m_next_buffer[m_key_buffer[hd]] << "(" << KeyTraitsType::hashFunction(m_next_buffer[m_key_buffer[hd]]) << ")\n";
          ++collision_count;
        }
    }
  o << "ItemGroupMapAbstractT::stats : Collision count = " << collision_count << " / " << m_key_buffer.size() << "\n";
}
