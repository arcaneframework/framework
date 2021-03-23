// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include "Utils/ItemTools.h"
#include <arcane/IVariable.h>
#include <arcane/IItemFamily.h>
#include <arcane/utils/FatalErrorException.h>
#include <arcane/ItemGroup.h>
#include <arcane/utils/Array.h>

/*---------------------------------------------------------------------------*/

Item itemFromLocalId(IVariable * ivar, Integer localId)
{
  IItemFamily * itemFamily = ivar->itemFamily();
  ItemInternalList list = itemFamily->itemsInternal();
  return list[localId];
}

/*---------------------------------------------------------------------------*/

ItemVector BinaryGroupOperations:: And(ItemGroup a, ItemGroup b)
{
  if (a.itemFamily() != b.itemFamily())
    throw FatalErrorException("Cannot compare incompatible groups");

  IItemFamily * item_family = a.itemFamily();
  ItemGroup groupMin;
  ItemGroup groupMax;
  if (a.size() > b.size())
    {
      groupMin = b;
      groupMax = a;
    }
  else
    {
      groupMin = a;
      groupMax = b;
    }

  BufferT<Int32> lids;
  lids.reserve(groupMin.size());

  // on pourrait aussi utiliser un set,
  // mais attention au performances.
  // Sur groupMin ca devrait aller, car c'est le plus petit
  BoolArray groupMinSet(item_family->maxLocalId(),false);
  ENUMERATE_ITEM(iitem,groupMin)
  {
    groupMinSet[iitem.localId()] = true;
  }

  ENUMERATE_ITEM(iitem,groupMax)
  {
    const Integer localId = iitem.localId();
    if (groupMinSet[localId])
      lids.add(localId);
  }
  
  return ItemVector(item_family, lids);
}

/*---------------------------------------------------------------------------*/

ItemVector BinaryGroupOperations::Or(ItemGroup a, ItemGroup b)
{
  if (a.itemFamily() != b.itemFamily())
    throw FatalErrorException("Cannot compare incompatible groups");

  IItemFamily * item_family = a.itemFamily();

  Array<Int32> lids;
  lids.reserve(a.size() + b.size());

  BoolArray aSet(item_family->maxLocalId(), false);
  ENUMERATE_ITEM(iitem, a) {
    const Integer localId = iitem.localId();
    lids.add(localId);
    aSet[localId] = true;
  }

  ENUMERATE_ITEM(iitem, b) {
    const Integer localId = iitem.localId();
    if (not aSet[localId])
      lids.add(localId);
  }

  return ItemVector(item_family, lids);
}

/*---------------------------------------------------------------------------*/

ItemVector BinaryGroupOperations::Substract(ItemGroup a, ItemGroup b)
{
  if (a.itemFamily() != b.itemFamily())
    throw FatalErrorException("Cannot compare incompatible groups");

  IItemFamily * item_family = a.itemFamily();

  Array<Int32> lids;
  lids.reserve(a.size());

  BoolArray aSet(item_family->maxLocalId(), false);
  ENUMERATE_ITEM(iitem, b) {
    const Integer localId = iitem.localId();
    aSet[localId] = true;
  }

  ENUMERATE_ITEM(iitem, a) {
    const Integer localId = iitem.localId();
    if (not aSet[localId])
      lids.add(localId);
  }

  return ItemVector(item_family, lids);
}

/*---------------------------------------------------------------------------*/


ItemVector BinaryGroupOperations::Concatenate(ItemGroup a, ItemGroup b)
{
  if (a.itemFamily() != b.itemFamily())
    throw FatalErrorException("Cannot compare incompatible groups");

  IItemFamily * item_family = a.itemFamily();

  Array<Int32> lids;
  lids.reserve(a.size() + b.size());

  ENUMERATE_ITEM(iitem, a) {
    lids.add(iitem.localId());
  }

  ENUMERATE_ITEM(iitem, b) {
    lids.add(iitem.localId());
  }

  return ItemVector(item_family, lids);
}

